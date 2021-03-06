import base64
import io
from collections import deque
from typing import List, Tuple, Deque

import os
import gym
import argparse
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import load
from os import path
from IPython.display import clear_output
from torch.distributions import Normal

RESCUE_SPEED = 3
RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15

starting_frames = 0
reset_frames = 0

ON_COLAB = os.environ.get('ON_COLAB', False)
if ON_COLAB:
    from .controller import rl_control
    from .controller import control
    from . import utils
    from . import planner
    from .pytux_utils import DeepRL
else:
    from controller import rl_control
    from controller import control
    import utils
    import planner
    from pytux_utils import DeepRL
    from IPython.display import HTML, display


def ipython_show_video(path: str) -> None:
    """Show a video at `path` within IPython Notebook."""
    if not os.path.isfile(path):
        raise NameError("Cannot access: {}".format(path))

    video = io.open(path, "r+b").read()
    encoded = base64.b64encode(video)

    display(HTML(
        data="""
        <video alt="test" controls>
        <source src="data:video/mp4;base64,{0}" type="video/mp4"/>
        </video>
        """.format(encoded.decode("ascii"))
    ))

    # list_of_files = glob.glob("videos/*.mp4")
    # latest_file = max(list_of_files, key=os.path.getctime)
#     # print(latest_file)
    # ipython_show_video(latest_file)

    # def display_frames_as_gif(frames):
    #     """Displays a list of frames as a gif, with controls."""
    #     patch = plt.imshow(frames[0])
    #     plt.axis('off')
    #
    #     def animate(i):
    #         patch.set_data(frames[i])
    #
    #     anim = animation.FuncAnimation(
    #         plt.gcf(), animate, frames=len(frames), interval=50
    #     )
    #     display(display_animation(anim, default_mode='loop'))
    #
    #
    # # display
    # display_frames_as_gif(frames)


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class Actor(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        log_std_min: int = -20,
        log_std_max: int = 0,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden = nn.Linear(in_dim, 32)

        self.mu_layer = nn.Linear(32, out_dim)
        self.mu_layer = init_layer_uniform(self.mu_layer)

        self.log_std_layer = nn.Linear(32, out_dim)
        self.log_std_layer = init_layer_uniform(self.log_std_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden(state))
        
        mu = torch.tanh(self.mu_layer(x))
        log_std = torch.tanh(self.log_std_layer(x))
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        self.hidden = nn.Linear(in_dim, 64)
        self.out = nn.Linear(64, 1)
        self.out = init_layer_uniform(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden(state))
        value = self.out(x)

        return value


def compute_gae(
        next_value: list,
        rewards: list,
        masks: list,
        values: list,
        gamma: float,
        tau: float
) -> List:
    """Compute gae."""
    values = values + [next_value]
    gae = 0
    returns: Deque[float] = deque()

    for step in reversed(range(len(rewards))):
        delta = (
                rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        )
        gae = delta + gamma * tau * masks[step] * gae
        returns.appendleft(gae + values[step])

    return list(returns)


def ppo_iter(
        epoch: int,
        mini_batch_size: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
):
    """Yield mini-batches."""
    batch_size = states.size(0)
    for _ in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[
                rand_ids
            ], log_probs[rand_ids], returns[rand_ids], advantages[rand_ids]


class PPOAgent(DeepRL):
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporary storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(self, pytux, track, batch_size: int, planner: None, gamma: float, tau: float, epsilon: float,
    epoch: int, rollout_len: int, entropy_weight: float, verbose=False, continue_training=False):
        """Initialize."""
        super().__init__(pytux, track, planner, gamma, entropy_weight)
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch = epoch
        self.rollout_len = rollout_len

        # networks
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        if continue_training:
            print('continue training')
            self.actor.load_state_dict(
                load(path.join(path.dirname(path.abspath(__file__)), 'ppo.th'), map_location='cpu'))
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.005)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

        self.verbose = verbose

        # off track
        self.last_rescue = 0
        self.restart_time = 0

        self.off_track_tolerance = 0

    def select_action(self, obs: np.ndarray, test_actor=None) -> np.ndarray:
        """Select an action from the input state."""
        obs = torch.FloatTensor(obs).to(self.device)
        # action, dist = self.actor(state)

        if self.is_test:
            action, dist = test_actor(obs)
        else:
            action, dist = self.actor(obs)

        selected_action = dist.mean if self.is_test else action
        
        if not self.is_test:
            value = self.critic(obs)
            self.states.append(obs)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action))

        return selected_action.cpu().detach().numpy()

    def step(self, state, track, prev_loc, action, aim_point):
        """Take an action and return the response of the env."""
        global starting_frames
        starting_frames += 1

        restarted = False
        done = False
        self.env.k.step(action)

        state.update()
        track.update()
        kart = state.players[0].kart
        off_track = False
        if abs(aim_point[0]) > 0.97:
            if self.off_track_tolerance < 20:
                self.off_track_tolerance += 1
            else:
                off_track = True
                self.off_track_tolerance = 0

        end_track = np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3)
        if off_track:
            if self.total_step - self.restart_time > 30:
                self.restart_time = self.total_step
                # if self.verbose:
                #     # print('off_track restarted ****************')
                action.rescue = True

        elif end_track:
            print('end_track restarted ****************')
            restarted = True
            done = True
            starting_frames = 0
            print("completed track")

        speed_weight = 0.5
        speed_threshold = 0.5

        cur_loc = kart.distance_down_track
        abs_aim = abs(aim_point[0])
        loc_change = (cur_loc - prev_loc) if abs(cur_loc - prev_loc) < 2.5 else 0

        reward_off = - abs_aim * 15  # range (-20, 0)
        reward_speed = 0 if (loc_change - speed_threshold) > 0 else (loc_change - speed_threshold) * 10 # range(-10, 0)
        reward = reward_speed*speed_weight + reward_off*(1-speed_weight)

        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        restarted = np.reshape(restarted, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - restarted).to(self.device))

        return cur_loc, reward, restarted, done

    def update_model(self, next_obs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines


        next_obs = torch.FloatTensor(next_obs).to(device)
        next_value = self.critic(next_obs)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, 4)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).view(-1, 1).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values

#         # print('values', values.shape)
#         # print('returns', returns.shape)
#         # print('advantages', advantages.shape)

        actor_losses, critic_losses = [], []

#         # print("states===================")
#         # print(states)
#         # print("returns===================")
#         # print(returns)

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
                epoch=self.epoch,
                mini_batch_size=self.batch_size,
                states=states,
                actions=actions,
                values=values,
                log_probs=log_probs,
                returns=returns,
                advantages=advantages,
        ):
            # calculate ratios

            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()
            
            # actor_loss
            surr_loss = ratio * adv
            clipped_surr_loss = (torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv)

            # entropy
            entropy = dist.entropy().mean()

            actor_loss = (
                    -torch.min(surr_loss, clipped_surr_loss).mean()
                    - entropy * self.entropy_weight)

            # critic_loss
            value = self.critic(state)
            # clipped_value = old_value + (value - old_value).clamp(-0.5, 0.5)
            critic_loss = (return_ - value).pow(2).mean()

            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

#         # print('actor_loss', actor_loss)
#         # print('critic_loss', critic_loss)

        return actor_loss, critic_loss

    def train(self, num_frames: int, plotting_interval: int = 2500):
        """Train the agent."""
        global starting_frames
        global reset_frames
        self.is_test = False
        self.last_rescue = 0

        actor_losses, critic_losses = [], []
        actor_epoch_losses, critic_epoch_losses = [], []
        scores = []
        score = 0
        prev_loc = 0
        best_score = -9999999999999
        

        state, track = self.init_track()
        state.update()
        track.update()

        if self.verbose:
            # show video
            fig, ax = plt.subplots(1, 1)

        while self.total_step <= num_frames + 1:
            for _ in range(self.rollout_len):
                self.total_step += 1
                aim_point, vel, _, _, _, kart = self.update_kart(track, state)
                obs = np.array((aim_point[0], aim_point[1], vel, kart.distance_down_track))
                steer = self.select_action(obs)
                # accel = self.select_action(obs)
                # print("starting frames is ", starting_frames)
                if aim_point[0] > 0 and steer < 0:
                    steer = -steer
                elif aim_point[0] < 0 and steer > 0:
                    steer = -steer
                # if starting_frames < TRACK_OFFSET:
                #     # print("Using controller")
                #     action = control(aim_point, vel)
                # else:
                    # print("Using RL")
                    # action = rl_control(aim_point, vel, ['steer', 'acceleration'], [steer, accel])
                action = rl_control(aim_point, vel, ['steer'], [steer])

                rescue = False
                # print("vel is ", vel)
                if vel < RESCUE_SPEED:
                    reset_frames += 1
                else:
                    reset_frames = 0
#                     # print("total_step is", self.total_step)
                if reset_frames > RESCUE_TIMEOUT and self.total_step - self.last_rescue > RESCUE_TIMEOUT:
                    self.last_rescue = self.total_step
                    action.rescue = True
                    rescue = True
                    starting_frames = 0

                prev_loc, reward, restarted, done = self.step(state, track, prev_loc, action, aim_point)
                if rescue:
                    reward = -15

                next_aim_point, next_vel, aim_point_world, proj, view, kart = self.update_kart(track, state)
                next_obs = np.array((next_aim_point[0], next_aim_point[1], next_vel, kart.distance_down_track))
                next_obs = np.reshape(next_obs, (1, -1)).astype(np.float64)
                # state = next_state
                score += reward

                if restarted:
                    state, track = self.init_track()
                    state.update()
                    track.update()
                    prev_loc = 0
                    starting_frames = 0

                # if episode ends
                if done:
                    print('done')
                    best_score = score if score > best_score else best_score
                    scores.append(score)
                    score = 0
                    # self._plot(self.total_step, scores, actor_losses, critic_losses)

                # print(self.total_step)

                if self.verbose:
                    title = "Time frame: {}; Score: {:.2f}; Best score: {:.2f}".format(self.total_step, score[0][0],
                                                                                       best_score)
                    DeepRL.verbose(self, title, kart, ax, proj, view, aim_point_world, track, next_aim_point)
                    # print('observation: ', obs)
                    # print('steering: ', steer)
                    # print('time frame: ', self.total_step)
                    # print('score: ', score, 'reward: ', reward)

            actor_loss, critic_loss = self.update_model(next_obs)
            actor_epoch_losses.append(actor_loss)
            critic_epoch_losses.append(critic_loss)

            actor_losses.append(np.mean(actor_epoch_losses))
            critic_losses.append(np.mean(critic_epoch_losses))
            actor_epoch_losses, critic_epoch_losses = [], []
            self.save_model(self.actor, Actor, 'ppo.th')
            s_in = scores if scores != [] else [best_score]
            if ON_COLAB:
              self._plot(self.total_step, s_in, actor_losses, critic_losses)
            elif not self.verbose:
              self._plot_cmd(self.total_step, s_in, actor_losses, critic_losses)

        # termination
        self.env.close()

    def test(self, max_frame):
        """Test the agent."""
        global starting_frames
        global reset_frames
        self.is_test = True
        self.last_rescue = 0
        # print('Testing')

        score = 0
        prev_loc = 0
        best_score = -9999999999999
        count = 0

        state, track = self.init_track()
        state.update()
        track.update()

        test_actor = self.load_model(Actor, 'ppo.th').eval()

        if self.verbose:
            # show video
            fig, ax = plt.subplots(1, 1)

        for cur_frame in range(max_frame):
            if count >= 9:
                break
            aim_point, vel, _, _, _, kart = self.update_kart(track, state)
            obs = np.array((aim_point[0], aim_point[1], vel, kart.distance_down_track))
            steer = self.select_action(obs, test_actor)
            accel = None
            # accel = self.select_action(obs, test_actor)
            # if starting_frames < TRACK_OFFSET:
            #     action = control(aim_point, vel)
            # else:
            if aim_point[0] > 0 and steer < 0:
                steer = -steer
                print("corrected")
            elif aim_point[0] < 0 and steer > 0:
                steer = -steer
                print("corrected")
            print("steer:", steer)
            action = rl_control(aim_point, vel, ['steer'], [steer])
                # action = rl_control(aim_point, vel, ['steer', 'acceleration'], [steer, accel])

            # print("vel is ", vel, ", cur frame is ", cur_frame, ", last rescue is ", self.last_rescue)
            if vel < RESCUE_SPEED:
                reset_frames += 1
            else:
                reset_frames = 0
            if reset_frames > RESCUE_TIMEOUT and cur_frame - self.last_rescue > RESCUE_TIMEOUT:
                self.last_rescue = cur_frame
                action.rescue = True
                rescue = True
                starting_frames = 0

            prev_loc, reward, restarted, done = self.step(state, track, prev_loc, action, aim_point)

            next_aim_point, next_vel, aim_point_world, proj, view, kart = self.update_kart(track, state)
            score += reward


            # if episode ends
            if restarted:
                state, track = self.init_track()
                state.update()
                track.update()
                count += 1
                prev_loc = 0
                cur_frame = 0

            if done:
                best_score = score if score > best_score else best_score
                score = 0

            if self.verbose:
                title = "Time frame: {}; Score: {:.2f}; Best score: {:.2f}".format(cur_frame, score[0][0],
                                                                                    count + 1)
                DeepRL.verbose(self, title, kart, ax, proj, view, aim_point_world)
                # print('observation: ', obs)
                # print('steering: ', steer)
                # print('time frame: ', cur_frame)
                # print('score: ', score, 'reward: ', reward)

        self.env.close()

    def _plot(
            self,
            frame_idx: int,
            scores: List[float],
            actor_losses: List[float],
            critic_losses: List[float],
    ):
        """Plot the training progresses."""

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"frame {frame_idx}. score: {scores[-1]}", scores),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        clear_output(True)
        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.show()


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action


def main(pytux, track, verbose=False, test=False, continue_training=False):
    num_frames = 1000000
    gamma = 0.9
    entropy_weight = 1e-2

    agent = PPOAgent(
        pytux=pytux,
        track=track,
        planner=planner.load_model().eval(),
        gamma=gamma,
        entropy_weight=entropy_weight,
        verbose=verbose,
        continue_training=continue_training,
        tau=0.8,
        batch_size=64,
        epsilon=0.2,
        epoch=64,
        rollout_len=2048,
        )

    if test:
        agent.test(num_frames)
    else:
        agent.train(num_frames)


if __name__ == '__main__':

    num_frames = 1000000
    gamma = 0.9
    entropy_weight = 1e-2

    parser = argparse.ArgumentParser()
    parser.add_argument('track')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-c', '--continue_training', action='store_true')
    args = parser.parse_args()

    # print(args)

    pytux = utils.PyTux()
    agent = PPOAgent(
        pytux=pytux,
        track=args.track,
        planner=planner.load_model().eval(),
        gamma=gamma,
        entropy_weight=entropy_weight,
        verbose=args.verbose,
        continue_training=args.continue_training,
        tau=0.8,
        batch_size=64,
        epsilon=0.2,
        epoch=64,
        rollout_len=2048,
    )

    if args.test:
        agent.test(num_frames)
    else:
        agent.train(num_frames)
