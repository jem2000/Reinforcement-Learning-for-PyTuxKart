import random
from collections import deque
from typing import List, Tuple, Deque, Dict

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pystk
from IPython.display import clear_output
from torch.distributions import Normal
from torch import save
import os
from torch import load
from os import path
import argparse

if IN_COLAB:  # for colab
    import base64
    import glob
    import io
    import os

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

    list_of_files = glob.glob("videos/*.mp4")
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    ipython_show_video(latest_file)

else:  # for jupyter
    from matplotlib import animation
    from JSAnimation.IPython_display import display_animation
    from IPython.display import display


    def display_frames_as_gif(frames):
        """Displays a list of frames as a gif, with controls."""
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(
            plt.gcf(), animate, frames = len(frames), interval=50
        )
        display(display_animation(anim, default_mode='loop'))


    # display
    display_frames_as_gif(frames)


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
            rewards[step]
            + gamma * values[step + 1] * masks[step]
            - values[step]
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


class PPOAgent:
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
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(
            self,
            env: pytux,
            track: track,
            batch_size: int,
            gamma: float,
            tau: float,
            epsilon: float,
            epoch: int,
            rollout_len: int,
            entropy_weight: float,
            verbose = False,
            continue_training = False
    ):
        """Initialize."""
        self.env = env
        self.track = track
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch = epoch
        self.rollout_len = rollout_len
        self.entropy_weight = entropy_weight

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks
        obs_dim = 4
        action_dim = 1
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

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

    def save_model(self, model):
        if isinstance(model, Actor):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'ppo.th'))
        raise ValueError("model type '%s' not supported!" % str(type(model)))

    def load_model(self):
        r = Actor(self.obs_dim, self.action_dim).to(self.device)
        r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'ppo.th'), map_location='cpu'))
        return r

    def select_action(self, obs: np.ndarray, test_actor=None) -> np.ndarray:
        """Select an action from the input state."""
        obs = torch.FloatTensor(state).to(self.device)
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

        # return selected_action.cpu().detach().numpy()
        return selected_action.clamp(-1.0, 1.0).cpu().detach().numpy()

    def step(self, state, track, prev_loc, action, aim_point):
        """Take an action and return the response of the env."""
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
            if (self.total_step - self.restart_time > 30):
                self.restart_time = self.total_step
                if self.verbose:
                    print('off_track restarted ****************')
                restarted = True
        elif end_track:
            if self.verbose:
                print('end_track restarted ****************')
            restarted = True
            done = True

        cur_loc = kart.distance_down_track
        # print('distance down track: ', cur_loc)
        speed_threshold = 0.5
        abs_aim = abs(aim_point[0])
        loc_change = (cur_loc - prev_loc) if abs(cur_loc - prev_loc) < 2.5 else 0
        # print('off: ', abs_aim)
        # print('loc: ', cur_loc - prev_loc, 'cur_loc: ', cur_loc, 'prev_loc', prev_loc)
        reward_off = - abs_aim * 30  # range (-25, 0)
        reward_speed = 0 if (loc_change - speed_threshold) > 0 else (loc_change - speed_threshold) * 10  # range(-5, 0)
        reward = reward_speed + reward_off

        if (self.total_step - self.restart_time < 5):
            reward = -30
        return cur_loc, reward / 2, restarted, done

    def update_kart(self, track, state):
        kart = state.players[0].kart
        cur_loc = kart.distance_down_track

        proj = np.array(state.players[0].camera.projection).T
        view = np.array(state.players[0].camera.view).T

        aim_point_world = self.env._point_on_track(cur_loc + TRACK_OFFSET, track)
        aim_point_image = self.env._to_image(aim_point_world, proj, view)
        current_vel = (np.linalg.norm(kart.velocity) - 10) / 10

        return aim_point_image, current_vel, aim_point_world, proj, view, kart

    def update_model(
            self, next_obs: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        states = torch.cat(self.states).view(-1, 3)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values

        actor_losses, critic_losses = [], []

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
            clipped_surr_loss = (
                    torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
            )

            # entropy
            entropy = dist.entropy().mean()

            actor_loss = (
                    -torch.min(surr_loss, clipped_surr_loss).mean()
                    - entropy * self.entropy_weight
            )

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

        return actor_loss, critic_loss

    def init_track(self, track='lighthouse'):
        if self.env.k is not None and self.env.k.config.track == track:
            # print('init restart +++++++++++++++++++++')
            self.env.k.restart()
            self.env.k.step()
        else:
            if self.env.k is not None:
                # print('init start +++++++++++++++++++++')
                self.env.k.stop()
                del self.env.k
            config = pystk.RaceConfig(num_kart=1, laps=1, track=track)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

            self.env.k = pystk.Race(config)
            self.env.k.start()
            self.env.k.step()

        return pystk.WorldState(), pystk.Track()

    def train(self, num_frames: int, plotting_interval: int = 2500):
        """Train the agent."""
        self.is_test = False

        state = self.env.reset()
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        actor_epoch_lossess, critic_epoch_losses = [], []
        scores = []
        score = 0
        prev_loc = 0
        best_score = -9999999999999
        last_rescue = 0

        if self.verbose:
            # show video
            fig, ax = plt.subplots(1, 1)

        while self.total_step <= num_frames + 1:
            for _ in range(self.rollout_len):
                self.total_step += 1
                aim_point, vel, _, _, _, kart = self.update_kart(track, state)
                obs = np.array((aim_point[0], aim_point[1], vel, kart.distance_down_track))
                steer = self.select_action(obs)
                action = rl_control(aim_point, vel, 'steer', steer)
                prev_loc, reward, restarted, done = self.step(state, track, prev_loc, action, aim_point)

                next_aim_point, next_vel, aim_point_world, proj, view, kart = self.update_kart(track, state)
                next_obs = np.array((next_aim_point[0], next_aim_point[1], next_vel, kart.distance_down_track))

                state = next_state
                score += reward[0][0]

                # if episode ends
                if done[0][0]:
                    state = env.reset()
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    score = 0

                    self._plot(
                        self.total_step, scores, actor_losses, critic_losses
                    )

            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        # termination
        self.env.close()

    def test(self):
        """Test the agent."""
        self.is_test = True

        state = self.env.reset()
        done = False
        score = 0

        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        return frames

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
            (131, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores),
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