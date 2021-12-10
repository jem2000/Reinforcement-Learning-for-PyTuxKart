import random
from typing import List, Tuple

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

RESCUE_SPEED = 0.5
RESCUE_TIMEOUT = 10
TRACK_OFFSET = 15

starting_frames = 0
reset_frames = 0

ON_COLAB = os.environ.get('ON_COLAB', False)
if ON_COLAB:
    from .controller import rl_control, control
    from .pytux_utils import DeepRL
    from . import utils
    from . import planner

else:
    from controller import rl_control, control
    from pytux_utils import DeepRL
    import utils
    import planner


def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 32)
        self.mu_layer = nn.Linear(32, out_dim)
        self.log_std_layer = nn.Linear(32, out_dim)

        initialize_uniformly(self.mu_layer)
        initialize_uniformly(self.log_std_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))

        mu = torch.tanh(self.mu_layer(x))
        log_std = F.softplus(self.log_std_layer(x))
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 64)
        self.out = nn.Linear(64, 1)

        initialize_uniformly(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        value = self.out(x)

        return value


class A2CAgent(DeepRL):
    """A2CAgent interacting with environment.
        
    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(self, pytux, track, planner: None, gamma: float, entropy_weight: float, verbose=False, continue_training=False,
                 screen_width=128, screen_height=96):
        super().__init__(pytux, track, planner, gamma, entropy_weight)
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        if continue_training:
            self.actor.load_state_dict(
                load(path.join(path.dirname(path.abspath(__file__)), 'A2C.th'), map_location='cpu'))
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # transition (obs, log_prob, next_obs, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

        # verbose
        self.verbose = verbose

        # off track
        self.restart_time = 0

        self.off_track_tolerance = 0

    def select_action(self, obs: np.ndarray, test_actor=None) -> np.ndarray:
        """Select an action from the input observation."""
        obs = torch.FloatTensor(obs).to(self.device)
        if self.is_test:
            action, dist = test_actor(obs)
        else:
            action, dist = self.actor(obs)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [obs, log_prob]

        return selected_action.clamp(-1.0, 1.0).cpu().detach().numpy()

    def step(self, state, track, prev_loc, action, aim_point):
        restarted = False
        done = False
        self.env.k.step(action)

        global starting_frames
        starting_frames += 1

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
                if self.verbose:
                    print('off_track restarted ****************')
                restarted = True
        elif end_track:
            if self.verbose:
                print('end_track restarted ****************')
            restarted = True
            done = True
            starting_frames = 0

        cur_loc = kart.distance_down_track

        # print('distance down track: ', cur_loc)
        speed_threshold = 1
        abs_aim = abs(aim_point[0])
        loc_change = (cur_loc - prev_loc) if abs(cur_loc - prev_loc) < 2.5 else 0
        forward_distance = cur_loc - prev_loc

        # print('off: ', abs_aim)
        # print('loc: ', cur_loc - prev_loc, 'cur_loc: ', cur_loc, 'prev_loc', prev_loc)
        reward_off = - abs_aim * 30  # range (-25, 0)
        reward_speed = forward_distance * 10 if (loc_change - speed_threshold) > 0 else (loc_change - speed_threshold) * 10  # range(-5, 0)
        reward = reward_speed + reward_off

        if self.total_step - self.restart_time < 5:
            reward = -15
        return cur_loc, reward / 2, restarted, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        obs, log_prob, next_obs, reward, done = self.transition

        # Q_t   = r + gamma * V(s_{t+1})  if obs != Terminal
        #       = r                       otherwise
        mask = 1 - done
        # print('==============================')
        # print('state: ', obs)
        # print( 'log_prob: ', log_prob)
        # print('next_state: ', next_obs)
        # print('reward: ', reward)
        # print('done: ', done)
        # print('mask: ', mask)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        pred_value = self.critic(obs)
        targ_value = reward + self.gamma * self.critic(next_obs) * mask
        value_loss = F.smooth_l1_loss(pred_value, targ_value.detach())

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # advantage = Q_t - V(s_t)
        advantage = (targ_value - pred_value).detach()  # not backpropagated
        policy_loss = -advantage * log_prob
        policy_loss += self.entropy_weight * -log_prob  # entropy maximization

        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # print('policy_loss: ', policy_loss)
        # print('value_loss: ', value_loss)

        return policy_loss.item(), value_loss.item()

    def train(self, num_frames: int, plotting_interval: int = 2500):
        global starting_frames
        global reset_frames
        self.is_test = False

        actor_losses, critic_losses, scores = [], [], []
        actor_epoch_losses, critic_epoch_losses = [], []
        score = 0
        prev_loc = 0
        best_score = -9999999999999
        last_rescue = 0

        state, track = self.init_track()
        state.update()
        track.update()

        if self.verbose:
            # show video
            fig, ax = plt.subplots(1, 1)

        for self.total_step in range(1, num_frames + 1):
            aim_point, vel, _, _, _, kart = self.update_kart(track, state)
            obs = np.array((aim_point[0], aim_point[1], vel, kart.distance_down_track))
            steer = self.select_action(obs)

            if aim_point[0] > 0 and steer < 0:
                steer = -steer
            elif aim_point[0] < 0 and steer > 0:
                steer = -steer

            accel = self.select_action(obs)
            accel = ((accel + 1) / 2) + 0.01

            if starting_frames < TRACK_OFFSET:
                # print("Using controller")
                action = control(aim_point, vel)
            else:
                # print("Using RL")
                action = rl_control(aim_point, vel, ['steer', 'acceleration'], [steer, accel])
                # action = rl_control(aim_point, vel, ['steer'], [steer])

            print("steer: ", action.steer)
            print("accel", action.acceleration)

            rescue = False
            # print("vel is ", vel, ", total step is ", self.total_step, ", last rescue is ", last_rescue)
            if vel < RESCUE_SPEED:
                reset_frames += 1
            else:
                reset_frames = 0
                # print("total_step is", self.total_step)
            if reset_frames > RESCUE_TIMEOUT and self.total_step - last_rescue > RESCUE_TIMEOUT:
                last_rescue = self.total_step
                action.rescue = True
                rescue = True
                starting_frames = 0

            prev_loc, reward, restarted, done = self.step(state, track, prev_loc, action, aim_point)
            if rescue:
                reward = -100

            next_aim_point, next_vel, aim_point_world, proj, view, kart = self.update_kart(track, state)
            next_obs = np.array((next_aim_point[0], next_aim_point[1], next_vel, kart.distance_down_track))

            self.transition.extend([next_obs, reward, done])

            actor_loss, critic_loss = self.update_model()

            actor_epoch_losses.append(actor_loss)
            critic_epoch_losses.append(critic_loss)

            if self.total_step % 200 == 0:
                actor_losses.append(np.mean(actor_epoch_losses))
                critic_losses.append(np.mean(critic_epoch_losses))
                actor_epoch_losses, critic_epoch_losses = [], []

            score += reward



            # if episode ends
            if restarted:
                state, track = self.init_track()
                state.update()
                track.update()
                prev_loc = 0
                starting_frames = 0

            if done:
                best_score = score if score > best_score else best_score
                scores.append(score)
                score = 0

            if self.verbose:
                title = "Time frame: {}; Score: {:.2f}; Best score: {:.2f}".format(self.total_step, score, best_score)
                DeepRL.verbose(self, title, kart, ax, proj, view, aim_point_world, track, next_aim_point)
                print('observation: ', obs)
                print('steering: ', steer)
                print('time frame: ', self.total_step)
                print('score: ', score, 'reward: ', reward)

            if self.total_step % plotting_interval == 0:
                # best_score = -9999999999999
                self.save_model(self.actor, Actor, 'A2C.th')
                if ON_COLAB:
                    self._plot(self.total_step, best_score, actor_losses, critic_losses)
                elif not self.verbose:
                    self._plot_cmd(self.total_step, best_score, actor_losses, critic_losses)

        self.env.close()

    def test(self, max_frame):
        """Test the agent."""
        global starting_frames
        global reset_frames
        self.is_test = True
        print('Testing')

        score = 0
        prev_loc = 0
        best_score = -9999999999999
        last_rescue = 0
        count = 0

        state, track = self.init_track()
        state.update()
        track.update()

        test_actor = self.load_model(Actor, 'A2C.th').eval()

        if self.verbose:
            # show video
            fig, ax = plt.subplots(1, 1)

        for cur_frame in range(max_frame):
            if count >= 9:
                break
            aim_point, vel, _, _, _, kart = self.update_kart(track, state)
            obs = np.array((aim_point[0], aim_point[1], vel, kart.distance_down_track))

            steer = self.select_action(obs, test_actor)

            if aim_point[0] > 0 and steer < 0:
                steer = -steer
            elif aim_point[0] < 0 and steer > 0:
                steer = -steer

            accel = self.select_action(obs, test_actor)
            accel = ((accel + 1) / 2) + 0.01

            if starting_frames < TRACK_OFFSET:
                action = control(aim_point, vel)
            else:
                action = rl_control(aim_point, vel, ['steer', 'acceleration'], [steer, accel])
            prev_loc, reward, restarted, done = self.step(state, track, prev_loc, action, aim_point)
            print("steer: ", action.steer)
            print("accel", action.acceleration)
            next_aim_point, next_vel, aim_point_world, proj, view, kart = self.update_kart(track, state)
            # next_obs = np.array((next_aim_point[0], next_aim_point[1], next_vel, kart.distance_down_track))

            score += reward

            # print("vel is ", vel, ", cur_frame is ", cur_frame, ", last rescue is ", last_rescue)
            if vel < RESCUE_SPEED:
                reset_frames += 1
            else:
                reset_frames = 0
            if reset_frames > RESCUE_TIMEOUT and cur_frame - last_rescue > RESCUE_TIMEOUT:
                last_rescue = cur_frame
                action.rescue = True
                rescue = True
                starting_frames = 0

            # if episode ends
            if restarted:
                state, track = self.init_track()
                state.update()
                track.update()
                count += 1
                prev_loc = 0
                cur_frame = 0
                starting_frames = 0

            if done:
                best_score = score if score > best_score else best_score
                score = 0

            if self.verbose:
                title = "Time frame: {}; Score: {:.2f}; Attempt: {}".format(cur_frame, score, count + 1)
                DeepRL.verbose(self, title, kart, ax, proj, view, aim_point_world)
                print('observation: ', obs)
                print('steering: ', steer)
                print('time frame: ', cur_frame)
                print('score: ', score, 'reward: ', reward)

        # print("score: ", score)
        self.env.close()

    def _plot(
            self,
            frame_idx: int,
            score: float,
            actor_losses: List[float],
            critic_losses: List[float],
    ):
        """Plot the training progresses."""

        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)

        subplot_params = [
            (131, f"frame {frame_idx}. score: {score}", score),
            (132, "actor_loss", actor_losses),
            (133, "critic_loss", critic_losses),
        ]

        clear_output(True)
        plt.figure(figsize=(30, 5))
        for loc, title, values in subplot_params:
            subplot(loc, title, values)
        plt.show()


def main(pytux, track, verbose=False, test=False, continue_training=False):
    num_frames = 1000000
    gamma = 0.9
    entropy_weight = 1e-2

    agent = A2CAgent(
        pytux=pytux,
        track=track,
        gamma=gamma,
        planner=planner.load_model().eval(),
        entropy_weight=entropy_weight,
        verbose=verbose,
        continue_training=continue_training
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

    print(args)

    pytux = utils.PyTux()
    agent = A2CAgent(
        pytux=pytux,
        track=args.track,
        planner=planner.load_model().eval(),
        gamma=gamma,
        entropy_weight=entropy_weight,
        verbose=args.verbose,
        continue_training=args.continue_training
    )

    if args.test:
        agent.test(num_frames)
    else:
        agent.train(num_frames)
