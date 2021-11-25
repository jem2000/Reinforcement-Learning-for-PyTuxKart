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
import argparse

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
ON_COLAB = os.environ.get('ON_COLAB', False)
if ON_COLAB:
    from .controller import rl_control
    from . import utils
else:
    from controller import rl_control
    import utils


def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
        
        self.hidden1 = nn.Linear(in_dim, 128)
        self.mu_layer = nn.Linear(128, out_dim)     
        self.log_std_layer = nn.Linear(128, out_dim)   
        
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
        
        self.hidden1 = nn.Linear(in_dim, 128)
        self.out = nn.Linear(128, 1)
        
        initialize_uniformly(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        value = self.out(x)
        
        return value



class A2CAgent:
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

    def __init__(self, pytux, track, gamma: float, entropy_weight: float, verbose=False, screen_width=128, screen_height=96):
        """Initialize."""
        self.env = pytux
        self.track = track
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)
        
        # networks
        obs_dim = 3
        action_dim = 1
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)
        
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
        self.restart_buffer = 0

    def save_model(self, model):
        from os import path
        if isinstance(model, Actor):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'actor.th'))
        raise ValueError("model type '%s' not supported!" % str(type(model)))
    
    def load_model(self):
        from torch import load
        from os import path
        r = Actor(3, 1)
        r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'actor.th'), map_location='cpu'))
        return r
        
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

        state.update()
        track.update()
        kart = state.players[0].kart
        off_track = abs(aim_point[0]) > 0.95
        end_track = np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3)
        if off_track:
            if (self.restart_buffer > 30):
                self.restart_buffer = 0
                if self.verbose:
                    print('off_track restarted ****************')
                restarted = True
            else:
                self.restart_buffer += 1
        elif end_track:
            if self.verbose:
                print('end_track restarted ****************')
            restarted = True
            done = True
        
        cur_loc = kart.distance_down_track
        # print('distance down track: ', cur_loc)
        steer_threshold = 0.35
        speed_threshold = 1
        reward_off = 0 if (off_track < steer_threshold) else (off_track - steer_threshold) * 5
        reward_speed = 0 if (cur_loc - prev_loc > speed_threshold) else (cur_loc - prev_loc - speed_threshold) * 5
        reward = reward_speed - reward_off

        return cur_loc, reward, restarted, done
    
    def update_kart(self, track, state):
        kart = state.players[0].kart
        cur_loc = kart.distance_down_track

        proj = np.array(state.players[0].camera.projection).T
        view = np.array(state.players[0].camera.view).T

        aim_point_world = self.env._point_on_track(cur_loc+TRACK_OFFSET, track)
        aim_point_image = self.env._to_image(aim_point_world, proj, view)
        current_vel = (np.linalg.norm(kart.velocity)-10)/10

        return aim_point_image, current_vel, aim_point_world, proj, view, kart

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
    
    def init_track(self, track = 'lighthouse'):
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

    def train(self, num_frames: int, plotting_interval: int = 300):
        self.is_test = False

        actor_losses, critic_losses, scores = [], [], []
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
            aim_point, vel, _, _, _, _ = self.update_kart(track, state)
            obs = np.array((aim_point[0], aim_point[1], vel))

            steer = self.select_action(obs)
            action = rl_control(aim_point, vel, 'steer', steer)
            prev_loc, reward, restarted, done = self.step(state, track, prev_loc, action, aim_point)

            next_aim_point, next_vel, aim_point_world, proj, view, kart = self.update_kart(track, state)
            next_obs = np.array((next_aim_point[0], next_aim_point[1], next_vel))

            self.transition.extend([next_obs, reward, done])
            
            actor_loss, critic_loss = self.update_model()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            score += reward
            
            if vel < 1.0 and self.total_step - last_rescue > RESCUE_TIMEOUT:
                last_rescue = self.total_step
                restarted = True
            
            # if episode ends
            if restarted:
                state, track = self.init_track()
                state.update()
                track.update()
                prev_loc = 0

            if done:
                best_score = score if score > best_score else best_score
                scores.append(score)
                score = 0

            if self.verbose:
                title = "Time frame: {}; Score: {:.2f}; Best score: {:.2f}".format(self.total_step, score, best_score)
                ax.clear()
                ax.set_title(title)
                ax.imshow(self.env.k.render_data[0].image)
                WH2 = np.array([self.env.config.screen_width, self.env.config.screen_height]) / 2
                ax.add_artist(plt.Circle(WH2*(1+self.env._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
                ax.add_artist(plt.Circle(WH2*(1+self.env._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
                plt.pause(1e-3)
                print('observation: ', obs)
                print('steering: ', steer)
                print('time frame: ', self.total_step)
                print('score: ', score, 'reward: ', reward)


            if self.total_step % plotting_interval == 0:
                best_score = -9999999999999
                self.save_model(self.actor)
                if ON_COLAB:
                    self._plot(self.total_step, scores, actor_losses, critic_losses)
                elif not self.verbose:
                    self._plot_cmd(self.total_step, scores, actor_losses, critic_losses, best_score)

        self.env.close()
    
    def test(self, max_frame):
        """Test the agent."""
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

        test_actor = self.load_model().eval()

        if self.verbose:
            # show video
            fig, ax = plt.subplots(1, 1)
        
        for cur_frame in range(max_frame):
            if count >= 9:
                break
            aim_point, vel, _, _, _, _ = self.update_kart(track, state)
            obs = np.array((aim_point[0], aim_point[1], vel))

            steer = self.select_action(obs, test_actor)
            action = rl_control(aim_point, vel, 'steer', steer)
            prev_loc, reward, restarted, done = self.step(state, track, prev_loc, action, aim_point)

            next_aim_point, next_vel, aim_point_world, proj, view, kart = self.update_kart(track, state)
            next_obs = np.array((next_aim_point[0], next_aim_point[1], next_vel))

            score += reward

            if vel < 1.0 and cur_frame - last_rescue > RESCUE_TIMEOUT:
                last_rescue = cur_frame
                restarted = True
            
            # if episode ends
            if restarted:
                state, track = self.init_track()
                state.update()
                track.update()
                count += 1
                print('count: ', count)
                prev_loc = 0
                cur_frame = 0

            if done:
                best_score = score if score > best_score else best_score
                score = 0

            if self.verbose:
                title = "Time frame: {}; Score: {:.2f}; Best score: {:.2f}; Attempt: {}".format(cur_frame, score, best_score, count+1)
                ax.clear()
                ax.set_title(title)
                ax.imshow(self.env.k.render_data[0].image)
                WH2 = np.array([self.env.config.screen_width, self.env.config.screen_height]) / 2
                ax.add_artist(plt.Circle(WH2*(1+self.env._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
                ax.add_artist(plt.Circle(WH2*(1+self.env._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
                plt.pause(1e-3)
                print('observation: ', obs)
                print('steering: ', steer)
                print('time frame: ', cur_frame)
                print('score: ', score, 'reward: ', reward)
        
        # print("score: ", score)
        self.env.close()

    def _plot_cmd(
        self, 
        frame_idx: int, 
        scores: List[float], 
        actor_losses: List[float], 
        critic_losses: List[float], 
        best_score: float, 
    ):
        print('==========================')
        print('frame: ', frame_idx)
        print('scores: ', scores[-1])
        print('actor losses: ', actor_losses[-1])
        print('critic losses: ', critic_losses[-1])
        print('best score: ', best_score)
    
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

def main(pytux, track, verbose, test):
    num_frames = 1000000
    gamma = 0.9
    entropy_weight = 1e-2

    agent = A2CAgent(pytux, track, gamma, entropy_weight, verbose=verbose)
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
    args = parser.parse_args()

    print(args)

    pytux = utils.PyTux()
    agent = A2CAgent(
        pytux=pytux,
        track=args.track,
        gamma=gamma,
        entropy_weight=entropy_weight,
        verbose=args.verbose
    )

    if args.test:
        agent.test(num_frames)
    else:
        agent.train(num_frames)