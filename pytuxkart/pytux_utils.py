import numpy as np
import pystk
import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import dense_transforms

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = 'drive_data'


class DeepRL:
    def __init__(self, pytux, track, gamma: float, entropy_weight: float):
        """Initialize."""
        self.env = pytux
        self.track = track
        self.gamma = gamma
        self.entropy_weight = entropy_weight

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.obs_dim = 4
        self.action_dim = 1

    def verbose(self, title):
        ax.clear()
        ax.set_title(title)
        ax.imshow(self.env.k.render_data[0].image)
        WH2 = np.array([self.env.config.screen_width, self.env.config.screen_height]) / 2
        ax.add_artist(
            plt.Circle(WH2 * (1 + self.env.to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
        ax.add_artist(
            plt.Circle(WH2 * (1 + self.env.to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
        plt.pause(1e-3)

    def update_kart(self, track, state):
        kart = state.players[0].kart
        cur_loc = kart.distance_down_track

        proj = np.array(state.players[0].camera.projection).T
        view = np.array(state.players[0].camera.view).T

        aim_point_world = self.env.point_on_track(cur_loc + TRACK_OFFSET, track)
        aim_point_image = self.env.to_image(aim_point_world, proj, view)
        current_vel = (np.linalg.norm(kart.velocity) - 10) / 10

        return aim_point_image, current_vel, aim_point_world, proj, view, kart

    def save_model(self, model):
        if isinstance(model, Actor):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'actor.th'))
        raise ValueError("model type '%s' not supported!" % str(type(model)))

    def load_model(self):
        r = Actor(self.obs_dim, self.action_dim).to(self.device)
        r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'actor.th'), map_location='cpu'))
        return r
