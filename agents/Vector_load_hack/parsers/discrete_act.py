import numpy as np
import gym.spaces
from rlgym.utils.gamestates import GameState
from math import pi


class DiscreteAction:
    """
    Simple discrete action space. All the analog actions have 3 bins by default: -1, 0 and 1.
    """
    POS_STD = 2300  # If you read this and wonder why, ping Rangler in the dead of night.
    ANG_STD = pi

    def __init__(self, n_bins=3):
        super().__init__()
        assert n_bins % 2 == 1, "n_bins must be an odd number"
        self._n_bins = n_bins
        # 43 matches the size of index 0
        self._simple_obs_action_store = np.empty((43, 0))

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.MultiDiscrete([self._n_bins] * 5 + [2] * 3)

    def parse_actions(self, actions: np.ndarray, state: GameState) -> np.ndarray:
        act = actions[0]
        for i in act:
            # x.append(self.parser.parse_actions(i, state))
            i = i.reshape((-1, 8))
            # map all ternary actions from {0, 1, 2} to {-1, 0, 1}.
            i[..., :5] = i[..., :5] / (self._n_bins // 2) - 1

        self._simple_obs_action_store = np.c_[self._simple_obs_action_store, self.get_obs(state, act)]

        return act

    def save_arr(self, file_name):
        np.save(file_name, self._simple_obs_action_store)
        self._simple_obs_action_store = np.empty((43, 0))

    def get_obs(self, state: GameState, actions: np.ndarray):
        ball = state.ball

        obs = [ball.position / self.POS_STD,
               ball.linear_velocity / self.POS_STD,
               ball.angular_velocity / self.ANG_STD]

        for action in actions:
            obs.append(action)

        for player in state.players:
            player_car = player.car_data

            obs.extend([
                player_car.position / self.POS_STD,
                player_car.linear_velocity / self.POS_STD,
                player_car.angular_velocity / self.ANG_STD])

        return np.concatenate(obs)
