# import numpy as np
from stable_baselines3 import PPO
import pathlib
from agents.Vector_load_hack.parsers.discrete_act import DiscreteAction


class TestOverrideLoad(PPO):
    def _get_torch_save_params(self):
        state_dicts = ["policy"]

        return state_dicts, []


class Agent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = {
            "lr_schedule": 0.000001,
            "clip_range": .02,
            "n_envs": 1,
        }
        
        self.actor = TestOverrideLoad.load(str(_path) + '/example_mdl', device='cpu', custom_objects=custom_objects)
        self.parser = DiscreteAction()

    def act(self, obs):
        action = self.actor.predict(obs, deterministic=True)
        # x = []
        # act = action[0]
        # for i in act:
        #     x.append(self.parser.parse_actions(i, state))

        return action


if __name__ == "__main__":
    print("You're doing it wrong.")
