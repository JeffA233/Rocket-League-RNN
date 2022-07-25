import rlgym.make
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from agents.Vector_load_hack.parsers.discrete_act import DiscreteAction
from terminal_conditions.custom_timeouts import KickoffTimeoutCondition
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from agents.Vector_load_hack.agent import Agent
import tqdm


env = rlgym.make(tick_skip=1, use_injector=True, action_parser=DiscreteAction(), obs_builder=AdvancedObs(),
                 terminal_conditions=[TimeoutCondition(60*30), KickoffTimeoutCondition(60*5)], self_play=True)

actor = Agent()

# amount of ticks to collect
ep_len = 120*1000
# the amount of steps taken in order to check for a chance to save
save_every = 2000
# directory to store data
directory = "data_collection"
# name of data
data_name = "arr_test"

prog_bar = tqdm.tqdm(desc=f"Collecting arrays", total=ep_len, leave=True, smoothing=0.01)

while True:
    obs = env.reset()
    actions = actor.act(obs)

    done = False
    ep_len_exceeded = False
    time_to_save = False

    x = 0
    while not ep_len_exceeded:
        x += 1
        prog_bar.update(1)
        obs, reward, done, gameinfo = env.step(actions)

        # check if we need to save
        if x % save_every == 0:
            time_to_save = True

        if done:
            obs = env.reset()
            # check when done if we need to save
            if time_to_save:
                env._match._action_parser.save_arr(f"{directory}/{data_name}{x}")
                time_to_save = False

        actions = actor.act(obs)

        if x == ep_len:
            ep_len_exceeded = True
    break

# see DiscreteAction.save_arr for understanding purposes
env._match._action_parser.save_arr(f"{directory}/{data_name}_final")
env.close()
