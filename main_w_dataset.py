from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
# import gzip
import rlgym.make
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from agents.Vector_load_hack.parsers.discrete_act import DiscreteAction
from terminal_conditions.custom_timeouts import KickoffTimeoutCondition
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from agents.Vector_load_hack.agent import Agent
import tqdm
import tkinter
from tkinter import messagebox
import os


# initialize 1d array for later use
arr = None

env = rlgym.make(tick_skip=1, use_injector=True, action_parser=DiscreteAction(), obs_builder=AdvancedObs(),
                 terminal_conditions=[TimeoutCondition(60*30), KickoffTimeoutCondition(60*5)], self_play=True)

model = Sequential()
model.add(LSTM(500, activation="relu", return_sequences=True))
model.add(LSTM(200, activation="relu", return_sequences=True))
model.add(LSTM(200, activation="relu", return_sequences=True))
model.add(LSTM(200, activation="relu", return_sequences=True))
model.add(LSTM(200, activation="relu", return_sequences=True))
model.add(LSTM(200, activation="relu", return_sequences=True))
model.add(LSTM(200, activation="relu", return_sequences=True))
model.add(LSTM(200, activation="relu", return_sequences=True))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(500, activation='tanh'))

opt = Adam(learning_rate=1e-3, decay=1e-4)

model.compile(loss='mean_squared_error', optimizer=opt,
              metrics=['mean_squared_error', 'accuracy'])

# UI stop training stuff
keep_training = True
top = tkinter.Tk()
top.geometry('150x100')


def stop_program():
    if messagebox.askyesno("Confirm", "Confirm Stop?"):
        T.delete("1.0", "end")
        T.insert("1.0", "Stopping...")
        global keep_training
        keep_training = False


B = tkinter.Button(top, text="Stop Training", command=stop_program)

T = tkinter.Text(top, padx=10, pady=10)
T.insert("1.0", "Training...")

B.pack(pady=10, padx=10)
T.pack()
# end of UI stuff

# amount of ticks to collect for the dataset
ep_len = 100_000
# initialize Vector agent
actor = Agent()
# the amount of steps taken in order to check for a chance to save an array
save_every = 2000
# batch size for learner
batch_size = 10_000
# directory to store data
directory = "data_collection"
# name of data
data_name = "arr_test"
# name of compiled final array
final_file = "Vector_data_full_arr"

while keep_training:
    # console progress bar
    prog_bar = tqdm.tqdm(desc="Collecting steps", total=ep_len, leave=True, smoothing=0.01, colour='green')

    # delete all files in data directory and reset
    for f_str in os.listdir(directory):
        os.remove(f_str)
    arr = None

    # #################################################################### #
    # start of data collection loop

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
            top.update()
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

    # see DiscreteAction.save_arr for understanding purposes, specialty function for data collection
    env._match._action_parser.save_arr(f"{directory}/{data_name}_final")

    # end of data gathering
    # #################################################################### #
    # start of array packing

    for f_str in os.listdir(directory):
        if f_str is f"{directory}/{final_file}_compressed":
            continue
            # maybe not necessary check to skip
        file_str = f"{directory}/{f_str}"
        f = np.load(file_str)
        if arr is None:
            arr = f
        else:
            arr = np.vstack((arr, f))

    # this is the data input arr
    arr_input = arr[:-1]
    # this is the target input arr
    arr_targ = arr[1:]

    dataset = tf.keras.utils.timeseries_dataset_from_array(arr_input, arr_targ, sequence_length=1,
                                                           batch_size=batch_size)

    # end of packing arrays into arr
    # #################################################################### #
    # start learning

    for batch in dataset:
        input_data, target = batch
        model.fit(
            input_data,
            target,
            verbose=1,
            epochs=3,
            validation_data=(input_data, target),
            workers=4,
            validation_split=0.2,
            use_multiprocessing=True)

# we're done, close it up
env.close()
