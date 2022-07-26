from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
# import rlgym
import gzip

# initialize 1d array
arr = np.empty(0)

# change this to use a new numpy array instead of the stored one
use_stored = True
# use_stored = False

# try to load stored numpy array to not have to start Rocket League for testing
f = None
if use_stored:
    try:
        f_comp = gzip.GzipFile(f"Vector_data_full_arr_compressed", "r")
        if f_comp is None:
            exit(1)
        f = np.load(f_comp)
        f_comp.close()

        arr = f.astype('float32')
    except OSError as e:
        print(e)
        exit(1)

# this is the data input arr
arr_input = arr[:-1]
# this is the target input arr
arr_targ = arr[1:]

dataset = tf.keras.utils.timeseries_dataset_from_array(arr_input, arr_targ, sequence_length=1, batch_size=10_000)
# for batch in dataset:
#     print(batch)
#     input_data, targets = batch

# if f was not loaded, create a numpy array from RLGym
# if f is None:
#     env = rlgym.make(use_injector=True)
#     while True:
#         x = 0
#         obs = env.reset()
#         x_arr = np.append(x_arr, obs, axis=0)
#         done = False
#         while not done:
#             x = x + 1
#             action = env.action_space.sample()
#             next_obs, reward, done, gameinfo = env.step(action)
#             obs = next_obs
#
#             x_arr = np.append(x_arr, obs, axis=0)
#             if x == 10000:
#                 break
#         break
#     np.save("arr_test", x_arr)
#     env.close()

# print loaded or generated array
# print(x_arr)
# expand dims to get to 3D instead of just 1D
# x_arr = np.expand_dims(x_arr, axis=1)
# x_arr = np.expand_dims(x_arr, axis=2)

print(len(arr))
print(arr.shape)

# necessary cast to f32 because otherwise tf throws an error?
# data_input = arr.astype('float32')

# y_train = x_train for now?
# y_train = x_train.copy()

# if x_train.shape != y_train.shape:
#     print(f"shape mis match {x_train.shape} {y_train.shape}")
#     exit(0)
#
# print(x_train.shape)
# print(y_train.shape)

# one layer for now because Sequential with more than one LSTM layer throws dimensional errors it seems
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

# loss function appears to not like -1, not sure if that is an output from the model error or if that is some other
# error? can we use another loss function?
model.compile(loss='mean_squared_error', optimizer=opt,
              metrics=['mean_squared_error', 'accuracy'])


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
