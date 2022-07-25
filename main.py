from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten
from keras.optimizers import Adam
import numpy as np
import rlgym
import gzip

# initialize 1d array
x_arr = np.empty(0)

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
        # f = np.load("arr_test.npy")
        x_arr = f
    except OSError as e:
        print(e)
        exit(1)

# if f was not loaded, create a numpy array from RLGym
if f is None:
    env = rlgym.make(use_injector=True)
    while True:
        x = 0
        obs = env.reset()
        x_arr = np.append(x_arr, obs, axis=0)
        done = False
        while not done:
            x = x + 1
            action = env.action_space.sample()
            next_obs, reward, done, gameinfo = env.step(action)
            obs = next_obs

            x_arr = np.append(x_arr, obs, axis=0)
            if x == 10000:
                break
        break
    np.save("arr_test", x_arr)
    env.close()

# print loaded or generated array
# print(x_arr)
# expand dims to get to 3D instead of just 1D
x_arr = np.expand_dims(x_arr, axis=1)
# x_arr = np.expand_dims(x_arr, axis=2)

print(len(x_arr))
print(x_arr.shape)

# necessary cast to f32 because otherwise tf throws an error?
x_train = x_arr.astype('float32')

# y_train = x_train for now?
y_train = x_train.copy()

if x_train.shape != y_train.shape:
    print(f"shape mis match {x_train.shape} {y_train.shape}")
    exit(0)

print(x_train.shape)
print(y_train.shape)

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

model.fit(
    x_train,
    y_train,
    verbose=1,
    epochs=3,
    validation_data=(x_train, y_train),
    workers=4,
    validation_split=0.2,
    use_multiprocessing=True)
