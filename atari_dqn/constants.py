import time

env_list = [
    'Breakout-v0',      # 0
    'Assault-v0',       # 1
    'SpaceInvaders-v0',  # 2
    'Pong-v0'           # 3
]  # just a reference for testing different environments

ENV_NAME = env_list[3]

GAMMA = 0.999
BATCH_SIZE = 32
BUFFER_SIZE = int(1e5)
MIN_REPLAY_SIZE = 50000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = int(1e6)
NUM_ENVS = 4
TARGET_UPDATE_FREQ = 10000 // NUM_ENVS
LR = 5e-5

SAVE_FOLDER = './trained_models/' + ENV_NAME + '_' + time.strftime("%Y-%m-%d-%H%M%S/")
PACK_PATH = SAVE_FOLDER + ENV_NAME + '.pack'
CSV_PATH = SAVE_FOLDER + ENV_NAME + '.csv'
SAVE_INTERVAL = 10000
LOG_INTERVAL = 1000
