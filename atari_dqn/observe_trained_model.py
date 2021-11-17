import itertools

import torch
import time
import numpy as np
from dqn import Network
from baselines_wrappers import DummyVecEnv
from wrappers.pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, \
    PytorchLazyFrames

path = 'trained_models/Pong-v0_2021-11-17-120751/Pong-v0.pack'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

make_env = lambda: make_atari_deepmind('Pong-v0', scale_values=True)
vec_env = DummyVecEnv([make_env for _ in range(1)])
env = BatchedPytorchFrameStack(vec_env, k=4)

net = Network(env, device)
net = net.to(device)

net.load(path)

obs = env.reset()
beginning_episode = True
for t in itertools.count():
    if isinstance(obs[0], PytorchLazyFrames):
        act_obs = np.stack([o.get_frames() for o in obs])
        action = net.act(act_obs, 0.0)
    else:
        action = net.act(obs, 0.0)

    if beginning_episode:
        action = [1]
        beginning_episode = False

    obs, rew, done, _ = env.step(action)
    env.render()
    time.sleep(0.005)  # Decides speed

    if done[0]:
        obs = env.reset()
        beginning_episode = True
