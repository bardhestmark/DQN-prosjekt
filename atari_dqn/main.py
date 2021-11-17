import os
import random
from collections import deque
import itertools
import numpy as np
import torch
import csv

from baselines_wrappers import SubprocVecEnv, Monitor
from dqn import Network
from constants import *
from wrappers.pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames


def log_headers():
    os.mkdir(SAVE_FOLDER)
    with open(CSV_PATH, 'w', encoding='UTF8', newline='') as f:  # will overwrite file
        writer = csv.writer(f)
        hyper_parameters = [ENV_NAME, GAMMA, BATCH_SIZE, BUFFER_SIZE, MIN_REPLAY_SIZE, EPSILON_START, EPSILON_END,
                            EPSILON_DECAY, NUM_ENVS, TARGET_UPDATE_FREQ, LR]
        writer.writerow(hyper_parameters)
        headers = ['step', 'rew_mean', 'len_mean', 'episode_count']
        writer.writerow(headers)


def log_row_to_file(row):
    with open(CSV_PATH, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


# def resume_model(path):

if __name__ == '__main__':
    log_headers()

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    make_env = lambda: Monitor(make_atari_deepmind(ENV_NAME, scale_values=True), allow_early_resets=True)
    vec_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])
    env = BatchedPytorchFrameStack(vec_env, k=4)

    # Initialize networks
    online_net = Network(env, device=device)
    target_net = Network(env, device=device)
    online_net = online_net.to(device)
    target_net = target_net.to(device)
    target_net.load_state_dict(online_net.state_dict())

    #  Optimizer using adam, switch to huber loss?
    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

    # Buffers
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    episode_infos_buffer = deque([], maxlen=100)

    episode_count = 0

    # Initialize Replay Buffer
    observations = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        actions = [env.action_space.sample() for _ in range(NUM_ENVS)]

        new_observations, rewards, dones, _ = env.step(actions)

        for observation, action, reward, done, new_observation in zip(observations, actions, rewards, dones,
                                                                      new_observations):
            transition = (observation, action, reward, done, new_observation)
            replay_buffer.append(transition)

        observations = new_observations

    # Main Training Loop
    observations = env.reset()
    for step in itertools.count():
        epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

        rnd_sample = random.random()

        if isinstance(observations[0], PytorchLazyFrames):
            act_observations = np.stack([o.get_frames() for o in observations])
            actions = online_net.act(act_observations, epsilon)
        else:
            actions = online_net.act(observations, epsilon)

        new_observations, rewards, dones, infos = env.step(actions)

        for observation, action, reward, done, new_observation, info in zip(observations, actions, rewards, dones,
                                                                            new_observations, infos):
            transition = (observation, action, reward, done, new_observation)
            replay_buffer.append(transition)

            if done:
                episode_infos_buffer.append(info['episode'])
                episode_count += 1

        observations = new_observations

        # Start Gradient Step
        transitions = random.sample(replay_buffer, BATCH_SIZE)
        loss = online_net.compute_loss(transitions, target_net)

        # Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Target Network
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Logging
        if step % LOG_INTERVAL == 0:
            rew_mean = np.mean([e['r'] for e in episode_infos_buffer]) or 0
            len_mean = np.mean([e['l'] for e in episode_infos_buffer]) or 0

            print()
            print('Step:', step)
            if step != 0:
                print('Avg Rew:', rew_mean)
                print('Avg episode length:', len_mean)
                log_row_to_file([step, rew_mean, len_mean, episode_count])
            print('Episodes', episode_count)

        # Save
        if step % SAVE_INTERVAL == 0 and step != 0:
            print('Saving')
            online_net.save(PACK_PATH)
