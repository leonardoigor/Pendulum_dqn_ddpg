import gym
import numpy as np
from agent import Agent


env = gym.make('Pendulum-v0')
agent = Agent(input_dims=env.observation_space.shape, env=env,
              n_actions=env.action_space.shape[0],)


n_games = 250


best_score = -env.reward_range[0]

score_history = []

load_checkpoint = True


if load_checkpoint:
    n_steps = 0
    while n_steps < agent.batch_size:
        obs = env.reset()

        action = agent.choose_action(obs)

        obs_, reward, done, info = env.step(action)
        agent.remember(obs, action, reward, obs_, done)
        n_steps += 1
    agent.learn()
    agent.load_models()
    evaluate = True
else:
    evaluate = False
for i in range(n_games):
    obs = env.reset()
    score = 0
    done = False
    while not done:
        if evaluate:
            env.render()
        action = agent.choose_action(obs)
        obs_, reward, done, info = env.step(action)
        agent.remember(obs, action, reward, obs_, done)
        if not load_checkpoint:
            agent.learn()
        score += reward
        obs = obs_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if not load_checkpoint:
        agent.save_models()
    print('episode ', i, 'score %.2f' %
          score, 'average score %.2f' % avg_score)
