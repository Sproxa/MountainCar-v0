import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
GENS = 25000
SHOW = 100
render = False

DISCRETE_OS_SIZE = [20, 20]
DISCRETE_OS_WIN_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def getdiscretestate(state):
        discrete_state = (state - env.observation_space.low) /  DISCRETE_OS_WIN_SIZE
        return tuple(discrete_state.astype(np.int))


for gen in range(GENS):
    if gen % SHOW == 0:
        render = True
    else:
        render = False
    discrete_state = getdiscretestate(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = getdiscretestate(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action, )] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state +(action, )] = 0
            print("Done It!")
        discrete_state =  new_discrete_state
