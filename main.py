# Setting the path to include the yet uninstalled environment package
import sys
import os
package_path = os.path.join(os.getcwd(),'environment package','warehouse')
sys.path.append(package_path)
# print(sys.path)

# Importing packages
import os
import numpy as np
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.models import load_model as keras_load_model
import gym

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy
from rl import policy as rl_policy
from rl import memory as rl_memory

from ExperimentManager import getManagerFromConfig
manager = getManagerFromConfig('config.json')

from models import models
import warehouse
from agents import MaskingDQNAgent, CustomDQNAgent
from policies import MaskingMaxBoltzmannQPolicy

import argparse

np.random.seed(0)

# Argument parser to run easily on gg colab
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--gg_colab", type=bool, default=None, help="Remove render for gg colab uses")
parser.add_argument("--do_testing", type=bool, default=None)

args = parser.parse_args()

agents = {"DQNAgent":DQNAgent, "CustomDQNAgent": CustomDQNAgent}

@manager.command
def basic_run(agent, load_model, architecture, training, optimization, memory, policy, testing):
    TRAINED = False

    env_name = "warehouse-v0"
    env = gym.make(env_name)

    # Overwrite the config variables with parse args
    if args.gg_colab is not None:
        training['visualize'], testing["render"] = (args.gg_colab,)*2
    if args.do_testing is not None:
        testing["do"] = args.do_testing

    # make the manager available in the environment
    env.manager = manager # used for metrics logging by the callbacks

    # The model
    nb_actions = env.action_space.n
    input_shape = env.OS.input_shape()
    if load_model is None:
        model = getattr(models,architecture)(input_shape, nb_actions, use_bias=False)
    else:
        model = keras_load_model(manager.get_load_path(load_model,'model.h5'))
        TRAINED = True
    model.build(input_shape)
    print(model.summary())

    # The memory
    memory = getattr(rl_memory,memory["name"])(*memory["args"],**memory["kwargs"])

    # The policy
    if policy['name'] == 'MaskingMaxBoltzmannQPolicy':
        policy = MaskingMaxBoltzmannQPolicy(*policy["args"],**policy["kwargs"])
    elif policy['name'] == "LinearAnnealedPolicy":
        policy = LinearAnnealedPolicy(getattr(rl_policy,policy["sub_policy"])(), *policy["args"],**policy["kwargs"])
    else:
        policy = getattr(rl_policy,policy["name"])(*policy["args"],**policy["kwargs"])

    # The agent
    if agent['name'] == 'MaskingDQNAgent':
        dqn = MaskingDQNAgent(model=model, environment=env,nb_actions=nb_actions, memory=memory, policy=policy, *agent["args"], **agent['kwargs'])
    else:
        dqn = agents[agent["name"]](model=model, nb_actions=nb_actions, memory=memory, policy=policy, *agent["args"], **agent['kwargs'])
    dqn.compile(getattr(optimizers,optimization["optimizer"])(lr=optimization["lr"]), metrics=['mae'])
    
    if not TRAINED:
        # # Training
        history = dqn.fit(env, verbose=1, **training) #callbacks=env.callbacks,
        # TODO: Debug verbose=2, error with TrainEpisodeLogger

        # Save history
        manager.log_scalars("history", [history.epoch, *history.history.values()], ["epoch", *history.history.keys()])
        manager.log_scalars("reward", history.history["episode_reward"])
        manager.log_scalars("avg_q", history.history["episode_avg_maxq"])
        manager.log_scalars("avg_minq", history.history["episode_avg_minq"])
        manager.log_scalars("n_step", history.history["nb_steps"])
        manager.log_scalars("nb_episode_steps", history.history["nb_episode_steps"])
        manager.log_scalars("epoch", history.epoch)

        # Post-training
        # manager.save(dqn.model,"model")
        dqn.model.save(os.path.join(manager.save_dir, "model.h5"), overwrite=True)

#    Finally, evaluate our algorithm for 5 episodes.
    if testing["do"]:
        print("\nTesting the agent")
        dqn.test(env, nb_episodes=testing["nb_episodes"], visualize=testing["render"], callbacks=env.callbacks)

if __name__ == "__main__":

    manager.run('basic_run')