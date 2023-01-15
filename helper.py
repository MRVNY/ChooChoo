import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from scipy.stats import entropy
import scipy as sp
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.step_utils.states import TrainState

import PIL
from flatland.utils.rendertools import RenderTool
from IPython.display import clear_output


# Render the environment
def render_env(env,wait=True):
    
    env_renderer = RenderTool(env, gl="PILSVG")
    env_renderer.render_env()

    image = env_renderer.get_image()
    pil_image = PIL.Image.fromarray(image)
    clear_output(wait=True)
    display(pil_image)
    

def prepare_input(env, next_obs, agent_id, action_onehot, reward, timestep, nb_inputs):
    my_obs = next_obs.get(agent_id)
    map = my_obs[TRANSITION_MAPS].flatten()
    pos_dir = my_obs[AGENT_STATES].T[MY_POS_DIR].T.flatten()
    target = my_obs[AGENT_TARGETS].T[0].T.flatten()
    timetable = np.array([env.agents[agent_id].earliest_departure, env.agents[agent_id].latest_arrival])
    
    return np.concatenate((map, pos_dir, target, action_onehot, timetable, np.array([reward,timestep]))).reshape(1,nb_inputs)

    

TRANSITION_MAPS = 0
AGENT_STATES = 1
AGENT_TARGETS = 2

MY_POS_DIR = 0
OTHER_POS_DIR = 1
MALFUNCTIONS = 2
FRAC_SPEEDS = 3
NUM_AGENTS_WITH_ME = 4

NOTHING = 0
LEFT = 1
FOWARD = 2
RIGHT = 3
STOP = 4

action_map = {
    NOTHING: "nothing",
    LEFT: "left",
    FOWARD: "foward",
    RIGHT: "right",
    STOP: "stop"
}