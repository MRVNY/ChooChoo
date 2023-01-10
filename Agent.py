import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import entropy
from scipy import signal
import scipy as sp
from functools import reduce

from helper import *

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

class LSTMAgent():
    def __init__(self, 
                 learning_rate, gamma, beta_v, beta_e,  #loss func
                 env, nb_trials, nb_episodes,       #train
                 path,                              #tfboard & ckpt
                 nb_hidden = 48
                 ) -> None:
        
        self.agent_id = 0
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.beta_v = beta_v
        self.beta_e = beta_e
        
        self.env = env
        self.nb_trials = nb_trials
        self.nb_episodes = nb_episodes
        
        # obs size: h x w x (16 + 1 + 1) = 30 x 30 x 18 = 16200
        # inputs = obs + action + reward + timestep + timetable
        self.nb_actions = 5
        self.nb_obs = 16200
        self.nb_inputs = self.nb_actions + self.nb_obs + 2 + 1 + 1
        self.nb_hidden = nb_hidden
        
        self.path = path
        self.log_dir = path+'/logs/'
        self.ckpt_dir = path+'/ckpt/'
        self.test_dir = path+'/test/'
        
        self.model, self.optimizer = self.LSTM_Model()
        

    def prepare_input(self, next_obs, agent_id, action_onehot, reward, timestep):
        my_obs = next_obs.get(self.agent_id)
        map = my_obs[TRANSITION_MAPS].flatten()
        pos_dir = my_obs[AGENT_STATES].T[MY_POS_DIR].T.flatten()
        target = my_obs[AGENT_TARGETS].T[0].T.flatten()
        timetable = np.array([self.env.agents[self.agent_id].earliest_departure, self.env.agents[self.agent_id].latest_arrival])
        
        return np.concatenate((map, pos_dir, target, action_onehot, timetable, np.array([reward,timestep]))).reshape(1,self.nb_inputs)
    
    def to_base_10(self, list):
        return reduce(lambda x,y: (x<<1) + y, list)
    
    def condense_map(self, obs):
        map = np.int32(obs[0][0])
        return np.apply_along_axis(self.to_base_10, 2, map)
        # out2 = np.zeros((30,30))
        # for i in range(30):
        #     for j in range(30):
        #         out2[i,j] = reduce(lambda x,y: (x<<1) + y, map[i,j])
    
    def LSTM_Model(self):
        inputs = layers.Input(shape=(self.nb_inputs))
        state_h = layers.Input(shape=(self.nb_hidden))
        state_c = layers.Input(shape=(self.nb_hidden))

        common, states = layers.LSTMCell(self.nb_hidden)(inputs, states=[state_h, state_c], training=True)
        action = layers.Dense(self.nb_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)

        model = keras.Model(inputs=[inputs,state_h,state_c], outputs=[action, critic, states], )
        optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        
        return model, optimizer

    def discount(self, x):
        return sp.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]

    def compute_loss(self, action_probs, values, rewards, entropy):
        """Computes the combined actor-critic loss."""
        
        bootstrap_n = tf.shape(rewards)[0]
        
        value_plus = np.append(values, bootstrap_n)
        rewards_plus = np.append(rewards, bootstrap_n)
        discounted_rewards = self.discount(rewards_plus)[:-1]
        advantages = rewards + self.gamma * value_plus[1:] - value_plus[:-1]
        advantages = self.discount(advantages)

        critic_loss = self.beta_v * 0.5 * tf.reduce_sum(input_tensor=tf.square(discounted_rewards - tf.reshape(values,[-1])))
        actor_loss = -tf.reduce_sum(tf.math.log(action_probs + 1e-7) * advantages)
        entropy_loss = self.beta_e * entropy

        total_loss = actor_loss + critic_loss + entropy

        return total_loss, actor_loss, critic_loss, entropy_loss

    def train(self):
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        for episode in range(self.nb_episodes):
            with tf.GradientTape() as tape:
                next_obs = self.env.reset()
                
                action_probs_history = []
                action_distribution = []
                critic_value_history = []
                rewards_history = []
                reward = 0.0
                action_onehot = np.zeros((self.nb_actions))
                cell_state = [tf.zeros((1,self.nb_hidden)),tf.zeros((1,self.nb_hidden))]
                entropy = 0.0
                next_obs, all_rewards, dones, _ = self.env.step({self.agent_id: NOTHING})
                reward = all_rewards[self.agent_id]
                done = dones[0]
            
                for timestep in range(self.nb_trials):
                    input = self.prepare_input(next_obs, self.agent_id, action_onehot, reward, timestep)
                    
                    # Predict action probabilities and estimated future rewards from environment state
                    action_probs, critic_value, cell_state = self.model([input,cell_state[0],cell_state[1]])
                    
                    critic_value_history.append(tf.squeeze(critic_value))

                    # Sample action from action probability distribution
                    action_probs = tf.squeeze(action_probs)
                    action = np.random.choice(self.nb_actions, p=action_probs.numpy())
                    action_probs_history.append(action_probs[action])
                    action_onehot = np.zeros((self.nb_actions))
                    action_onehot[action] = 1.0
                    action_distribution.append(action)

                    # Apply the sampled action in our environment
                    next_obs, all_rewards, dones, _ = self.env.step({self.agent_id: action})
                    reward = all_rewards[self.agent_id]
                    done = dones[0]
                    rewards_history.append(reward)
                    
                    # entropy
                    entropy += sp.stats.entropy(action_probs)
                    
                    if done: break
                    
                
                total_loss, actor_loss, critic_loss, entropy_loss = self.compute_loss(
                    tf.convert_to_tensor(action_probs_history,dtype=tf.float32), 
                    tf.convert_to_tensor(critic_value_history, dtype=tf.float32), 
                    tf.convert_to_tensor(rewards_history, dtype=tf.float32), 
                    entropy)
                        
                # Backpropagation
                grads = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
                # Log
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss/total_loss', total_loss, step=episode)
                    tf.summary.scalar('loss/actor_loss', actor_loss, step=episode)
                    tf.summary.scalar('loss/critic_loss', critic_loss, step=episode)
                    tf.summary.scalar('loss/entropy', entropy_loss, step=episode)
                    tf.summary.scalar('game/reward', np.sum(rewards_history), step=episode)
                    tf.summary.histogram('game/action_distribution', action_distribution, step=episode)
                
            # Checkpoint
            if episode % 2000 == 0:
                checkpoint = tf.train.Checkpoint(self.model)
                checkpoint.save(self.ckpt_dir+'checkpoints_'+str(episode)+'/two_steps.ckpt')
                
        self.model.save(self.path+'/model.h5')

    # def test(self, test_model=None, nb_trials=100):
    #     #test_summary_writer = tf.summary.create_file_writer(self.test_dir)

    #     next_obs, _ = self.env.reset()
    #     print(self.env.agents[0].earliest_departure)
    #     print(self.env.agents[0].latest_arrival)
    #     score = 0
    #     reward = 0.0
    #     action_onehot = np.zeros((self.nb_actions))
    #     cell_state = [tf.zeros((1,self.nb_hidden)),tf.zeros((1,self.nb_hidden))]
            
    #     for step in range(100):
    #         input = prepare_input(next_obs, self.agent_id, action_onehot,reward)
            
    #         action_probs, critic_value, cell_state = self.model([input,cell_state[0],cell_state[1]])
            
    #         action_probs = tf.squeeze(action_probs)
    #         action = np.random.choice(self.nb_actions, p=action_probs.numpy())
    #         action_onehot = np.zeros((self.nb_actions))
    #         action_onehot[action] = 1

    #         next_obs, all_rewards, dones, _ = self.env.step({self.agent_id: action})

    #         for agent_handle in self.env.get_agent_handles():
    #             score += all_rewards[agent_handle]

    #         render_env(env)
    #         print('Timestep {}, action = {}, total score = {}'.format(step, action_map[action], score))
    #         tf.print(action_probs)

    #         if dones['__all__']:
    #             print('All done!')
    #             return        

    #         # with test_summary_writer.as_default():
    #         #     tf.summary.scalar('loss/entropy', entropy, step=episode)
    #         #     tf.summary.scalar('game/reward', np.sum(rewards_history), step=episode)
    #         #     tf.summary.histogram('game/action_probs', action_probs_history, step=episode)
                