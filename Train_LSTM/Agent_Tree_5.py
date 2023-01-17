import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import entropy
from scipy import signal
from scipy import special
import scipy as sp
from functools import reduce

from helper import *

class LSTMAgent():
    def __init__(self, 
                 learning_rate, gamma, beta_v, beta_e,  #loss func
                 env, nb_trials, nb_episodes, nb_agent,
                 path,
                 nb_hidden = 128
                 ) -> None:
        
        self.nb_agent = nb_agent
        
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
        self.nb_obs = 231
        self.nb_inputs = self.nb_actions + self.nb_obs + (2 + 1) + 1
        self.nb_hidden = nb_hidden
        
        self.path = path
        self.log_dir = path+'/logs/'
        self.ckpt_dir = path+'/ckpt/'
        self.test_dir = path+'/test/'
        
        self.model, self.optimizer = self.LSTM_Model()
        

    def prepare_input(self, next_obs, agent_id, action_onehot, reward, timestep):
        obs = normalize_observation(next_obs[agent_id], tree_depth=2, observation_radius=10)
        timetable = np.array([self.env.agents[agent_id].earliest_departure, self.env.agents[agent_id].latest_arrival])
        
        return np.concatenate((obs, action_onehot[agent_id], timetable, np.array([reward,timestep]))).reshape(1,self.nb_inputs)
    
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
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
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

        total_loss = actor_loss + critic_loss #+ entropy

        return total_loss, actor_loss, critic_loss, entropy_loss

    def train(self):
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        
        for episode in range(self.nb_episodes):
            with tf.GradientTape() as tape:
                next_obs, _ = self.env.reset()
                
                action_probs_history = []
                action_distribution = []
                critic_value_history = []
                rewards_history = []
                reward = 0.0
                action_onehot = []
                cell_state = [tf.zeros((1,self.nb_hidden)),tf.zeros((1,self.nb_hidden))]
                entropy = 0
                reward = 0.0
                timestep = 0
                
                for agent_id in range(self.nb_agent):
                    action_probs_history.append([])
                    critic_value_history.append([])
                    action_onehot.append(np.zeros((self.nb_actions)))
                        
                while True:
                    
                    # Predict action probabilities and estimated future rewards from environment state
                    for agent_id in range(self.nb_agent):
                        input = self.prepare_input(next_obs, agent_id, action_onehot, reward, timestep)
                        action_probs, critic_value, cell_state = self.model([input,cell_state[0],cell_state[1]])
                    
                        critic_value_history[agent_id].append(tf.squeeze(critic_value))

                        # Sample action from action probability distribution
                        actions = {}
                        action_probs = tf.squeeze(action_probs)                        
                        action_onehot[agent_id] = np.zeros((self.nb_actions))
                        actions[agent_id] = np.random.choice(self.nb_actions, p=action_probs.numpy())
                        entropy += sp.stats.entropy(action_probs)
                        action_onehot[agent_id][actions[agent_id]] = 1.0
                        action_probs_history[agent_id].append(action_probs[actions[agent_id]])
                        action_distribution.append(actions[agent_id])
                        
                        entropy += sp.stats.entropy(action_probs)

                    # Apply the sampled action in our environment
                    next_obs, all_rewards, dones, info = self.env.step(actions)
                    reward = sum([all_rewards[r] for r in range(self.nb_agent)])
                    
                    # if action == STOP:
                    #     reward -= 1
                        
                    # normalize rewards
                    reward = reward / self.env._max_episode_steps
                
                    rewards_history.append(reward)
                    
                    
                    timestep += 1
                    if dones['__all__'] or timestep == self.nb_trials:
                        break
                    
                # for i in range(self.nb_actions):
                #     if action_distribution.count(i) > timestep/2:
                #         rewards_history[-1] -= (action_distribution.count(i) - timestep/2)/self.env._max_episode_steps
                
                all_total, all_actor, all_critic = 0,0,0
                for agent_id in range(self.nb_agent):
                    total_loss, actor_loss, critic_loss, entropy_loss = self.compute_loss(
                        tf.convert_to_tensor(action_probs_history[agent_id],dtype=tf.float32), 
                        tf.convert_to_tensor(critic_value_history[agent_id], dtype=tf.float32), 
                        tf.convert_to_tensor(rewards_history, dtype=tf.float32), 
                        entropy)
                    
                    # Backpropagation
                    
                    all_total += total_loss
                    all_actor += actor_loss
                    all_critic += critic_loss
                    
                grads = tape.gradient(all_total, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
                # Log
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss/total_loss', all_total, step=episode)
                    tf.summary.scalar('loss/actor_loss', all_actor, step=episode)
                    tf.summary.scalar('loss/critic_loss', all_critic, step=episode)
                    tf.summary.scalar('loss/entropy', entropy_loss, step=episode)
                    tf.summary.scalar('game/reward', np.sum(rewards_history), step=episode)
                    tf.summary.histogram('game/action_distribution', action_distribution, step=episode)
                
            # Checkpoint
            if episode % 1000 == 0:
                checkpoint = tf.train.Checkpoint(self.model)
                checkpoint.save(self.ckpt_dir+'checkpoints_'+str(episode)+'/tree5.ckpt')
                
        self.model.save(self.path+'/model.h5')