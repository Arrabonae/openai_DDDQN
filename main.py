import gym
from gym import wrappers

import numpy as np
import os
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

from utils import plot_learning_curve, loss_plot, save_frames_as_gif, clip_reward
from env import make_env
from replay import ReplayBuffer


class DeepQ(nn.Module):
    """
    Dueling double Deep Q-Learning with Pytorch.
    """
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQ, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4, bias= False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, bias = False)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, bias = False)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        
        lin_input_dims = self.calculate_conv_output_dims(input_dims)

        self.lin1 = nn.Linear(lin_input_dims, 1024)
        nn.init.kaiming_normal_(self.lin1.weight, mode='fan_out', nonlinearity='relu')
        self.lin2 = nn.Linear(1024, 512)
        nn.init.kaiming_normal_(self.lin2.weight, mode='fan_out', nonlinearity='relu')
        self.value = nn.Linear(512,1)
        self.advantage = nn.Linear(512,n_actions)

        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))


    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0],-1)
        flat = F.relu(self.lin1(conv_state))
        flat2 = F.relu(self.lin2(flat))
        #during training I have also tried dropout, it did not yield any better results.

        """
        This is the dueling part of the Network where the "head" of the splits between value and advantage
        """
        value = self.value(flat2)
        advantage = self.advantage(flat2)

        return value, advantage

    """
    Saving is done when the Best score ovbserved since training started. Loading is only done when the 
    load_checkpoint parameter set to "True" which is essentially the testing phase wehere so sit back and enjoy whatching your algorithm play the game. 
    """
    def save_checkpoint(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        print ("checkpoint saved...")
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print("checkpoint loaded...")
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent():
    def __init__(self, lr, input_dims, n_actions, mem_size, batch_size,
                replace,chkpt_dir, gamma, temperature, temp_min, temp_dec1, temp_dec2, algo=None, env_name=None,
                ): #epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.replace_target = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.n_actions = n_actions
        self.gamma = gamma
        #self.epsilon = epsilon
        self.temp_dec1 = temp_dec1
        self.temp_dec2 = temp_dec2
        self.temp_min = temp_min
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_count = 0
        self.beta = 0.4
        self.temperature = temperature
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        
        self.q_eval = DeepQ(self.lr, self.n_actions, input_dims = self.input_dims,
                            name = self.env_name+'_'+self.algo+'_q_eval',
                            chkpt_dir=self.chkpt_dir)
        self.q_next = DeepQ(self.lr, self.n_actions, input_dims = self.input_dims,
                            name = self.env_name+'_'+self.algo+'_q_target',
                            chkpt_dir=self.chkpt_dir)
        
    def choose_action(self, observation):
        """
        This function uses softmax action selection with continousily decreasing temperature. During training I have tried many variations of the temperature 
        and found that a decreasing version of this is the best way to takle between exploration and exploitation.
        Starting with higher temperature and decreasing the temperature so the probabilities will be sckewed to the highest probability.
        """
            
        state = T.tensor([observation], dtype=T.float32).to(self.q_eval.device)
        _, advantage = self.q_eval.forward(state)
        soft = nn.Softmax(dim=-1)
        prob = soft(advantage/self.temperature)
        prob = prob.cpu().detach().numpy()[0]
        action = np.random.choice(self.action_space, p= prob)

        
        if action == T.argmax(advantage).item():
            greedy.append(0)
        else: 
            greedy.append(1)

        return action


    def store(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def sample(self):
        self.beta = np.min([1., self.beta+0.001])
        state, action, reward, next_state, done, importance, batch = self.memory.sample(self.batch_size, self.beta)  

        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        next_states = T.tensor(next_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        importance = T.tensor(importance, dtype= T.float32).to(self.q_eval.device)
        batch = T.tensor(batch).to(self.q_eval.device)
        return states, actions, rewards, next_states, dones, importance, batch
    
    def replace_network(self):
        """
        Double network feature, where the network weights shared between the Eval network and Target network
        """
        if self.learn_step_count % self.replace_target ==0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            print("Target Network's weights replaced")

    def decrement_temperature(self):
          if n_steps <= 199000:
            self.temperature = self.temperature - self.temp_dec1 if self.temperature > self.temp_min else self.temp_min
          elif n_steps <= 300000: 
            self.temperature
          else:
            self.temperature = self.temperature - self.temp_dec2 if self.temperature > self.temp_min else self.temp_min

        

    def save(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_network()

        states, actions, rewards, next_states, dones, importance, batch = self.sample()  
        indices = np.arange(self.batch_size)

        #Value shape (batch,)
        #Advantage shape (batch,action_space.n)
        value_s, adv_s = self.q_eval.forward(states)
        value_next_s, adv_next_s = self.q_next.forward(next_states)
        value_next_s_eval, adv_next_s_eval = self.q_eval(next_states)

        q_pred = T.add(value_s, (adv_s - adv_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(value_next_s,(adv_next_s - adv_next_s.mean(dim=1, keepdim=True)))
        q_eval = T.add(value_next_s_eval, (adv_next_s_eval-adv_next_s_eval.mean(dim=1, keepdim=True)))
        max_actions = T.argmax(q_eval, dim=1)

        #masking terminal states
        q_next[dones] = 0.0
        #Bellman equation 
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        #Temporal-difference Error term for prioritized experience replay
        diff = T.abs(q_pred - q_target)

        for i in range(self.batch_size):
            idx = batch[i]
            self.memory.set_priorities(idx, diff[i].cpu().detach().numpy())

        loss = (T.cuda.FloatTensor(importance) * F.smooth_l1_loss(q_pred, q_target)).mean().to(self.q_eval.device)
        loss.backward()

        losses.append(loss.item())

        self.q_eval.optimizer.step()
        self.learn_step_count +=1
       
        self.decrement_temperature()

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    prev_avg = -np.inf
    load_checkpoint = False
    n_games = 351
    #print(env.unwrapped.get_action_meanings())
    #print(T.cuda.is_available())
    #np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)


    agent = Agent(gamma=0.99,lr=0.0001, input_dims=(env.observation_space.shape),
                    n_actions = env.action_space.n, mem_size=70000, batch_size=32, replace=10000, temperature=0.2, temp_min=0.0004, temp_dec1 = 1e-6, temp_dec2 = 1e-8,
                    chkpt_dir='models/', algo='DuelingDoubleDQNAgent', 
                    env_name='PongNoFrameskip-v4') #epsilon = 1.0
    
    if load_checkpoint: 
        agent.load()

    filename = agent.algo +'_'+agent.env_name + '_lr' + str(agent.lr) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename +'.png'
    losses_file = 'plots/'+agent.env_name+ 'loss.png'
    
    n_steps = 0
    scores, eps_history, steps_array, losses, greedy_hist = [], [], [], [], []
    
    for i in range(n_games):
        t0 = time.time()
        done = False
        obs = env.reset()
        previous_life = 0
        done_store = True
        frames, greedy = [], []
        score = 0

        while not done:

            if i  < 275:
              if i % 25 == 0:
                frames.append(env.render(mode='rgb_array'))
            else:
              if i % 5 == 0:
                frames.append(env.render(mode='rgb_array'))

            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            if info['ale.lives'] >= previous_life:
                done_store = done
            else:
                done_store = True
            previous_life = info['ale.lives']
            score += reward
            reward = clip_reward(reward)

            #I have tried clipping the rewards to control the loss function, but learned that this would only result less "certain" predictions in terms of the next Action to take.  
            #reward = np.clip(reward,0, 1)

            if not load_checkpoint:
                agent.store(obs, action, reward, next_obs, int(done_store))
                agent.learn()

            obs = next_obs
            n_steps += 1
          
        if i  < 275:
          if i % 25 == 0:
            save_frames_as_gif(frames,i)
        else:
          if i % 5 == 0:
            save_frames_as_gif(frames,i)
        
        
        scores.append(score)
        steps_array.append(n_steps)
        avg_score = np.mean(scores[-25:])
        t1 = time.time()
        t = t1-t0
        print('episode', i, 'last score %.0f, average score %.2f, best score %.2f, temperature %.4f, softmax greedy %.4f,' %
            (score, avg_score, best_score, agent.temperature, np.sum(greedy)/len(greedy)),
            'steps ', n_steps, 'time ', t)
    
        if avg_score > prev_avg:
            if not load_checkpoint:
                agent.save()
                prev_avg = avg_score              

        if score > best_score:    
          best_score = score
        #I also tried to use Decaying Epsilon Greedy 
        #eps_history.append(agent.epsilon)
        greedy_hist.append(np.sum(greedy)/len(greedy))

    plot_learning_curve(steps_array, scores, greedy_hist, figure_file)
    loss_plot(losses, losses_file)
