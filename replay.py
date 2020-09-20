import collections
import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.priorities = collections.deque(maxlen=max_size)


        self.state_memory = np.zeros((self.mem_size,*input_shape), dtype=np.float32)
        self.new_state_memory =np.zeros((self.mem_size,*input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size,dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size 
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.priorities.append(max(self.priorities, default=1))
        self.mem_cntr += 1
    #stochastic sampling method that interpolates between pure greedy prioritization and uniform random sampling (alpha determines how much prioritization is used)
    def get_probabilities(self):
        scaled_priorities = np.array(self.priorities)
        scaled_priorities = scaled_priorities/ scaled_priorities.sum()
        return scaled_priorities
    
    #Importance-sampling (IS) weight with Beta
    #For stability reasons, we always normalize weights by 1/max(importance) so that they only scale the update downwards.
    def get_importance(self, probabilities, beta):
        self.beta = beta
        importance =  np.power(1/self.mem_size * 1/probabilities, -self.beta)
        importance = importance / max(importance)
        return importance

    def sample(self, batch_size, beta):
        max_mem = min(self.mem_cntr, self.mem_size)
        sample_probs = self.get_probabilities()
        batch = np.random.choice(max_mem, batch_size, replace=False, p= sample_probs)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        importance = self.get_importance(sample_probs[batch], beta)

        return states, actions, rewards, next_states, dones, importance, batch
    
    #proportional prioritization
    def set_priorities(self, idx, errors, offset=1.1, alpha = 0.7):
            self.priorities[idx] = (np.abs(errors) + offset)** alpha
