# Double Dueling Deep Q-Learn with Pytorch

**Episode 0          |  Solarized Ocean**
:-------------------------:|:-------------------------:
![](gifs/Episode_0.gif)  |  ![](gifs/Episode_350.gif)

## OpenAI Atari solved with DDDQN Pytorch
This code is an example of Double Dueling Deep Q-Learn network solving the OpenAI Atari environment: **Pong**. The following methods were implemented in this code. <br/>
Basics of Q-learning: https://en.wikipedia.org/wiki/Q-learning <br/>

### Dueling Double Deep Q - learning
Like the standard DQN architecture, we have convolutional layers to process game-play frames. From there, we split the network into two separate streams, one for estimating the state-value and the other for estimating state-dependent action advantages. After the two streams, the last module of the network combines the state-value and advantage outputs.<br/> [source: https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751] <br/>
<br/>
<br/>
**Architecture**<br/>
![](gifs/DDDQN.png)<br/>
[soruce:https://www.freecodecamp.org/news/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682/]

### Softmax action selection
During training and testing of the algorithm I found the best way to takle exploration - exploitation dilemma is to use a softmax action selection with decreasing temperature.<br/>
Although epsilon greedy action selection is an effective and popular means of balancing exploration - exploitation in reinforcement learning, one drawback is that when it explores it chooses equally among all actions. This drawback can be corrected by using a probability based action selection algoritm.<br/>
**The softmax action selection formula**<br/>
![](gifs/softmax.png)<br/>
[source: see research papers below]<br/>
With the temperature continously dropping, the action selection drop from 80% exploration down to aprox. 10% exproration at the end of the training. The agent learn much faster given that even if the action selection is not greedy, it is based on probabilities.

### Prioritised experience replay (PER)
Experience replay lets reinforcement learning agents remember and reuse experiences from the past. Without prioritisation, experience transitions were uniformly sampled from a replay memory. However, this approach simply replays transitions at the same frequency that they were originally experienced, regardless of their significance. The Prioritised experience replay essentially gives weights to different experiances based on their importance in the elarning process, speeding up the learning process and make it more efficient. PER, generally, can be achived in two ways: ranking and stochastic prioritisation <br/>
![](gifs/PER.png)<br/>
[source:see research papers below]

## Links to research papers and other repositories
Base structure: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf <br/>
Softmax action selection: https://arxiv.org/pdf/1903.05926.pdf <br/>
Prioritised experience replay: https://arxiv.org/pdf/1511.05952.pdf <br/>
Inspired by: https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code <br/>
This code also uses parts of the OpenAI Baselines: https://github.com/openai/baselines <br/>
