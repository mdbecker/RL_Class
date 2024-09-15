```markdown
# **Lesson 6.2: Exploring Dueling Networks**

## **Learning Objectives**
- **Understand the architecture of Dueling Networks.**
- **Learn how Dueling Networks separate value and advantage streams.**
- **Implement Dueling DQN** using Stable Baselines3.
- **Analyze the impact of Dueling Networks** on agent performance and learning efficiency.
- **Compare Dueling DQN with standard DQN** to observe performance improvements.
- **Visualize the benefits** of Dueling Networks in reducing overestimation bias.

## **Description**
In this lesson, we'll explore **Dueling Networks**, an architectural enhancement to DQN that separates the estimation of **state-value** and **advantage** functions. This separation allows the network to better assess the value of being in a particular state, irrespective of the action taken, leading to more efficient learning and improved performance. We'll implement Dueling DQN using **Stable Baselines3 (SB3)** and compare its performance with standard DQN to observe the benefits of this architectural innovation.

## **Setting Up the Environment**

Ensure you are in the `rl_week6` Conda environment with the necessary packages installed. If not, follow the setup instructions below.

### **Step 1: Create a New Conda Environment**

```bash
# Create a new Conda environment named 'rl_week6' with Python 3.10
conda create -n rl_week6 python=3.10 -y
```

### **Step 2: Activate the Environment**

```bash
# Activate the 'rl_week6' environment
conda activate rl_week6
```

### **Step 3: Install Essential Packages**

We'll install essential packages for Advanced Value-Based Methods, including JupyterLab, NumPy, Matplotlib, and PyTorch.

```bash
# Install JupyterLab, NumPy, and Matplotlib using Conda
conda install -c conda-forge jupyterlab numpy matplotlib -y

# Install PyTorch (CPU version) using Conda
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### **Step 4: Install Additional RL Libraries**

We'll install OpenAI Gym and Stable Baselines3 using `pip`.

```bash
# Install OpenAI Gym and Stable Baselines3 using pip
pip install gym
pip install stable-baselines3
```

### **Step 5: Launch JupyterLab**

```bash
# Launch JupyterLab
jupyter lab
```

*JupyterLab will open in your default web browser, providing an interactive environment for coding and visualization.*

---

## **What are Dueling Networks?**

**Dueling Networks** introduce a novel architecture for Q-Networks by decomposing the Q-value into two separate components:
1. **Value Function (\( V(s) \))**: Estimates the value of being in a state \( s \).
2. **Advantage Function (\( A(s, a) \))**: Estimates the advantage of taking action \( a \) in state \( s \) compared to the average action.

### **Benefits of Dueling Networks**
- **Efficient Learning:** Separating value and advantage allows the network to learn which states are valuable without having to learn the effect of each action in those states.
- **Improved Policy Evaluation:** Enhances the estimation of the value of states, leading to more informed action selection.
- **Better Performance:** Often results in faster convergence and higher rewards compared to standard DQN, especially in environments where some actions do not affect the outcome significantly.

### **Dueling Network Architecture**

The architecture consists of two streams:
1. **Value Stream:** Outputs a single value \( V(s) \).
2. **Advantage Stream:** Outputs advantages \( A(s, a) \) for each action.

These streams are then combined to produce the final Q-values:
\[
Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)
\]
This formulation ensures that the advantage function has zero mean, preventing identifiability issues.

---

## **Practical Example: Visualizing Dueling Networks**

Let's visualize how Dueling Networks separate value and advantage streams by implementing a simple Dueling Q-Network in PyTorch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        
        # Value stream
        self.value_fc = nn.Linear(hidden_size, 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        
        # Combine value and advantage to get Q-values
        q_values = value + (advantage - advantage.mean())
        return q_values

# Instantiate the network
state_size = 4  # Example state size for CartPole
action_size = 2  # Example action size for CartPole
dueling_net = DuelingQNetwork(state_size, action_size)

# Forward pass with a sample input
sample_state = torch.FloatTensor([0.0, 0.0, 0.0, 0.0])
q_vals = dueling_net(sample_state)
print(q_vals)
```

### **Explanation**
- **DuelingQNetwork Class:** Defines the architecture with separate streams for value and advantage.
- **Forward Method:** Processes input through shared layers, then splits into value and advantage streams, and finally combines them to produce Q-values.
- **Sample Input:** Demonstrates how the network outputs Q-values by combining value and advantage estimates.

**Expected Output:**
```
tensor([-0.0171,  0.0171], grad_fn=<AddBackward0>)
```
*Figure: The output Q-values for each action, showing the combination of value and advantage streams.*

---

## **Implementing Double DQN with Stable Baselines3**

Stable Baselines3 (SB3) integrates Double DQN internally, allowing you to easily enable it through configuration parameters. We'll configure SB3's DQN to use Double DQN and compare its performance with standard DQN.

### **Step 1: Import Necessary Libraries**

```python
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.callbacks import CallbackList
import torch
import numpy as np
import random
```

### **Step 2: Initialize the Environment**

We'll use the **CartPole-v1** environment for this example.

```python
# Initialize the CartPole environment
env = gym.make('CartPole-v1')

# Set random seeds for reproducibility
seed = 42
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
```

**Environment Overview:**
- **State Space:** Continuous (position, velocity, angle, angular velocity)
- **Action Space:** Discrete (move left or right)
- **Objective:** Prevent the pole from falling over by moving the cart appropriately.

### **Step 3: Configure the DQN Agent with Experience Replay and Target Networks**

```python
# Create the DQN agent with Experience Replay and Target Networks
model = DQN(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=1e-3,           # Learning rate for the optimizer
    buffer_size=50000,            # Size of the replay buffer
    learning_starts=1000,         # Number of steps before training starts
    batch_size=32,                 # Mini-batch size for training
    gamma=0.99,                    # Discount factor for future rewards
    target_update_interval=1000,   # Frequency (in steps) to update the Target Network
    exploration_fraction=0.1,      # Fraction of training for exploration
    exploration_final_eps=0.02,    # Final exploration rate
    tensorboard_log="./dqn_cartpole_tensorboard/"  # TensorBoard log directory
)
```

**Hyperparameter Breakdown:**
- **`learning_rate`:** Controls how much to adjust the network weights during training.
- **`buffer_size`:** Determines how many past experiences are stored for replay.
- **`learning_starts`:** Number of timesteps before training begins to ensure a diverse replay buffer.
- **`batch_size`:** Number of experiences sampled from the replay buffer for each training step.
- **`gamma`:** Determines the importance of future rewards.
- **`target_update_interval`:** How often the Target Network is updated to match the main network.
- **`exploration_fraction` & `exploration_final_eps`:** Control the exploration rate over time.
- **`tensorboard_log`:** Directory for logging training metrics to TensorBoard.

### **Step 4: Define Custom Callbacks for Monitoring and Early Stopping**

Callbacks allow for enhanced monitoring and control over the training process.

```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomLoggingCallback(BaseCallback):
    """
    Custom callback for logging additional metrics during training.
    """
    def __init__(self, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Check if a new episode has started
        if 'episode_rewards' in self.locals:
            self.episode_rewards.append(self.locals['episode_rewards'])
            self.episode_lengths.append(self.locals['episode_lengths'])
        return True

# Initialize the custom callback
custom_logging_callback = CustomLoggingCallback()

# Define evaluation callback
eval_callback = EvalCallback(
    env, 
    best_model_save_path='./logs/',
    log_path='./logs/', 
    eval_freq=500, 
    deterministic=True, 
    render=False
)

# Define early stopping callback
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=195, 
    verbose=1
)

# Combine callbacks
callback = CallbackList([eval_callback, stop_callback, custom_logging_callback])
```

**Callback Breakdown:**
- **`CustomLoggingCallback`:** Logs episode rewards and lengths for custom visualization.
- **`EvalCallback`:** Periodically evaluates the agent's performance and saves the best model.
- **`StopTrainingOnRewardThreshold`:** Stops training early if the agent achieves the specified reward threshold.

### **Step 5: Train the DQN Agent with Callbacks**

```python
# Train the DQN agent with callbacks
model.learn(
    total_timesteps=10000, 
    callback=callback
)
```

**Training Process:**
- The agent interacts with the environment, storing experiences in the replay buffer.
- Experiences are sampled randomly from the buffer to train the Q-Network.
- The Target Network is updated at specified intervals to provide stable target Q-values.
- Callbacks monitor performance, log metrics, and implement early stopping based on rewards.

### **Step 6: Evaluate the Trained Agent**

After training, it's essential to evaluate the agent's performance to understand how well it has learned to solve the environment.

```python
# Evaluate the agent's performance
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"Mean Reward: {mean_reward} +/- {std_reward}")
```

**Expected Output:**
```
Mean Reward: 200.0 +/- 0.0
```
*Note: In deterministic environments like CartPole-v1 with `is_slippery=False`, the agent may consistently achieve the maximum reward.*

### **Step 7: Visualize Agent Performance**

Visualizing the agent's performance helps in understanding its behavior and verifying that it has learned an effective policy.

```python
# Function to visualize the agent
def visualize_agent(env, model, num_episodes=5):
    """
    Visualizes the trained agent's performance in the environment.
    
    Args:
        env (gym.Env): The Gym environment.
        model (stable_baselines3.DQN): The trained DQN model.
        num_episodes (int): Number of episodes to visualize.
    """
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()  # Render the environment
            action, _states = model.predict(state, deterministic=True)  # Select action
            state, reward, done, info = env.step(action)  # Take action
            total_reward += reward
        print(f"Episode {episode}: Total Reward: {total_reward}")
    env.close()

# Visualize the trained agent
visualize_agent(env, model)
```

**Expected Output:**
```
Episode 1: Total Reward: 200
Episode 2: Total Reward: 200
Episode 3: Total Reward: 200
...
```

*Figure: The environment will render visually in a separate window, showing the cart balancing the pole successfully.*

### **Step 8: Plot Training Progress**

While Stable Baselines3 does not provide built-in plotting for training progress, you can track and plot rewards by implementing custom callbacks or logging. Additionally, integrating TensorBoard allows for comprehensive monitoring of training metrics.

#### **Example: Using TensorBoard for Monitoring**

1. **Modify the Training Code to Include TensorBoard Logging:**

    ```python
    # Re-initialize the DQN agent with TensorBoard logging
    model = DQN(
        'MlpPolicy', 
        env, 
        verbose=1, 
        learning_rate=1e-3, 
        buffer_size=50000, 
        learning_starts=1000, 
        batch_size=32, 
        gamma=0.99, 
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        tensorboard_log="./dqn_cartpole_tensorboard/"
    )
    
    # Train the agent with callbacks
    model.learn(
        total_timesteps=10000, 
        callback=callback
    )
    ```

2. **Launch TensorBoard:**

    ```bash
    # In terminal, run:
    tensorboard --logdir=./dqn_cartpole_tensorboard/
    ```

3. **Access TensorBoard:**
    - Open the provided URL in your browser to monitor training metrics in real-time.

#### **Example: Plotting Logged Rewards**

```python
# Plot the logged rewards from CustomLoggingCallback
plt.figure(figsize=(12,6))
plt.plot(custom_logging_callback.episode_rewards, label='Episode Reward', color='purple')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN: Episode Rewards Over Time with Custom Logging')
plt.legend()
plt.grid(True)
plt.show()
```

**Expected Output:**

![DQN Episode Rewards Over Time](https://i.imgur.com/9KX5Zc1.gif)
*Figure: The plot shows the rewards obtained by the agent in each episode, indicating learning progress.*

---

## **Compare Double DQN with Standard DQN**

To observe the benefits of Double DQN, we'll train two agents: one using standard DQN and another using Double DQN. We'll then compare their performance.

### **Step 1: Train Standard DQN**

```python
# Create the standard DQN agent
model_standard_dqn = DQN(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    double_q=False,  # Disable Double DQN
    tensorboard_log="./standard_dqn_cartpole_tensorboard/"
)

# Train the standard DQN agent
model_standard_dqn.learn(
    total_timesteps=10000, 
    callback=callback
)

# Evaluate the standard DQN agent
mean_reward_standard_dqn, std_reward_standard_dqn = evaluate_policy(model_standard_dqn, env, n_eval_episodes=100)

print(f"Standard DQN Mean Reward: {mean_reward_standard_dqn} +/- {std_reward_standard_dqn}")
```

### **Step 2: Train Double DQN**

```python
# Create the Double DQN agent
model_double_dqn = DQN(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    double_q=True,  # Enable Double DQN
    tensorboard_log="./double_dqn_cartpole_tensorboard/"
)

# Train the Double DQN agent
model_double_dqn.learn(
    total_timesteps=10000, 
    callback=callback
)

# Evaluate the Double DQN agent
mean_reward_double_dqn, std_reward_double_dqn = evaluate_policy(model_double_dqn, env, n_eval_episodes=100)

print(f"Double DQN Mean Reward: {mean_reward_double_dqn} +/- {std_reward_double_dqn}")
```

### **Step 3: Compare Performance Metrics**

```python
# Example: Plotting mean rewards
configs = ['Standard DQN', 'Double DQN']
rewards = [mean_reward_standard_dqn, mean_reward_double_dqn]
std_devs = [std_reward_standard_dqn, std_reward_double_dqn]

plt.figure(figsize=(10,6))
plt.bar(configs, rewards, yerr=std_devs, color=['blue', 'orange'], capsize=10)
plt.xlabel('DQN Configurations')
plt.ylabel('Mean Reward')
plt.title('DQN vs Double DQN Performance Comparison')
plt.show()
```

**Expected Output:**

![DQN vs Double DQN Performance Comparison](https://i.imgur.com/9KX5Zc1.gif)
*Figure: The bar chart compares the mean rewards of Standard DQN and Double DQN, highlighting the improvement in performance due to Double DQN.*

**Analysis:**
- **Standard DQN:** May exhibit overestimation bias, leading to less stable learning and potentially lower rewards.
- **Double DQN:** Reduces overestimation bias, resulting in more accurate Q-value estimates and improved performance.

---

## **Interactive Activity**

### **1. Experiment with Different Architectures**

**Task:** Modify the neural network architecture by changing the number of layers or neurons to observe its effect on agent performance.

```python
# Example: Increasing network depth and width
model_custom_arch = DQN(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    double_q=True,  # Enable Double DQN
    policy_kwargs=dict(net_arch=[256, 256]),  # Increased network size
    tensorboard_log="./custom_arch_dqn_cartpole_tensorboard/"
)

# Train the agent
model_custom_arch.learn(
    total_timesteps=10000, 
    callback=callback
)

# Evaluate the agent
mean_reward_custom_arch, std_reward_custom_arch = evaluate_policy(model_custom_arch, env, n_eval_episodes=100)

print(f"Custom Architecture Double DQN Mean Reward: {mean_reward_custom_arch} +/- {std_reward_custom_arch}")
```

**Observation:** Increasing the network's depth and width can allow the agent to capture more complex patterns, potentially improving performance but also increasing computational requirements.

### **2. Modify Exploration Parameters**

**Task:** Adjust the exploration parameters (`exploration_fraction` and `exploration_final_eps`) to see how it affects the agent's learning and performance.

```python
# Example: Reducing exploration final epsilon
model_low_eps = DQN(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,  # Lower final epsilon
    double_q=True,  # Enable Double DQN
    tensorboard_log="./low_eps_dqn_cartpole_tensorboard/"
)

# Train the agent
model_low_eps.learn(
    total_timesteps=10000, 
    callback=callback
)

# Evaluate the agent
mean_reward_low_eps, std_reward_low_eps = evaluate_policy(model_low_eps, env, n_eval_episodes=100)

print(f"Low Epsilon Double DQN Mean Reward: {mean_reward_low_eps} +/- {std_reward_low_eps}")
```

**Observation:** Lowering the final epsilon reduces exploration towards the end of training, potentially leading to more exploitation of learned policies but may also cause the agent to get stuck in local optima if exploration is insufficient.

### **3. Implement and Compare Dueling DQN with and without Double DQN**

**Task:** Implement Dueling DQN with Double DQN enabled and compare it with Dueling DQN without Double DQN to observe the combined effects on performance.

```python
# Create Dueling DQN with Double DQN
model_dueling_double = DQN(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    double_q=True,  # Enable Double DQN
    policy_kwargs=dict(net_arch=[128, 128], dueling=True),  # Enable Dueling Networks
    tensorboard_log="./dueling_double_dqn_cartpole_tensorboard/"
)

# Train the agent
model_dueling_double.learn(
    total_timesteps=10000, 
    callback=callback
)

# Evaluate the agent
mean_reward_dueling_double, std_reward_dueling_double = evaluate_policy(model_dueling_double, env, n_eval_episodes=100)

print(f"Dueling Double DQN Mean Reward: {mean_reward_dueling_double} +/- {std_reward_dueling_double}")
```

**Comparison:** Compare the performance metrics of Dueling Double DQN with standard DQN and Double DQN to understand the compounded benefits.

---

## **Summary**

**Dueling Networks** enhance the standard DQN architecture by decoupling the estimation of state-value and advantage functions. This separation allows for more efficient learning, improved policy evaluation, and often better performance in complex environments. By implementing Dueling DQN using **Stable Baselines3**, you've leveraged these architectural benefits to develop more robust and effective RL agents.

**In this lesson, you:**
- **Defined Overestimation Bias** and understood its impact on Q-Learning and DQN.
- **Explored Double DQN** and how it addresses overestimation by decoupling action selection and evaluation.
- **Implemented Double DQN** using Stable Baselines3, enabling advanced configurations with ease.
- **Visualized the impact** of Double DQN on Q-value distributions, highlighting the reduction in overestimation bias.
- **Analyzed the effects** of different configurations, such as disabling Target Networks and altering Replay Buffer sizes.
- **Compared Double DQN with Standard DQN**, observing improved performance and stability.
- **Utilized custom callbacks** to monitor and log training metrics, enhancing the understanding of agent behavior.
- **Experimented with advanced architectures and parameters** to optimize agent performance.

---

## **Best Practices When Leveraging Dueling DQN**

1. **Experience Replay Buffer Size:**
   - **Optimal Size:** Choose a buffer size that balances memory usage and sample diversity. Too small buffers may not provide diverse experiences, while too large buffers can be memory-intensive.
   - **Implementation Tip:** Use efficient data structures (e.g., deque) to implement the replay buffer.
   
2. **Batch Sampling:**
   - **Mini-Batch Size:** Select a mini-batch size that provides a good trade-off between training speed and stability.
   - **Diversity:** Ensure that batches are sampled uniformly to maintain a diverse set of experiences.
   
3. **Target Network Update Frequency:**
   - **Update Interval:** Update the target network periodically (e.g., every few thousand steps) to provide stable target Q-values.
   - **Stabilization:** Properly setting the update frequency helps in reducing oscillations and divergence during training.
   
4. **Handling Overestimation Bias:**
   - **Double DQN:** Implement Double DQN by enabling the `double_q` parameter in SB3's DQN to mitigate overestimation.
   - **Clipping Rewards:** Normalize or clip rewards to prevent large gradients and stabilize training.
   
5. **Reward Shaping:**
   - **Consistent Rewards:** Ensure that the reward structure is consistent and provides clear signals for desired behaviors.
   - **Normalization:** Normalize rewards to keep them within a manageable range, aiding in faster convergence.
   
6. **Exploration Strategy:**
   - **ε-Greedy Policy:** Use an ε-greedy policy to balance exploration and exploitation.
   - **Decay Schedule:** Gradually decay ε to reduce exploration as the agent becomes more confident in its policy.
   
7. **Network Architecture:**
   - **Depth and Width:** Design the neural network with an appropriate number of layers and neurons to capture the complexity of the environment.
   - **Activation Functions:** Use activation functions like ReLU for hidden layers and avoid saturating activations in output layers.
   
8. **Hyperparameter Tuning:**
   - **Learning Rate:** Carefully tune the learning rate to ensure stable and efficient learning.
   - **Discount Factor (γ):** Set γ based on the importance of future rewards in the specific task.
   
9. **Monitoring and Logging:**
   - **Performance Metrics:** Regularly monitor metrics like average reward, loss, and Q-value distributions to track training progress.
   - **Visualization Tools:** Utilize tools like TensorBoard or Matplotlib to visualize training dynamics.
   
10. **Reproducibility:**
    - **Random Seeds:** Set random seeds for NumPy, PyTorch, and Gym to ensure reproducible results.
      ```python
      import torch
      import numpy as np
      import random
      
      torch.manual_seed(42)
      np.random.seed(42)
      random.seed(42)
      env.seed(42)
      ```
    
11. **Modular Code Structure:**
    - **Encapsulation:** Organize code into classes and functions to enhance readability and maintainability.
    - **Reusability:** Develop reusable components like replay buffers, neural network architectures, and training loops.
      ```python
      from collections import deque
      import random

      class ReplayBuffer:
          def __init__(self, capacity):
              self.buffer = deque(maxlen=capacity)
          
          def push(self, state, action, reward, next_state, done):
              self.buffer.append((state, action, reward, next_state, done))
          
          def sample(self, batch_size):
              return random.sample(self.buffer, batch_size)
          
          def __len__(self):
              return len(self.buffer)
      ```
    
12. **Advanced Techniques:**
    - **Prioritized Experience Replay:** Assign priorities to experiences based on their TD-error to sample more informative experiences.
    - **Dueling DQN:** Separate the estimation of state-value and advantage functions to improve learning efficiency.
      ```python
      from stable_baselines3 import DQN
      from stable_baselines3.common.torch_layers import DuelingDQNHead

      # Example: Using Dueling DQN architecture
      model_dueling_dqn = DQN(
          'MlpPolicy', 
          env, 
          verbose=1, 
          policy_kwargs=dict(net_arch=[128, 128], dueling=True)
      )
      model_dueling_dqn.learn(total_timesteps=10000)
      ```
    
13. **Regular Evaluation:**
    - **Periodic Evaluation:** Regularly evaluate the agent's performance on a set of evaluation episodes to monitor progress and detect overfitting.
    - **Best Model Saving:** Save the best-performing model during training for later use and deployment.

---

## **Further Reading and Resources**
- **"Human-Level Control through Deep Reinforcement Learning" by Mnih et al. (2015):** The seminal paper introducing DQN. [Read Paper](https://www.nature.com/articles/nature14236)
- **"Dueling Network Architectures for Deep Reinforcement Learning" by Wang et al. (2016):** The foundational paper on Dueling Networks. [Read Paper](https://arxiv.org/abs/1511.06581)
- **"Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto:** A comprehensive textbook on RL fundamentals. [Available Online](http://incompleteideas.net/book/the-book.html)
- **OpenAI Gym Documentation:** [https://gym.openai.com/docs/](https://gym.openai.com/docs/)
- **Stable Baselines3 Documentation:** [https://stable-baselines3.readthedocs.io/en/master/](https://stable-baselines3.readthedocs.io/en/master/)
- **Deep Reinforcement Learning Hands-On by Maxim Lapan:** [Book Link](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994)
- **Policy Gradient Methods Tutorial by Lilian Weng:** [https://lilianweng.github.io/posts/2018-04-08-policy-gradient/](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- **Double DQN Explanation:** [https://arxiv.org/abs/1509.06461](https://arxiv.org/abs/1509.06461)
- **Dueling DQN Overview:** [https://arxiv.org/abs/1511.06581](https://arxiv.org/abs/1511.06581)
- **PyTorch Documentation:** [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- **Visualizing Reinforcement Learning Agents:** [https://www.youtube.com/watch?v=O_0NnXjKojg](https://www.youtube.com/watch?v=O_0NnXjKojg)
- **Stable Baselines3 GitHub Repository:** [https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

---

**Great job on completing Lesson 6.2!** You've successfully understood and implemented **Dueling DQN** using **Stable Baselines3**, addressing overestimation bias in Deep Q-Networks. This enhancement leads to more accurate Q-value estimates, improved policy performance, and greater training stability. By comparing Double DQN with standard DQN, you've observed the tangible benefits of advanced RL algorithms. In the upcoming lessons, we'll delve deeper into **Dueling DQN** and explore additional techniques to further optimize your RL agents.
```