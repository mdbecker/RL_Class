```markdown
# **Lesson 5.3: Experience Replay and Target Networks**

## **Learning Objectives**
- **Implement Experience Replay** to stabilize DQN training.
- **Understand the role of Target Networks** in Deep Q-Networks.
- **Enhance learning stability and performance** by integrating Experience Replay and Target Networks.
- **Analyze the impact of these components** on the convergence and efficiency of the DQN agent.
- **Visualize the effects** of Experience Replay and Target Networks on training dynamics.
- **Experiment with different configurations** to observe their influence on agent performance.

## **Description**
In this lesson, we'll explore two critical components that enhance the performance and stability of Deep Q-Networks (DQN): **Experience Replay** and **Target Networks**. We'll implement these components using **Stable Baselines3 (SB3)**, understand their roles in mitigating common RL challenges, and observe their effects on agent training. By integrating Experience Replay and Target Networks, we'll demonstrate how they contribute to more efficient and reliable learning in DQN agents.

## **Setting Up the Environment**

Ensure you are in the `rl_week5` Conda environment with the necessary packages installed. If not, refer to **Lesson 5.1** for setup instructions.

### **Step 1: Activate the Environment**

```bash
# Activate the 'rl_week5' environment
conda activate rl_week5
```

### **Step 2: Launch JupyterLab**

```bash
# Launch JupyterLab
jupyter lab
```

*JupyterLab will open in your default web browser, providing an interactive environment for coding and visualization.*

---

## **Understanding Experience Replay and Target Networks**

### **What is Experience Replay?**

**Experience Replay** is a technique where agents store their experiences \( (s, a, r, s') \) in a replay buffer and sample mini-batches of experiences randomly during training. This approach offers several benefits:

- **Breaks Correlations:** Random sampling breaks the temporal correlations between consecutive experiences, leading to more stable and efficient learning.
- **Reusability of Data:** Experiences are reused multiple times, improving sample efficiency.
- **Reduces Variance:** By averaging over a diverse set of experiences, learning updates become less noisy.

### **What are Target Networks?**

**Target Networks** are a copy of the main Q-network used to compute target Q-values during training. They are updated less frequently than the main network, which helps stabilize training by providing consistent target values.

- **Main Network (\( Q(s, a; \theta) \)):** The network being trained to approximate the Q-function.
- **Target Network (\( Q(s, a; \theta^-) \)):** A fixed network used to compute target Q-values. It is updated periodically with the main network's weights.

### **How They Enhance DQN**

- **Experience Replay:** Prevents the agent from learning from correlated data, leading to more generalized and robust Q-value estimates.
- **Target Networks:** Mitigate the issue of moving targets, reducing oscillations and divergence during training.

### **Practical Example: Implementing Experience Replay and Target Networks**

Let's implement these components in a DQN agent using Stable Baselines3.

---

## **Implementing Experience Replay and Target Networks with Stable Baselines3**

Stable Baselines3 integrates Experience Replay and Target Networks internally. However, understanding how they work conceptually is essential. We'll configure these components and observe their impact on training.

### **Step 1: Import Necessary Libraries**

Begin by importing the essential libraries required for implementing and training the DQN agent.

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
- **`StopTrainingOnRewardThreshold`:** Stops training early if a specified reward threshold is achieved.

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
    
    # Train the agent with TensorBoard callback
    model.learn(
        total_timesteps=10000, 
        callback=CallbackList([eval_callback, stop_callback, custom_logging_callback])
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
plt.plot(custom_logging_callback.episode_rewards, label='Episode Reward', color='blue')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN: Episode Rewards Over Time')
plt.legend()
plt.grid(True)
plt.show()
```

**Expected Output:**

![DQN Episode Rewards Over Time](https://i.imgur.com/9KX5Zc1.gif)

*Figure: The plot shows the rewards obtained by the agent in each episode, indicating learning progress.*

---

## **Analyze the Impact of Experience Replay and Target Networks**

Compare the training performance with different configurations to understand the impact of Experience Replay and Target Networks.

### **Example: Disabling Target Networks**

```python
# Create a DQN agent without Target Networks
model_no_target = DQN(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=0,   # Disable target network updates
    exploration_fraction=0.1,
    exploration_final_eps=0.02
)

# Train the agent
model_no_target.learn(
    total_timesteps=10000, 
    callback=CallbackList([eval_callback, stop_callback, custom_logging_callback])
)

# Evaluate the agent
mean_reward_no_target, std_reward_no_target = evaluate_policy(model_no_target, env, n_eval_episodes=100)

print(f"Mean Reward without Target Networks: {mean_reward_no_target} +/- {std_reward_no_target}")
```

### **Example: Reducing Replay Buffer Size**

```python
# Create a DQN agent with a smaller replay buffer
model_small_buffer = DQN(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=1e-3,
    buffer_size=10000,           # Smaller buffer
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02
)

# Train the agent
model_small_buffer.learn(
    total_timesteps=10000, 
    callback=CallbackList([eval_callback, stop_callback, custom_logging_callback])
)

# Evaluate the agent
mean_reward_small_buffer, std_reward_small_buffer = evaluate_policy(model_small_buffer, env, n_eval_episodes=100)

print(f"Mean Reward with Small Replay Buffer: {mean_reward_small_buffer} +/- {std_reward_small_buffer}")
```

### **Step 9: Compare Performance Metrics**

Compare the performance of agents trained with different configurations to understand the benefits of Experience Replay and Target Networks.

```python
# Example: Plotting mean rewards
configs = ['Standard DQN', 'No Target Networks', 'Small Replay Buffer']
rewards = [mean_reward, mean_reward_no_target, mean_reward_small_buffer]

plt.figure(figsize=(10,6))
plt.bar(configs, rewards, color=['blue', 'orange', 'green'])
plt.xlabel('DQN Configurations')
plt.ylabel('Mean Reward')
plt.title('DQN Performance Comparison')
plt.show()
```

**Expected Output:**

![DQN Performance Comparison](https://i.imgur.com/9KX5Zc1.gif)

*Figure: The bar chart compares the mean rewards of different DQN configurations, highlighting the impact of Experience Replay and Target Networks on performance.*

---

## **Interactive Activity**

### **1. Experiment with Target Network Update Intervals**

**Task:** Change `target_update_interval` to observe its effect on training stability and performance.

```python
# Example: More frequent target updates
model = DQN(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=500,  # More frequent updates
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    tensorboard_log="./dqn_cartpole_tensorboard_freq500/"
)

# Train the agent
model.learn(
    total_timesteps=10000, 
    callback=CallbackList([eval_callback, stop_callback, custom_logging_callback])
)
```

**Observation:** More frequent updates of the Target Network can lead to faster convergence but may introduce instability if updates are too frequent. Conversely, less frequent updates provide more stable targets but may slow down learning.

### **2. Adjust Replay Buffer Size**

**Task:** Increase or decrease `buffer_size` to see how it influences learning efficiency and agent performance.

```python
# Example: Larger replay buffer
model = DQN(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=1e-3,
    buffer_size=100000,          # Larger buffer
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    tensorboard_log="./dqn_cartpole_tensorboard_large_buffer/"
)

# Train the agent
model.learn(
    total_timesteps=10000, 
    callback=CallbackList([eval_callback, stop_callback, custom_logging_callback])
)
```

**Observation:** A larger replay buffer allows the agent to learn from a more diverse set of experiences, potentially improving generalization and performance. However, it also increases memory usage and may require more training steps to fully utilize the stored experiences.

### **3. Compare with Other Algorithms**

**Task:** Experiment with other policy-based algorithms like **A2C** or **SAC** provided by SB3 and compare their performance with DQN.

```python
from stable_baselines3 import A2C, SAC

# Initialize the A2C agent
a2c_model = A2C('MlpPolicy', env, verbose=1, learning_rate=1e-3, gamma=0.99, tensorboard_log="./a2c_cartpole_tensorboard/")
a2c_model.learn(total_timesteps=10000)

# Evaluate the A2C agent
mean_reward_a2c, std_reward_a2c = evaluate_policy(a2c_model, env, n_eval_episodes=100)
print(f"A2C Mean Reward: {mean_reward_a2c} +/- {std_reward_a2c}")

# Initialize the SAC agent (note: SAC is typically used for continuous action spaces)
# For demonstration, we'll use Pendulum-v1 which has a continuous action space
sac_env = gym.make('Pendulum-v1')
sac_model = SAC('MlpPolicy', sac_env, verbose=1, learning_rate=3e-4, tensorboard_log="./sac_pendulum_tensorboard/")
sac_model.learn(total_timesteps=10000)

# Evaluate the SAC agent
mean_reward_sac, std_reward_sac = evaluate_policy(sac_model, sac_env, n_eval_episodes=100)
print(f"SAC Mean Reward: {mean_reward_sac} +/- {std_reward_sac}")
```

**Observation:** Different algorithms excel in different environments. **A2C** is an on-policy method that can be faster in some environments, while **SAC** is suited for continuous action spaces. Comparing their performance helps in selecting the right algorithm for specific tasks.

### **4. Implement Custom Callbacks**

**Task:** Create custom callbacks to log additional metrics or implement advanced training strategies.

```python
from stable_baselines3.common.callbacks import BaseCallback

class AdvancedLoggingCallback(BaseCallback):
    """
    Custom callback for logging additional metrics during training.
    """
    def __init__(self, verbose=0):
        super(AdvancedLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Check if a new episode has started
        if 'episode_rewards' in self.locals:
            self.episode_rewards.append(self.locals['episode_rewards'])
            self.episode_lengths.append(self.locals['episode_lengths'])
        return True

# Initialize the advanced logging callback
advanced_logging_callback = AdvancedLoggingCallback()

# Train the agent with the advanced callback
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
    tensorboard_log="./dqn_cartpole_tensorboard_advanced/"
)
model.learn(
    total_timesteps=10000, 
    callback=CallbackList([eval_callback, stop_callback, advanced_logging_callback])
)

# Plot the logged rewards
plt.figure(figsize=(12,6))
plt.plot(advanced_logging_callback.episode_rewards, label='Episode Reward', color='purple')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN: Episode Rewards Over Time with Advanced Logging')
plt.legend()
plt.grid(True)
plt.show()
```

**Explanation:**
- **`AdvancedLoggingCallback`:** Extends SB3's `BaseCallback` to log episode rewards and lengths for custom visualization.
- **Visualization:** Provides insights into the agent's performance trends over episodes, allowing for deeper analysis of training dynamics.

---

## **Summary**

Integrating **Experience Replay** and **Target Networks** significantly enhances the stability and performance of Deep Q-Networks. Experience Replay ensures that the agent learns from a diverse set of experiences, breaking temporal correlations and improving sample efficiency. Target Networks provide consistent target Q-values, reducing oscillations and convergence issues during training. By implementing and experimenting with these components using **Stable Baselines3**, you've reinforced your understanding of advanced DQN mechanisms and their practical benefits in Reinforcement Learning.

**In this lesson, you:**
- **Defined Experience Replay** and **Target Networks** and understood their significance in DQN.
- **Configured and trained** a DQN agent using Stable Baselines3, integrating Experience Replay and Target Networks.
- **Implemented custom callbacks** to monitor and log training metrics for enhanced visualization.
- **Analyzed the impact** of Experience Replay and Target Networks by comparing different DQN configurations.
- **Experimented with different environments and hyperparameters**, observing their influence on agent performance.
- **Compared DQN with other algorithms** like A2C and SAC to understand their relative strengths.

---

## **Best Practices When Leveraging Deep Q-Networks**

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
   - **Double DQN:** Implement Double DQN to mitigate overestimation of Q-values by decoupling action selection and evaluation.
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
      model = DQN(
          'MlpPolicy', 
          env, 
          verbose=1, 
          policy_kwargs=dict(net_arch=[128, 128], dueling=True)
      )
      model.learn(total_timesteps=10000)
      ```

---

## **Further Reading and Resources**
- **"Human-Level Control through Deep Reinforcement Learning" by Mnih et al. (2015):** The seminal paper introducing DQN.
- **"Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto:** A comprehensive textbook on RL fundamentals.
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

**Great job on completing Lesson 5.3!** You've successfully implemented and understood the critical components of **Experience Replay** and **Target Networks** in Deep Q-Networks using **Stable Baselines3**. This hands-on experience enhances your ability to develop and optimize deep RL agents effectively, paving the way for tackling more complex environments and advanced RL algorithms in future lessons.
```