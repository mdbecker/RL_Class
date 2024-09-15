```markdown
# **Lesson 6.3: Prioritized Experience Replay**

## **Learning Objectives**
- **Grasp the concept of Prioritized Experience Replay (PER).**
- **Understand how prioritized sampling enhances learning efficiency.**
- **Implement Prioritized Experience Replay** using Stable Baselines3.
- **Evaluate the impact of PER** on agent performance compared to uniform sampling.
- **Visualize the benefits** of PER in focusing learning on significant experiences.
- **Experiment with PER parameters** to optimize agent learning.

## **Description**
In this lesson, we'll delve into **Prioritized Experience Replay (PER)**, an advanced technique that improves the efficiency of Experience Replay by prioritizing important transitions. Standard Experience Replay samples experiences uniformly, which can be inefficient as not all experiences are equally valuable for learning. PER assigns higher sampling probabilities to transitions with higher **Temporal-Difference (TD) errors**, ensuring that the agent focuses on learning from more informative experiences. We'll implement PER using **Stable Baselines3 (SB3)** and evaluate its impact on agent performance, comparing it to the standard uniform sampling approach.

## **Setting Up the Environment**

Ensure you are in the `rl_week6` Conda environment with the necessary packages installed. If not, follow the setup instructions below.

### **Step 1: Activate the Environment**

```bash
# Activate the 'rl_week6' environment
conda activate rl_week6
```

### **Step 2: Launch JupyterLab**

```bash
# Launch JupyterLab
jupyter lab
```

*JupyterLab will open in your default web browser, providing an interactive environment for coding and visualization.*

---

## **What is Prioritized Experience Replay?**

**Prioritized Experience Replay (PER)** enhances the standard Experience Replay by prioritizing the sampling of important experiences. Instead of sampling transitions uniformly, PER assigns higher probabilities to transitions with larger **Temporal-Difference (TD) errors**, indicating that these experiences are more informative for learning.

### **Benefits of PER**
- **Improved Sample Efficiency:** Focuses learning on more significant transitions, leading to faster convergence.
- **Enhanced Learning Stability:** Reduces the likelihood of the agent getting stuck in suboptimal policies by emphasizing critical experiences.
- **Better Performance:** Empirical results show that PER can lead to higher rewards and more stable training compared to uniform sampling.

### **How PER Works**
1. **Assign Priorities:** Each transition \( (s, a, r, s') \) is assigned a priority based on its TD error \( \delta \):
   \[
   p_i = |\delta_i| + \epsilon
   \]
   where \( \epsilon \) is a small positive constant to ensure all transitions have non-zero probability.

2. **Sampling Probabilities:** Transitions are sampled with probabilities proportional to their priorities:
   \[
   P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
   \]
   where \( \alpha \) determines the level of prioritization ( \( \alpha = 0 \) corresponds to uniform sampling).

3. **Importance Sampling Weights:** To correct for the non-uniform probabilities, importance sampling weights \( w_i \) are applied:
   \[
   w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta
   \]
   where \( N \) is the total number of transitions, and \( \beta \) is annealed from a small value to 1 during training.

### **Practical Example: Visualizing PER**

Let's visualize how PER prioritizes transitions based on TD errors by simulating TD errors and computing sampling probabilities.

```python
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Simulated TD errors
td_errors = np.abs(np.random.normal(loc=0.0, scale=1.0, size=1000))

# Assign priorities with a small epsilon to ensure no zero probabilities
epsilon = 0.01
priorities = td_errors + epsilon

# Calculate sampling probabilities with alpha=0.6
alpha = 0.6
probabilities = priorities ** alpha
probabilities /= probabilities.sum()

# Plot the distribution of sampling probabilities
plt.figure(figsize=(10,6))
plt.hist(probabilities, bins=50, alpha=0.7, color='green')
plt.title('Sampling Probability Distribution with PER (alpha=0.6)')
plt.xlabel('Sampling Probability')
plt.ylabel('Frequency')
plt.show()

# Highlight high priority samples (top 5%)
high_priority_indices = np.argsort(td_errors)[-50:]
high_priorities = probabilities[high_priority_indices]

plt.figure(figsize=(10,6))
plt.hist(high_priorities, bins=50, alpha=0.7, color='red')
plt.title('High Priority Sampling Probabilities (Top 5% TD Errors)')
plt.xlabel('Sampling Probability')
plt.ylabel('Frequency')
plt.show()
```

### **Explanation**
- **Simulated TD Errors:** We simulate TD errors to represent the importance of transitions. Larger TD errors indicate more significant learning opportunities.
- **Priority Assignment:** Priorities are assigned by adding a small constant \( \epsilon \) to TD errors to ensure all transitions have a non-zero probability.
- **Sampling Probabilities:** Calculated using \( \alpha = 0.6 \) to determine the level of prioritization. Higher \( \alpha \) increases the emphasis on high-priority transitions.
- **Visualization:** Histograms show how PER assigns higher probabilities to transitions with larger TD errors, emphasizing their importance during sampling.

**Expected Output:**

![PER Sampling Probability Distribution](https://i.imgur.com/9KX5Zc1.gif)
*Figure 1: Sampling Probability Distribution with PER (alpha=0.6)*

![High Priority Sampling Probabilities](https://i.imgur.com/9KX5Zc1.gif)
*Figure 2: High Priority Sampling Probabilities (Top 5% TD Errors)*

---

## **Implementing Prioritized Experience Replay with Stable Baselines3**

Stable Baselines3 (SB3) integrates Prioritized Experience Replay (PER) into its DQN implementation, allowing you to enable it through configuration parameters. We'll configure SB3's DQN to use PER and compare its performance with standard DQN.

### **Step 1: Import Necessary Libraries**

Begin by importing the essential libraries required for implementing and training the DQN agent with PER.

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

### **Step 3: Configure the DQN Agent with PER and Target Networks**

```python
# Create the DQN agent with Experience Replay, Target Networks, and PER
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
    prioritized_replay=True,       # Enable Prioritized Experience Replay
    prioritized_replay_alpha=0.6,  # PER exponent alpha
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
- **`prioritized_replay`:** Enables Prioritized Experience Replay.
- **`prioritized_replay_alpha`:** Determines the level of prioritization ( \( \alpha = 0 \) corresponds to uniform sampling).
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

### **Step 5: Train the DQN Agent with PER and Callbacks**

```python
# Train the DQN agent with PER and callbacks
model.learn(
    total_timesteps=10000, 
    callback=callback
)
```

**Training Process:**
- The agent interacts with the environment, storing experiences in the replay buffer.
- Experiences are sampled based on priority from the buffer to train the Q-Network.
- The Target Network is updated at specified intervals to provide stable target Q-values.
- Callbacks monitor performance, log metrics, and implement early stopping based on rewards.

### **Step 6: Evaluate the Trained Agent**

After training, it's essential to evaluate the agent's performance to understand how well it has learned to solve the environment.

```python
# Evaluate the agent's performance
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"Double DQN with PER Mean Reward: {mean_reward} +/- {std_reward}")
```

**Expected Output:**
```
Double DQN with PER Mean Reward: 200.0 +/- 0.0
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
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
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
plt.title('DQN with PER: Episode Rewards Over Time')
plt.legend()
plt.grid(True)
plt.show()
```

**Expected Output:**

![DQN Episode Rewards Over Time](https://i.imgur.com/9KX5Zc1.gif)
*Figure: The plot shows the rewards obtained by the agent in each episode, indicating learning progress.*

---

## **Compare PER-enhanced DQN with Standard DQN**

To observe the benefits of PER, we'll train two agents: one using standard uniform sampling and another using PER. We'll then compare their performance.

### **Step 1: Train Standard DQN (Uniform Sampling)**

```python
# Create the standard DQN agent without PER
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
    prioritized_replay=False,      # Disable PER
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    tensorboard_log="./standard_dqn_cartpole_tensorboard/"
)

# Train the standard DQN agent
model_standard_dqn.learn(
    total_timesteps=10000, 
    callback=callback
)

# Evaluate the standard DQN agent
mean_reward_standard, std_reward_standard = evaluate_policy(model_standard_dqn, env, n_eval_episodes=100)

print(f"Standard DQN Mean Reward: {mean_reward_standard} +/- {std_reward_standard}")
```

### **Step 2: Train DQN with PER**

```python
# Create the DQN agent with PER
model_per_dqn = DQN(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    prioritized_replay=True,       # Enable PER
    prioritized_replay_alpha=0.6,  # PER exponent alpha
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    tensorboard_log="./per_dqn_cartpole_tensorboard/"
)

# Train the DQN agent with PER
model_per_dqn.learn(
    total_timesteps=10000, 
    callback=callback
)

# Evaluate the DQN agent with PER
mean_reward_per, std_reward_per = evaluate_policy(model_per_dqn, env, n_eval_episodes=100)

print(f"DQN with PER Mean Reward: {mean_reward_per} +/- {std_reward_per}")
```

### **Step 3: Compare Performance Metrics**

```python
# Compare Mean Rewards
configs = ['Standard DQN', 'DQN with PER']
rewards = [mean_reward_standard, mean_reward_per]
std_devs = [std_reward_standard, std_reward_per]

plt.figure(figsize=(10,6))
plt.bar(configs, rewards, yerr=std_devs, color=['gray', 'green'], capsize=10)
plt.xlabel('DQN Configurations')
plt.ylabel('Mean Reward')
plt.title('Standard DQN vs. DQN with Prioritized Experience Replay')
plt.show()
```

**Expected Output:**

![DQN Performance Comparison](https://i.imgur.com/9KX5Zc1.gif)
*Figure: The bar chart compares the mean rewards of Standard DQN and DQN with PER, highlighting the improvement in performance due to PER.*

**Analysis:**
- **Standard DQN:** May exhibit lower mean rewards due to uniform sampling, leading to less efficient learning.
- **DQN with PER:** Demonstrates higher mean rewards by focusing learning on more informative transitions, resulting in faster convergence and better performance.

---

## **Interactive Activity**

### **1. Experiment with Different PER Parameters**

**Task:** Adjust the prioritization exponent \( \alpha \) and the importance-sampling exponent \( \beta \) to observe their effects on agent performance.

```python
# Example: Increase PER alpha to 0.8
model_per_alpha = DQN(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    prioritized_replay=True,       # Enable PER
    prioritized_replay_alpha=0.8,  # Increased PER exponent alpha
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    tensorboard_log="./per_alpha_dqn_cartpole_tensorboard/"
)

# Train the agent
model_per_alpha.learn(
    total_timesteps=10000, 
    callback=callback
)

# Evaluate the agent
mean_reward_per_alpha, std_reward_per_alpha = evaluate_policy(model_per_alpha, env, n_eval_episodes=100)

print(f"DQN with PER (alpha=0.8) Mean Reward: {mean_reward_per_alpha} +/- {std_reward_per_alpha}")
```

**Observation:** Increasing \( \alpha \) places more emphasis on high TD error transitions, potentially leading to faster learning but may also introduce bias if set too high.

### **2. Modify the Replay Buffer Size**

**Task:** Change the `buffer_size` parameter to see how it influences learning efficiency and agent performance.

```python
# Example: Reduce replay buffer size to 10000
model_per_small_buffer = DQN(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=1e-3,
    buffer_size=10000,           # Reduced buffer size
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    prioritized_replay=True,      # Enable PER
    prioritized_replay_alpha=0.6, 
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    tensorboard_log="./per_small_buffer_dqn_cartpole_tensorboard/"
)

# Train the agent
model_per_small_buffer.learn(
    total_timesteps=10000, 
    callback=callback
)

# Evaluate the agent
mean_reward_per_small_buffer, std_reward_per_small_buffer = evaluate_policy(model_per_small_buffer, env, n_eval_episodes=100)

print(f"DQN with PER (Small Buffer) Mean Reward: {mean_reward_per_small_buffer} +/- {std_reward_per_small_buffer}")
```

**Observation:** A smaller replay buffer may lead to less diverse experiences being available for training, potentially slowing down learning and reducing performance.

### **3. Compare with Other Algorithms**

**Task:** Experiment with other value-based algorithms like **Double DQN** and **Dueling DQN** provided by SB3 and compare their performance with PER-enhanced DQN.

```python
from stable_baselines3 import DQN

# Create the Dueling DQN agent with PER
model_dueling_per = DQN(
    'MlpPolicy', 
    env, 
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    prioritized_replay=True,       # Enable PER
    prioritized_replay_alpha=0.6,  
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    policy_kwargs=dict(net_arch=[128, 128], dueling=True),  # Enable Dueling Networks
    tensorboard_log="./dueling_per_dqn_cartpole_tensorboard/"
)

# Train the Dueling DQN agent with PER
model_dueling_per.learn(
    total_timesteps=10000, 
    callback=callback
)

# Evaluate the agent
mean_reward_dueling_per, std_reward_dueling_per = evaluate_policy(model_dueling_per, env, n_eval_episodes=100)

print(f"Dueling DQN with PER Mean Reward: {mean_reward_dueling_per} +/- {std_reward_dueling_per}")
```

**Explanation:**
- **`dueling=True`:** Enables the Dueling Network architecture within SB3's DQN implementation.
- **Expected Outcome:** The Dueling DQN with PER should exhibit improved performance compared to standard DQN and DQN with only PER, leveraging both architectural and sampling enhancements.

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

# Combine callbacks
advanced_callback = CallbackList([eval_callback, stop_callback, advanced_logging_callback])

# Train the agent with the advanced callback
model_per_dqn.learn(
    total_timesteps=10000, 
    callback=advanced_callback
)

# Plot the logged rewards
plt.figure(figsize=(12,6))
plt.plot(advanced_logging_callback.episode_rewards, label='Episode Reward', color='purple')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN with PER: Episode Rewards Over Time with Advanced Logging')
plt.legend()
plt.grid(True)
plt.show()
```

**Explanation:**
- **`AdvancedLoggingCallback`:** Extends SB3's `BaseCallback` to log episode rewards and lengths for custom visualization.
- **Visualization:** Provides insights into the agent's performance trends over episodes, allowing for deeper analysis of training dynamics.

**Expected Output:**
```
Episode 1: Total Reward: 200
Episode 2: Total Reward: 200
Episode 3: Total Reward: 200
...
```

*Figure: The plot shows the rewards obtained by the DQN agent with PER in each episode, indicating learning progress.*

---

## **Summary**

**Prioritized Experience Replay (PER)** significantly enhances the efficiency and effectiveness of Experience Replay by focusing learning on more informative transitions. By prioritizing experiences with higher TD errors, PER ensures that the agent learns from the most critical experiences, leading to faster convergence and improved performance. Implementing PER using **Stable Baselines3** allows you to leverage these benefits seamlessly, enhancing your Deep Q-Networks for more robust Reinforcement Learning applications.

**In this lesson, you:**
- **Defined Prioritized Experience Replay (PER)** and understood its significance in improving learning efficiency.
- **Explored how PER works** by assigning priorities based on TD errors and adjusting sampling probabilities.
- **Implemented PER** using Stable Baselines3's DQN with configuration parameters.
- **Visualized the impact** of PER on sampling probabilities, highlighting its focus on significant transitions.
- **Compared PER-enhanced DQN with standard DQN**, observing improved performance and stability.
- **Experimented with different PER parameters** to optimize agent learning and performance.
- **Combined PER with advanced architectures** like Dueling Networks to further enhance agent capabilities.
- **Utilized custom callbacks** to monitor and log training metrics, providing deeper insights into agent behavior.

---

## **Best Practices When Leveraging Prioritized Experience Replay**

1. **Experience Replay Buffer Size:**
   - **Optimal Size:** Choose a buffer size that balances memory usage and sample diversity. Too small buffers may not provide diverse experiences, while too large buffers can be memory-intensive.
   - **Implementation Tip:** Use efficient data structures (e.g., deque) to implement the replay buffer.

2. **Batch Sampling:**
   - **Mini-Batch Size:** Select a mini-batch size that provides a good trade-off between training speed and stability.
   - **Diversity:** Ensure that batches are sampled uniformly to maintain a diverse set of experiences.

3. **Prioritization Exponent (\( \alpha \)):**
   - **Role:** Determines the level of prioritization. \( \alpha = 0 \) corresponds to uniform sampling, while \( \alpha = 1 \) fully prioritizes based on TD errors.
   - **Recommendation:** Start with \( \alpha = 0.6 \) and adjust based on performance.

4. **Importance Sampling Exponent (\( \beta \)):**
   - **Role:** Corrects the bias introduced by prioritized sampling. \( \beta \) is annealed from a small value to 1 during training.
   - **Recommendation:** Start with \( \beta = 0.4 \) and increase it to 1 as training progresses.

5. **Target Network Update Frequency:**
   - **Update Interval:** Update the target network periodically (e.g., every few thousand steps) to provide stable target Q-values.
   - **Stabilization:** Properly setting the update frequency helps in reducing oscillations and divergence during training.

6. **Handling Overestimation Bias:**
   - **Double DQN:** Implement Double DQN by enabling the `double_q` parameter in SB3's DQN to mitigate overestimation.
   - **Clipping Rewards:** Normalize or clip rewards to prevent large gradients and stabilize training.

7. **Reward Shaping:**
   - **Consistent Rewards:** Ensure that the reward structure is consistent and provides clear signals for desired behaviors.
   - **Normalization:** Normalize rewards to keep them within a manageable range, aiding in faster convergence.

8. **Exploration Strategy:**
   - **ε-Greedy Policy:** Use an ε-greedy policy to balance exploration and exploitation.
   - **Decay Schedule:** Gradually decay ε to reduce exploration as the agent becomes more confident in its policy.

9. **Network Architecture:**
   - **Depth and Width:** Design the neural network with an appropriate number of layers and neurons to capture the complexity of the environment.
   - **Activation Functions:** Use activation functions like ReLU for hidden layers and avoid saturating activations in output layers.

10. **Hyperparameter Tuning:**
    - **Learning Rate:** Carefully tune the learning rate to ensure stable and efficient learning.
    - **Discount Factor (\( \gamma \)):** Set \( \gamma \) based on the importance of future rewards in the specific task.

11. **Monitoring and Logging:**
    - **Performance Metrics:** Regularly monitor metrics like average reward, loss, and Q-value distributions to track training progress.
    - **Visualization Tools:** Utilize tools like TensorBoard or Matplotlib to visualize training dynamics.

12. **Reproducibility:**
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

13. **Modular Code Structure:**
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

14. **Advanced Techniques:**
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

15. **Regular Evaluation:**
    - **Periodic Evaluation:** Regularly evaluate the agent's performance on a set of evaluation episodes to monitor progress and detect overfitting.
    - **Best Model Saving:** Save the best-performing model during training for later use and deployment.

---

## **Further Reading and Resources**
- **"Prioritized Experience Replay" by Schaul et al. (2015):** The foundational paper introducing PER. [Read Paper](https://arxiv.org/abs/1511.05952)
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

**Great job on completing Lesson 6.3!** You've successfully understood and implemented **Prioritized Experience Replay (PER)** using **Stable Baselines3**, enhancing the efficiency and performance of your Deep Q-Networks. By prioritizing significant transitions, PER ensures that your agent focuses on the most informative experiences, leading to faster convergence and more robust learning. Comparing PER-enhanced DQN with standard DQN highlights the tangible benefits of advanced RL techniques. In the upcoming lessons, we'll explore **Advanced Exploration Strategies** and **Multi-Agent Reinforcement Learning**, further expanding your RL toolkit.
```