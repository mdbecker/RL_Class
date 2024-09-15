```markdown
# **Lesson 5.2: Implementing DQN with Stable Baselines3**

## **Learning Objectives**
- **Set up and train a DQN agent** using Stable Baselines3.
- **Monitor and evaluate the agent's performance** in environments like CartPole.
- **Understand the configuration and hyperparameters** associated with DQN in Stable Baselines3.
- **Compare the DQN agent's performance** with baseline methods.
- **Leverage advanced features** of Stable Baselines3 for enhanced training and evaluation.
- **Visualize training metrics** to gain insights into agent performance.

## **Description**
In this lesson, we'll implement a **Deep Q-Network (DQN)** agent using the **Stable Baselines3 (SB3)** library. We'll train the agent on the **CartPole-v1** environment, monitor its performance, and evaluate how well it learns to balance the pole. This hands-on approach leverages SB3's robust implementations, enabling efficient training and evaluation of deep RL agents. Additionally, we'll explore advanced components like **Experience Replay** and **Target Networks**, and utilize **TensorBoard** for monitoring training metrics.

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

## **Implementing a DQN Agent with Stable Baselines3**

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

We'll use the **CartPole-v1** environment for this example, a classic control problem where the agent must balance a pole on a moving cart.

```python
# Initialize the CartPole environment
env = gym.make('CartPole-v1')
```

**Environment Overview:**
- **State Space:** Continuous (position, velocity, angle, angular velocity)
- **Action Space:** Discrete (move left or right)
- **Objective:** Prevent the pole from falling over by moving the cart appropriately.

### **Step 3: Create and Train the DQN Agent**

Using SB3, creating and training a DQN agent is straightforward. We'll instantiate the DQN model with an MLP (Multi-Layer Perceptron) policy and train it for a specified number of timesteps.

```python
# Create the DQN agent
model = DQN(
    'MlpPolicy', 
    env, 
    verbose=1, 
    learning_rate=1e-3, 
    buffer_size=50000, 
    learning_starts=1000, 
    batch_size=32, 
    gamma=0.99, 
    target_update_interval=1000
)

# Train the agent
model.learn(total_timesteps=10000)
```

**Hyperparameter Breakdown:**
- **`MlpPolicy`:** Defines the policy network architecture (MLP in this case).
- **`verbose=1`:** Enables detailed logging during training.
- **`learning_rate=1e-3`:** Learning rate for the optimizer.
- **`buffer_size=50000`:** Size of the Experience Replay buffer.
- **`learning_starts=1000`:** Number of timesteps before training starts (to populate the replay buffer).
- **`batch_size=32`:** Size of the mini-batches sampled from the replay buffer.
- **`gamma=0.99`:** Discount factor for future rewards.
- **`target_update_interval=1000`:** Frequency (in steps) to update the Target Network.

### **Step 4: Evaluate the Trained Agent**

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

### **Step 5: Visualize Agent Performance**

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
            state, reward, done, _ = env.step(action)  # Take action
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

### **Step 6: Plot Training Progress**

While Stable Baselines3 does not provide built-in plotting for training progress, you can track and plot rewards by implementing custom callbacks or logging. Additionally, integrating TensorBoard allows for comprehensive monitoring of training metrics.

#### **Example: Using Callbacks for Monitoring and Early Stopping**

```python
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
callback = CallbackList([eval_callback, stop_callback])

# Re-initialize the DQN agent with callbacks
model = DQN(
    'MlpPolicy', 
    env, 
    verbose=1, 
    learning_rate=1e-3, 
    buffer_size=50000, 
    learning_starts=1000, 
    batch_size=32, 
    gamma=0.99, 
    target_update_interval=1000
)

# Train the agent with callbacks
model.learn(total_timesteps=10000, callback=callback)
```

**Explanation:**
- **`EvalCallback`:** Periodically evaluates the agent's performance and saves the best model.
- **`StopTrainingOnRewardThreshold`:** Stops training early if the agent achieves the specified reward threshold.

#### **Visualizing with TensorBoard**

TensorBoard provides a powerful interface to monitor training metrics such as loss, reward, and more.

1. **Modify the Training Code to Include TensorBoard Logging:**

    ```python
    from stable_baselines3.common.callbacks import TensorboardCallback

    # Initialize the PPO agent with TensorBoard logging
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
        tensorboard_log="./dqn_cartpole_tensorboard/"
    )

    # Train the agent with TensorBoard callback
    model.learn(total_timesteps=10000, callback=TensorboardCallback())
    ```

2. **Launch TensorBoard:**

    ```bash
    # In terminal, run:
    tensorboard --logdir=./dqn_cartpole_tensorboard/
    ```

3. **Access TensorBoard:**
    - Open the provided URL in your browser to monitor training metrics in real-time.

**Note:** SB3's TensorBoard integration allows you to visualize various metrics such as reward per episode, loss curves, and more, providing deeper insights into the agent's learning process.

### **Step 7: Save and Load the Trained Model**

Saving and loading models allows you to reuse trained agents without retraining.

```python
# Save the trained model
model.save("dqn_cartpole")

# To load the model later
# model = DQN.load("dqn_cartpole", env=env)
```

### **Step 8: Compare with Custom Policy Gradient Implementation**

Comparing SB3's DQN with the custom Policy Gradient implementation from **Lesson 4.2** highlights the advantages of using optimized, library-based algorithms.

**Key Comparisons:**
- **Performance:** SB3's DQN often converges faster and achieves higher rewards due to optimized implementations.
- **Ease of Use:** SB3 abstracts much of the complexity, allowing quick experimentation with different algorithms and hyperparameters.
- **Flexibility:** While custom implementations offer deeper insights, SB3 provides flexibility through its extensive API for customization.
- **Stability:** SB3's algorithms include advanced techniques like Experience Replay and Target Networks to ensure stable training.

**Performance Metrics Comparison:**

| **Metric**           | **Custom Policy Gradient** | **SB3's DQN** |
|----------------------|----------------------------|---------------|
| **Average Reward**   | Varies based on implementation and hyperparameters | Consistently high with proper tuning |
| **Training Time**    | Longer due to manual implementation | Optimized for faster convergence |
| **Ease of Implementation** | Requires building and debugging from scratch | Plug-and-play with SB3's API |
| **Stability**        | May suffer from high variance and instability | Enhanced stability with SB3's built-in mechanisms |

---

## **Interactive Activity**

### **1. Adjust Hyperparameters and Observe Their Impact**

**Task:** Experiment with different values of the learning rate, buffer size, or batch size and observe their impact on training performance.

```python
# Example: Change learning rate and batch size
model = DQN(
    'MlpPolicy', 
    env, 
    verbose=1, 
    learning_rate=5e-4,  # Decreased learning rate
    buffer_size=100000,  # Increased buffer size
    batch_size=64,       # Increased batch size
    gamma=0.95, 
    target_update_interval=500
)
model.learn(total_timesteps=20000)
```

**Observation:** Adjusting hyperparameters can significantly affect the speed and stability of learning. A lower learning rate may slow down convergence, while increasing the buffer size and batch size can improve learning stability and sample efficiency.

### **2. Train on a Different Environment**

**Task:** Replace `CartPole-v1` with another Gym environment (e.g., `MountainCar-v0`) and implement the DQN agent.

```python
# Initialize the MountainCar environment
env = gym.make('MountainCar-v0')

# Create the DQN agent
model = DQN(
    'MlpPolicy', 
    env, 
    verbose=1, 
    learning_rate=1e-3, 
    buffer_size=50000, 
    learning_starts=1000, 
    batch_size=32, 
    gamma=0.99, 
    target_update_interval=1000
)

# Train the agent
model.learn(total_timesteps=10000)

# Evaluate the agent's performance
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Mean Reward: {mean_reward} +/- {std_reward}")

# Visualize the trained agent
visualize_agent(env, model)
```

**Observation:** The `MountainCar-v0` environment presents different challenges, requiring the agent to build momentum to reach the goal. Observe how the DQN adapts to this environment and compare the learning curve with that of `CartPole-v1`.

### **3. Compare with Random Agent**

**Task:** Compare the performance of the DQN agent with the Random Agent implemented in **Lesson 2.2**.

```python
# Assuming you have a Random Agent implemented as follows:
def random_agent(env, num_episodes=100):
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()  # Select random action
            state, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

# Evaluate the Random Agent
mean_reward_random, std_reward_random = random_agent(env, num_episodes=100)
print(f"Random Agent Mean Reward: {mean_reward_random} +/- {std_reward_random}")

# Compare with DQN Agent
print(f"DQN Agent Mean Reward: {mean_reward} +/- {std_reward}")
```

**Observation:** The DQN agent should significantly outperform the Random Agent, demonstrating the effectiveness of learning-based approaches over random action selection.

### **4. Implement Custom Callbacks**

**Task:** Create custom callbacks to log additional metrics or implement advanced training strategies.

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

# Train the agent with the custom callback
model = DQN(
    'MlpPolicy', 
    env, 
    verbose=1, 
    learning_rate=1e-3, 
    buffer_size=50000, 
    learning_starts=1000, 
    batch_size=32, 
    gamma=0.99, 
    target_update_interval=1000
)
model.learn(total_timesteps=10000, callback=custom_logging_callback)

# Plot the logged rewards
plt.figure(figsize=(12,6))
plt.plot(custom_logging_callback.episode_rewards, label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN: Episode Rewards Over Time')
plt.legend()
plt.grid(True)
plt.show()
```

**Explanation:**
- **CustomLoggingCallback:** Extends SB3's `BaseCallback` to log episode rewards and lengths during training.
- **Visualization:** Provides insights into the agent's performance trends over episodes.

### **5. Experiment with Different Hyperparameters**

**Task:** Modify DQN's hyperparameters such as learning rate, buffer size, or batch size and observe their effects on training performance.

```python
# Example: Change learning rate and batch size
model = DQN(
    'MlpPolicy', 
    env, 
    verbose=1, 
    learning_rate=5e-4,  # Decreased learning rate
    buffer_size=100000,  # Increased buffer size
    batch_size=64,       # Increased batch size
    gamma=0.95, 
    target_update_interval=500  # Increased target update frequency
)
model.learn(total_timesteps=20000)
```

**Observation:** Adjusting hyperparameters can significantly affect the speed and stability of learning. A lower learning rate may slow down convergence, while increasing the buffer size and batch size can enhance learning stability and sample efficiency.

---

## **Summary**

Implementing a **Deep Q-Network (DQN)** agent using **Stable Baselines3** allows you to leverage optimized and reliable RL algorithms with minimal effort. By training the DQN agent on the CartPole environment, you've observed how deep neural networks can effectively approximate Q-values, enabling agents to learn complex behaviors in high-dimensional state spaces. Monitoring and evaluating the agent's performance provides insights into the learning dynamics and effectiveness of the DQN approach. Additionally, experimenting with hyperparameters and leveraging callbacks enhances your ability to fine-tune and optimize RL agents for various tasks.

In the next lesson, we'll delve into advanced DQN components like **Experience Replay** and **Target Networks**, further enhancing agent stability and performance.

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

**Great job on completing Lesson 5.2!** You've successfully implemented a **Deep Q-Network (DQN)** agent using **Stable Baselines3**, trained it in the CartPole environment, and evaluated its performance. This hands-on experience with DQNs and SB3's robust implementations enhances your ability to develop and deploy deep RL agents effectively. In the upcoming lessons, we'll delve deeper into advanced DQN components like **Experience Replay** and **Target Networks**, further refining your RL toolkit.
```