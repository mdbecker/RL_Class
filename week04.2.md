```markdown
# **Lesson 4.2: Implementing a Policy Gradient Method**

## **Learning Objectives**
- **Code a basic Policy Gradient algorithm** using PyTorch.
- **Train the policy** on a simple environment like CartPole.
- **Understand the components and workflow** of Policy Gradient methods.
- **Visualize the training progress** and evaluate agent performance.
- **Analyze the impact of hyperparameters** on learning efficiency and stability.
- **Implement variance reduction techniques** to stabilize training.

## **Description**
In this lesson, we'll implement a **Policy Gradient** algorithm from scratch using PyTorch, a popular deep learning framework. We'll train a policy-based agent on the **CartPole-v1** environment from OpenAI Gym, observing how the policy parameters are updated to maximize cumulative rewards. This hands-on approach will solidify your understanding of policy gradients, their practical implementation, and their advantages over value-based methods.

## **Setting Up the Environment**

Ensure you are in the `rl_week4` Conda environment with the necessary packages installed. If not, refer to **Lesson 4.1** for setup instructions.

### **Step 1: Activate the Environment**

```bash
# Activate the 'rl_week4' environment
conda activate rl_week4
```

### **Step 2: Launch JupyterLab**

```bash
# Launch JupyterLab
jupyter lab
```

*JupyterLab will open in your default web browser, providing an interactive environment for coding and visualization.*

---

## **Implementing a Basic Policy Gradient Agent**

### **Step 1: Import Necessary Libraries**

We'll start by importing the essential libraries required for our implementation.

```python
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
```

### **Step 2: Define the Policy Network**

We'll create a simple neural network with one hidden layer to represent our policy. This network will take the state as input and output the probabilities of taking each possible action.

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Initializes the Policy Network.
        
        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Number of possible actions.
            hidden_size (int): Number of neurons in the hidden layer.
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()                          # Activation function
        self.fc2 = nn.Linear(hidden_size, action_size) # Second fully connected layer
        self.softmax = nn.Softmax(dim=1)               # Softmax to output probabilities
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state.
        
        Returns:
            torch.Tensor: Action probabilities.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        action_probs = self.softmax(x)
        return action_probs
```

### **Step 3: Initialize Environment and Network**

We'll initialize the CartPole environment, define the policy network, and set up the optimizer.

```python
# Initialize the CartPole environment
env = gym.make('CartPole-v1')

# Get size of state and action from environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize the policy network
policy = PolicyNetwork(state_size, action_size)

# Define the optimizer (Adam optimizer)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

# Define the discount factor
gamma = 0.99
```

### **Step 4: Define the Policy Gradient Algorithm**

We'll implement the core of the Policy Gradient method, including action selection, reward computation, and policy updates.

```python
def select_action(state):
    """
    Selects an action based on the current policy.
    
    Args:
        state (numpy.ndarray): Current state.
    
    Returns:
        action (int): Selected action.
        log_prob (torch.Tensor): Log probability of the selected action.
    """
    state = torch.from_numpy(state).float().unsqueeze(0)  # Convert state to tensor
    probs = policy(state)                                 # Get action probabilities from policy
    m = Categorical(probs)                                # Create a categorical distribution
    action = m.sample()                                    # Sample an action
    return action.item(), m.log_prob(action)              # Return action and log probability

def compute_returns(rewards, gamma):
    """
    Computes discounted returns for each timestep.
    
    Args:
        rewards (list): List of rewards collected in an episode.
        gamma (float): Discount factor.
    
    Returns:
        torch.Tensor: Discounted returns.
    """
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # Normalize returns for better performance
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns

# Training parameters
num_episodes = 1000
print_every = 100
reward_history = []

for episode in range(1, num_episodes + 1):
    state = env.reset()
    log_probs = []
    rewards = []
    done = False
    total_reward = 0
    
    while not done:
        action, log_prob = select_action(state)
        state, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward
    
    reward_history.append(total_reward)
    
    returns = compute_returns(rewards, gamma)
    
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    policy_loss = torch.cat(policy_loss).sum()
    
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    # Print average reward every 'print_every' episodes
    if episode % print_every == 0:
        avg_reward = np.mean(reward_history[-print_every:])
        print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")
```

### **Step 5: Visualize Training Progress**

We'll plot the total rewards obtained in each episode to observe the agent's learning progress over time.

```python
# Plot the reward history
plt.figure(figsize=(12,6))
plt.plot(reward_history, color='blue', alpha=0.6)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Policy Gradient: Reward per Episode')
plt.grid(True)
plt.show()
```

**Expected Output:**

![Reward per Episode](https://i.imgur.com/9KX5Zc1.gif)

*Figure: The plot shows the rewards obtained by the agent in each episode, indicating learning progress.*

### **Step 6: Test the Trained Policy**

After training, we'll evaluate the performance of the trained policy by running a few test episodes without exploration.

```python
# Function to test the trained policy
def test_policy(env, policy, num_episodes=5):
    """
    Tests the trained policy in the environment.
    
    Args:
        env (gym.Env): The environment to test in.
        policy (nn.Module): The trained policy network.
        num_episodes (int): Number of test episodes.
    """
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()  # Render the environment
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                probs = policy(state)
            action = torch.argmax(probs, dim=1).item()  # Choose the best action
            state, reward, done, _ = env.step(action)
            total_reward += reward
        print(f"Test Episode {episode}: Total Reward: {total_reward}")
    env.close()

# Test the trained policy
test_policy(env, policy)
```

**Expected Output:**

```
Test Episode 1: Total Reward: 200
Test Episode 2: Total Reward: 200
Test Episode 3: Total Reward: 200
Test Episode 4: Total Reward: 200
Test Episode 5: Total Reward: 200
```

*Note: The exact rewards may vary, but with sufficient training, the agent should consistently achieve high rewards, often reaching the maximum reward per episode (e.g., 200 for CartPole).*

---

## **Interactive Activity**

### **1. Modify Hyperparameters and Observe Their Impact**

**Task:** Experiment with different values of the learning rate (`lr`), hidden layer size, or discount factor (`gamma`) and observe how they affect training.

```python
# Example: Change learning rate and hidden layer size
policy = PolicyNetwork(state_size, action_size, hidden_size=256)  # Increase hidden layer size
optimizer = optim.Adam(policy.parameters(), lr=1e-3)             # Decrease learning rate

# Reinitialize reward history
reward_history = []

# Re-run the training loop with new hyperparameters
for episode in range(1, num_episodes + 1):
    state = env.reset()
    log_probs = []
    rewards = []
    done = False
    total_reward = 0
    
    while not done:
        action, log_prob = select_action(state)
        state, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward
    
    reward_history.append(total_reward)
    
    returns = compute_returns(rewards, gamma)
    
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    policy_loss = torch.cat(policy_loss).sum()
    
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    if episode % print_every == 0:
        avg_reward = np.mean(reward_history[-print_every:])
        print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")
```

**Observation:** Adjusting hyperparameters can significantly affect the speed and stability of learning. A lower learning rate may slow convergence, while a larger hidden layer can capture more complex patterns but may require more training episodes.

### **2. Implement Baseline for Variance Reduction**

**Task:** Introduce a baseline (e.g., a value function) to reduce the variance of gradient estimates, leading to more stable and efficient training.

```python
# Define a simple baseline network
class BaselineNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(BaselineNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Outputs a single value
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        value = self.fc2(x)
        return value

# Initialize the baseline network and optimizer
baseline = BaselineNetwork(state_size)
baseline_optimizer = optim.Adam(baseline.parameters(), lr=1e-2)

def compute_advantages(rewards, returns, baseline_values):
    """
    Computes advantages by subtracting baseline values from returns.
    
    Args:
        rewards (list): List of rewards.
        returns (torch.Tensor): Discounted returns.
        baseline_values (torch.Tensor): Estimated state values.
    
    Returns:
        torch.Tensor: Advantages.
    """
    advantages = returns - baseline_values.squeeze()
    return advantages

# Modify the training loop to include baseline
reward_history = []

for episode in range(1, num_episodes + 1):
    state = env.reset()
    log_probs = []
    rewards = []
    done = False
    total_reward = 0
    
    while not done:
        action, log_prob = select_action(state)
        state, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward
    
    reward_history.append(total_reward)
    
    returns = compute_returns(rewards, gamma)
    
    # Compute baseline values
    states = torch.tensor([state for state in env.env.state if True], dtype=torch.float32)
    # Note: For simplicity, we're not passing actual states here. In a full implementation, states should be collected.
    # Here, we'll use the returns as a mock baseline.
    baseline_values = returns.detach()
    
    advantages = compute_advantages(rewards, returns, baseline_values)
    
    policy_loss = []
    for log_prob, advantage in zip(log_probs, advantages):
        policy_loss.append(-log_prob * advantage)
    policy_loss = torch.cat(policy_loss).sum()
    
    # Update policy network
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    # Update baseline network
    baseline_loss = nn.MSELoss()(baseline_values, returns)
    baseline_optimizer.zero_grad()
    baseline_loss.backward()
    baseline_optimizer.step()
    
    # Decay exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    # Print average reward every 'print_every' episodes
    if episode % print_every == 0:
        avg_reward = np.mean(reward_history[-print_every:])
        print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")
```

**Note:** Implementing a proper baseline requires collecting the actual states during episodes and passing them through the baseline network to estimate their values. This example uses returns as a mock baseline for simplicity. For a complete implementation, integrate state collection and baseline value estimation accurately.

### **3. Experiment with Different Environments**

**Task:** Apply the Policy Gradient method to another Gym environment like `MountainCar-v0` and analyze performance.

```python
# Initialize the MountainCar environment
env = gym.make('MountainCar-v0')

# Update state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Reinitialize the policy network for the new environment
policy = PolicyNetwork(state_size, action_size)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

# Reset reward history
reward_history = []

for episode in range(1, num_episodes + 1):
    state = env.reset()
    log_probs = []
    rewards = []
    done = False
    total_reward = 0
    
    while not done:
        action, log_prob = select_action(state)
        state, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward
    
    reward_history.append(total_reward)
    
    returns = compute_returns(rewards, gamma)
    
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    policy_loss = torch.cat(policy_loss).sum()
    
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    # Decay exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    # Print average reward every 'print_every' episodes
    if episode % print_every == 0:
        avg_reward = np.mean(reward_history[-print_every:])
        print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")
```

**Observation:** The `MountainCar-v0` environment presents different challenges compared to `CartPole-v1`, such as the need to build momentum to reach the goal. Observe how the Policy Gradient agent adapts to this environment and how its performance evolves over time.

### **4. Visualize Policy Parameter Updates Over Time**

**Task:** Simulate multiple policy gradient updates and visualize the evolution of action probabilities.

```python
# Initialize theta for a simple policy (for illustration)
theta = np.array([0.0, 0.0])  # Parameters for Left and Right actions

# Define a simple environment where 'Left' yields reward 1 and 'Right' yields reward 0
class SimpleEnv:
        def __init__(self):
            self.actions = [0, 1]  # 0: Left, 1: Right
            self.state = None  # Not used in this simple example

        def reset(self):
            self.state = 0
            return self.state

        def step(self, action):
            if action == 0:  # Left
                reward = 1.0
            else:             # Right
                reward = 0.0
            done = True  # Single-step environment
            return self.state, reward, done, {}

# Initialize the simple environment
simple_env = SimpleEnv()

# Define the policy network for the simple environment
class SimplePolicyNetwork(nn.Module):
    def __init__(self, action_size):
        super(SimplePolicyNetwork, self).__init__()
        self.fc = nn.Linear(1, action_size)  # Dummy input
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc(x)
        action_probs = self.softmax(x)
        return action_probs

# Initialize the simple policy network and optimizer
simple_policy = SimplePolicyNetwork(action_size=2)
simple_optimizer = optim.Adam(simple_policy.parameters(), lr=1e-2)

# Training parameters
simple_num_episodes = 50
simple_reward_history = []
left_probs = []
right_probs = []

for episode in range(1, simple_num_episodes + 1):
    state = simple_env.reset()
    log_probs = []
    rewards = []
    done = False
    total_reward = 0
    
    while not done:
        state_tensor = torch.tensor([[state]], dtype=torch.float32)  # Dummy input
        probs = simple_policy(state_tensor)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        
        new_state, reward, done, _ = simple_env.step(action.item())
        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward
        
    simple_reward_history.append(total_reward)
    
    returns = compute_returns(rewards, gamma=1.0)
    
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    policy_loss = torch.cat(policy_loss).sum()
    
    simple_optimizer.zero_grad()
    policy_loss.backward()
    simple_optimizer.step()
    
    # Store action probabilities
    with torch.no_grad():
        probs = simple_policy(torch.tensor([[state]], dtype=torch.float32))
        left_probs.append(probs[0][0].item())
        right_probs.append(probs[0][1].item())
    
    if episode % 10 == 0:
        avg_reward = np.mean(simple_reward_history[-10:])
        print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")

# Plotting the action probabilities over episodes
plt.figure(figsize=(10,6))
plt.plot(left_probs, label='Left Action Probability')
plt.plot(right_probs, label='Right Action Probability')
plt.title('Policy Parameter Updates Over Episodes (Simple Environment)')
plt.xlabel('Episode')
plt.ylabel('Action Probability')
plt.legend()
plt.grid(True)
plt.show()
```

**Expected Output:**

![Policy Parameter Updates Over Episodes](https://i.imgur.com/9KX5Zc1.gif)

*Figure: The plot shows the probability of choosing the left action increasing over episodes as the agent learns to maximize rewards.*

---

## **Summary**

Policy Gradient methods are a powerful class of algorithms in Reinforcement Learning that optimize policies directly by adjusting their parameters to maximize expected rewards. Unlike value-based methods, which estimate value functions to derive policies, policy-based methods offer several advantages, especially in environments with continuous action spaces and when stochastic policies are beneficial.

In this lesson, you:
- **Defined and differentiated** between policy-based and value-based methods.
- **Explored the mathematical foundation** of policy gradients, understanding how gradients of the expected reward are computed.
- **Implemented a basic policy gradient example**, observing how policy parameters influence action probabilities.
- **Visualized the learning process**, witnessing how the agent's policy evolves to maximize rewards.
- **Experimented with different environments and techniques**, enhancing the robustness and efficiency of the Policy Gradient method.

This foundational knowledge is essential as we progress to more sophisticated Policy Gradient methods, such as REINFORCE and Actor-Critic algorithms, in the upcoming lessons.

---

## **Best Practices When Working with Policy Gradients**

1. **Parameter Initialization:**
   - Initialize policy parameters carefully to avoid saturation of activation functions (e.g., softmax). Small random values can help in maintaining a balanced initial policy.

2. **Handling High Variance:**
   - Policy gradients can have high variance, leading to unstable training. Techniques like **baseline subtraction** (using value functions) and **variance reduction methods** can improve learning stability.

3. **Balancing Exploration and Exploitation:**
   - Ensure sufficient exploration by maintaining a balance between exploring new actions and exploiting known rewarding actions. Strategies like **entropy regularization** can encourage exploration.

4. **Learning Rate Tuning:**
   - Carefully tune the learning rate (`alpha`) to ensure stable and efficient learning. A learning rate that's too high can cause divergence, while one that's too low can slow down convergence.

5. **Using Entropy Regularization:**
   - Encourage exploration by adding an entropy term to the loss function, promoting a more stochastic policy. This prevents the policy from prematurely converging to suboptimal deterministic policies.

   ```python
   # Example: Adding entropy regularization
   entropy = -torch.sum(probs * torch.log(probs + 1e-9))
   policy_loss = policy_loss - 0.01 * entropy  # 0.01 is the entropy coefficient
   ```

6. **Reproducibility:**
   - Set random seeds to ensure consistent results across runs.

   ```python
   import torch
   import numpy as np
   import random
   
   torch.manual_seed(42)
   np.random.seed(42)
   random.seed(42)
   ```

7. **Modular Code Structure:**
   - Organize your code into functions or classes for better readability and maintenance.

   ```python
   def train_policy_gradient(env, policy, optimizer, num_episodes, gamma):
       reward_history = []
       for episode in range(1, num_episodes + 1):
           state = env.reset()
           log_probs = []
           rewards = []
           done = False
           total_reward = 0
           
           while not done:
               action, log_prob = select_action(state)
               state, reward, done, _ = env.step(action)
               log_probs.append(log_prob)
               rewards.append(reward)
               total_reward += reward
           
           reward_history.append(total_reward)
           
           returns = compute_returns(rewards, gamma)
           
           policy_loss = []
           for log_prob, R in zip(log_probs, returns):
               policy_loss.append(-log_prob * R)
           policy_loss = torch.cat(policy_loss).sum()
           
           optimizer.zero_grad()
           policy_loss.backward()
           optimizer.step()
           
           if episode % 100 == 0:
               avg_reward = np.mean(reward_history[-100:])
               print(f"Episode {episode}\tAverage Reward: {avg_reward:.2f}")
       
       return reward_history
   ```

---

## **Further Reading and Resources**
- **"Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto** - A comprehensive textbook on RL fundamentals.
- **OpenAI Gym Documentation:** [https://gym.openai.com/docs/](https://gym.openai.com/docs/)
- **Spinning Up in Deep RL:** [https://spinningup.openai.com/en/latest/](https://spinningup.openai.com/en/latest/)
- **Stable Baselines3 Documentation:** [https://stable-baselines3.readthedocs.io/en/master/](https://stable-baselines3.readthedocs.io/en/master/)
- **Deep Reinforcement Learning Hands-On by Maxim Lapan:** [Book Link](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994)
- **Policy Gradient Methods Tutorial by Lilian Weng:** [https://lilianweng.github.io/posts/2018-04-08-policy-gradient/](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- **REINFORCE Algorithm Explanation:** [https://towardsdatascience.com/reinforce-policy-gradient-method-8fc1221c745e](https://towardsdatascience.com/reinforce-policy-gradient-method-8fc1221c745e)
- **PyTorch Documentation:** [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- **Visualizing Reinforcement Learning Agents:** [https://www.youtube.com/watch?v=O_0NnXjKojg](https://www.youtube.com/watch?v=O_0NnXjKojg)

---

**Great job on completing Lesson 4.2!** You've successfully implemented a Policy Gradient algorithm using PyTorch, trained an agent in the CartPole environment, and visualized its learning progress. This hands-on experience is crucial as we move forward to more advanced Policy Gradient methods and explore their applications in complex environments in the upcoming lessons.
```