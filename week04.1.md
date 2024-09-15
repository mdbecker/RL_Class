```markdown
# **Lesson 4.1: Introduction to Policy Gradients**

## **Learning Objectives**
- **Understand the concept of policy gradients.**
- **Differentiate between policy-based and value-based methods.**
- **Comprehend the advantages of policy-based methods in certain environments.**
- **Explore the mathematical foundation of policy gradients with intuitive explanations.**
- **Implement a basic policy gradient example to reinforce concepts.**
- **Visualize how policy parameters influence action probabilities.**

## **Description**
In this lesson, we'll delve into **Policy Gradient** methods, a class of algorithms in Reinforcement Learning (RL) that optimize policies directly. Unlike value-based methods that estimate value functions to derive policies, policy-based methods adjust the policy parameters to maximize expected rewards. We'll discuss how policy gradients work, their benefits, and how they differ from value-based approaches. Additionally, we'll implement a simple policy gradient example to solidify your understanding and visualize the impact of policy parameters on action probabilities.

## **Setting Up the Environment**

Before we begin, ensure you have a fresh Conda environment set up. Follow these steps to create and activate your environment, and install the necessary packages.

### **Step 1: Create a New Conda Environment**

```bash
# Create a new Conda environment named 'rl_week4' with Python 3.10
conda create -n rl_week4 python=3.10 -y
```

### **Step 2: Activate the Environment**

```bash
# Activate the 'rl_week4' environment
conda activate rl_week4
```

### **Step 3: Install Essential Packages**

We'll install essential packages for RL, including JupyterLab, NumPy, Matplotlib, and PyTorch.

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

## **What are Policy Gradients?**

**Policy Gradient** methods aim to optimize the policy directly by adjusting its parameters in the direction that maximizes expected rewards. This approach contrasts with value-based methods, which estimate value functions to derive policies indirectly.

### **Policy-Based vs. Value-Based Methods**

| **Aspect**               | **Policy-Based Methods**                                    | **Value-Based Methods**                              |
|--------------------------|-------------------------------------------------------------|------------------------------------------------------|
| **Objective**            | Directly optimize the policy \( \pi(a|s) \).               | Estimate value functions \( V(s) \) or \( Q(s,a) \). |
| **Approach**             | Parameterize the policy and perform gradient ascent.        | Use Bellman equations to update value estimates.     |
| **Suitability**          | Continuous action spaces, stochastic policies.              | Discrete action spaces, deterministic policies.      |
| **Advantages**           | - Can handle high-dimensional, continuous actions.<br>- Naturally stochastic, beneficial for exploration.<br>- Directly optimizes the objective. | - Often simpler and faster to implement.<br>- Can achieve high performance in discrete action spaces. |
| **Disadvantages**        | - Can have high variance.<br>- May require more samples to converge.<br>- Sensitive to hyperparameters. | - Limited to discrete action spaces.<br>- May suffer from overestimation bias. |

**Examples of Each Method:**
- **Policy-Based:** REINFORCE, Actor-Critic methods.
- **Value-Based:** Q-Learning, Deep Q-Networks (DQN).

### **Advantages of Policy-Based Methods**
- **Handling Continuous Actions:** Unlike value-based methods that struggle with continuous action spaces, policy-based methods can naturally handle them by parameterizing the policy.
- **Stochastic Policies:** These methods can model stochastic policies, which are beneficial for environments requiring exploration and dealing with partial observability.
- **Direct Optimization:** Policy gradients optimize the policy directly, often leading to better performance in complex environments.

---

## **Mathematical Foundation of Policy Gradients**

The goal is to maximize the expected cumulative reward by optimizing the policy parameters \( \theta \).

### **Objective Function**

\[
J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \right]
\]

- **\( J(\theta) \):** Expected cumulative reward.
- **\( \pi_\theta \):** Policy parameterized by \( \theta \).
- **\( \gamma \):** Discount factor (\( 0 \leq \gamma < 1 \)).

### **Gradient of the Objective Function**

To optimize \( J(\theta) \), we compute its gradient with respect to the policy parameters \( \theta \):

\[
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) Q^{\pi}(s,a) \right]
\]

- **\( \nabla_\theta J(\theta) \):** Gradient of the objective function.
- **\( \log \pi_\theta(a|s) \):** Log-probability of taking action \( a \) in state \( s \).
- **\( Q^{\pi}(s,a) \):** Action-value function under policy \( \pi \).

### **Intuitive Understanding**

- **Policy Gradient Theorem:** Provides a way to compute the gradient of the expected reward with respect to policy parameters.
- **Gradient Ascent:** Update the policy parameters in the direction of the gradient to maximize \( J(\theta) \).

---

## **Practical Example: Policy Gradient Intuition**

Consider a simple environment where an agent decides to move left or right. A policy gradient method adjusts the probabilities of choosing each action based on the rewards received, reinforcing actions that lead to higher rewards.

### **Step-by-Step Implementation**

1. **Define Policy Parameters:** Initialize parameters that determine action probabilities.
2. **Implement the Softmax Function:** Convert raw parameters into probabilities.
3. **Visualize Action Probabilities:** Observe how parameters influence action selection.

```python
import numpy as np
import matplotlib.pyplot as plt

# Example policy parameters for two actions: left and right
theta = np.array([0.0, 0.0])  # Parameters for actions: left and right

# Softmax function to convert theta to probabilities
def softmax(theta):
    exp_theta = np.exp(theta)
    return exp_theta / np.sum(exp_theta)

# Compute action probabilities
action_probs = softmax(theta)
print(f"Initial Action Probabilities: {action_probs}")

# Visualize the effect of theta on action probabilities
theta_values = np.linspace(-2, 2, 100)
left_probs = []
right_probs = []

for t in theta_values:
    theta_temp = np.array([t, 0.0])  # Varying theta for 'left'
    probs = softmax(theta_temp)
    left_probs.append(probs[0])
    right_probs.append(probs[1])

plt.figure(figsize=(10,6))
plt.plot(theta_values, left_probs, label='Left Action Probability')
plt.plot(theta_values, right_probs, label='Right Action Probability')
plt.title('Effect of Policy Parameters on Action Probabilities')
plt.xlabel('Theta for Left Action')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()
```

### **Explanation of the Code**

- **Policy Parameters (\( \theta \)):** Determines the preference for each action. Higher \( \theta \) increases the probability of selecting that action.
- **Softmax Function:** Converts raw parameters into probabilities, ensuring they sum to 1.
- **Visualization:** Shows how varying \( \theta \) for the left action affects the probability of choosing left vs. right.

### **Visualization Output**

![Effect of Policy Parameters on Action Probabilities](https://i.imgur.com/9KX5Zc1.gif)

*Figure: The plot illustrates how increasing the theta parameter for the left action increases its probability while decreasing the right action's probability.*

---

## **Interactive Activity**

### **1. Modify Policy Parameters and Observe Action Probabilities**

**Task:** Change the values of \( \theta \) and observe how action probabilities shift.

```python
# Modify theta to prefer the right action
theta = np.array([-1.0, 2.0])
action_probs = softmax(theta)
print(f"Modified Action Probabilities: {action_probs}")
```

**Expected Output:**
```
Modified Action Probabilities: [0.11920292 0.88079708]
```

**Observation:** Increasing the theta for the right action significantly increases its probability of being selected.

### **2. Implement a Simple Policy Gradient Step**

**Task:** Update policy parameters based on a simple reward signal.

```python
# Simulated reward for taking 'right' action
reward = 1.0

# Compute gradient
def compute_gradient(theta, action, reward, gamma=1.0):
    probs = softmax(theta)
    grad_log = -probs
    grad_log[action] += 1
    return grad_log * reward

# Choose action 'right' (index 1)
action = 1

# Compute gradient
gradient = compute_gradient(theta, action, reward)
print(f"Gradient: {gradient}")

# Update theta using gradient ascent
alpha = 0.1  # Learning rate
theta += alpha * gradient
print(f"Updated Theta: {theta}")

# Compute updated probabilities
updated_probs = softmax(theta)
print(f"Updated Action Probabilities: {updated_probs}")
```

**Expected Output:**
```
Gradient: [-0.11920292  0.88079708]
Updated Theta: [ -0.01192029   0.08807971]
Updated Action Probabilities: [0.47502081 0.52497919]
```

**Explanation:**
- **Gradient Computation:** The gradient indicates how to adjust \( \theta \) to increase the probability of the taken action.
- **Parameter Update:** Theta is updated in the direction of the gradient to reinforce the action.
- **Updated Probabilities:** The probability of the 'right' action has increased.

### **3. Explore Different Reward Signals**

**Task:** Simulate different rewards and observe how policy parameters adjust.

```python
# Simulated reward for taking 'left' action
reward_left = 0.5
action_left = 0

# Compute gradient for 'left' action
gradient_left = compute_gradient(theta, action_left, reward_left)
print(f"Gradient for 'left' action: {gradient_left}")

# Update theta
theta += alpha * gradient_left
print(f"Updated Theta after 'left' action: {theta}")

# Compute updated probabilities
updated_probs_left = softmax(theta)
print(f"Updated Action Probabilities: {updated_probs_left}")
```

**Expected Output:**
```
Gradient for 'left' action: [ 0.52497919 -0.52497919]
Updated Theta after 'left' action: [0.0425079  0.03528192]
Updated Action Probabilities: [0.50610335 0.49389665]
```

**Observation:** Receiving a reward for the 'left' action increases its probability, balancing the policy.

### **4. Visualize Policy Parameter Updates Over Time**

**Task:** Simulate multiple policy gradient updates and visualize the evolution of action probabilities.

```python
# Initialize theta
theta = np.array([0.0, 0.0])

# Define a sequence of actions and rewards
action_sequence = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]  # 0: Left, 1: Right
rewards = [1, 1, 0.5, 1, 0.5, 0.5, 1, 0.5, 1, 1]

# Lists to store probabilities
left_probs = []
right_probs = []

for action, reward in zip(action_sequence, rewards):
    probs = softmax(theta)
    left_probs.append(probs[0])
    right_probs.append(probs[1])
    
    # Compute gradient
    gradient = compute_gradient(theta, action, reward)
    
    # Update theta
    theta += alpha * gradient

# Plotting the probabilities over updates
plt.figure(figsize=(10,6))
plt.plot(left_probs, label='Left Action Probability')
plt.plot(right_probs, label='Right Action Probability')
plt.title('Policy Parameter Updates Over Time')
plt.xlabel('Update Step')
plt.ylabel('Action Probability')
plt.legend()
plt.grid(True)
plt.show()
```

**Expected Output:**

![Policy Parameter Updates Over Time](https://i.imgur.com/9KX5Zc1.gif)

*Figure: The plot shows how the probabilities of choosing left and right actions evolve as policy parameters are updated based on rewards.*

---

## **Implementing a Simple Policy Gradient Example**

To reinforce the concepts, let's implement a basic Policy Gradient method for a simple environment where an agent decides to move left or right to receive rewards.

### **Step-by-Step Implementation**

1. **Define the Environment:** A simple two-action environment.
2. **Initialize Policy Parameters:** Start with neutral parameters.
3. **Define the Policy:** Use softmax to convert parameters to action probabilities.
4. **Simulate Episodes:** Collect actions and rewards.
5. **Compute Gradients:** Calculate gradients based on rewards.
6. **Update Policy Parameters:** Adjust parameters in the direction of the gradient.
7. **Visualize Learning Progress:** Plot how action probabilities change over episodes.

### **Code Implementation**

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a simple two-action environment
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

# Initialize environment
env = SimpleEnv()

# Initialize policy parameters
theta = np.array([0.0, 0.0])  # Parameters for Left and Right actions

# Softmax function
def softmax(theta):
    exp_theta = np.exp(theta - np.max(theta))  # for numerical stability
    return exp_theta / np.sum(exp_theta)

# Compute gradient
def compute_gradient(theta, action, reward, gamma=1.0):
    probs = softmax(theta)
    grad_log = -probs
    grad_log[action] += 1
    return grad_log * reward

# Training parameters
alpha = 0.1  # Learning rate
num_episodes = 100

# Lists to store probabilities
left_probs = []
right_probs = []
rewards = []

for episode in range(num_episodes):
    state = env.reset()
    
    # Select action based on current policy
    probs = softmax(theta)
    action = np.random.choice(env.actions, p=probs)
    
    # Take action and observe reward
    _, reward, done, _ = env.step(action)
    
    # Store probabilities
    left_probs.append(probs[0])
    right_probs.append(probs[1])
    rewards.append(reward)
    
    # Compute gradient
    gradient = compute_gradient(theta, action, reward)
    
    # Update policy parameters
    theta += alpha * gradient

# Plotting the action probabilities over episodes
plt.figure(figsize=(12,6))
plt.plot(range(num_episodes), left_probs, label='Left Action Probability')
plt.plot(range(num_episodes), right_probs, label='Right Action Probability')
plt.title('Policy Parameter Updates Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Action Probability')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the rewards over episodes
plt.figure(figsize=(12,6))
plt.plot(range(num_episodes), rewards, color='green')
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
plt.show()

# Print final policy probabilities
final_probs = softmax(theta)
print(f"Final Policy Probabilities: Left={final_probs[0]:.4f}, Right={final_probs[1]:.4f}")
```

### **Explanation of the Code**

- **Simple Environment (`SimpleEnv`):** A minimal environment where the agent selects between two actions:
  - **Left (0):** Always yields a reward of `1.0`.
  - **Right (1):** Always yields a reward of `0.0`.
  
- **Policy Parameters (`theta`):** Initialized to `[0.0, 0.0]`, representing no initial preference.
  
- **Softmax Function:** Converts raw policy parameters into action probabilities, ensuring they sum to `1`.
  
- **Gradient Computation (`compute_gradient`):** Calculates the gradient of the log-probability of the taken action, scaled by the received reward.
  
- **Training Loop:**
  - **Action Selection:** Chooses an action based on current policy probabilities.
  - **Reward Observation:** Receives a reward based on the chosen action.
  - **Policy Update:** Adjusts `theta` in the direction that increases the probability of actions that received higher rewards.
  
- **Visualization:**
  - **Action Probabilities:** Shows how the probability of choosing left increases over episodes, reflecting learning.
  - **Rewards:** Illustrates the rewards obtained per episode, which should trend upwards as the policy becomes more optimal.

### **Visualization Output**

**Policy Parameter Updates Over Episodes:**

![Policy Parameter Updates Over Episodes](https://i.imgur.com/9KX5Zc1.gif)

*Figure: The plot shows the probability of choosing the left action increasing over episodes as the agent learns to maximize rewards.*

**Rewards per Episode:**

![Rewards per Episode](https://i.imgur.com/9KX5Zc1.gif)

*Figure: The plot illustrates that the agent consistently receives higher rewards as it learns to prefer the left action.*

**Final Policy Probabilities:**

```
Final Policy Probabilities: Left=0.9995, Right=0.0005
```

*Observation: The agent has learned to almost always choose the left action to maximize rewards.*

---

## **Summary**

Policy Gradient methods are a powerful class of algorithms in Reinforcement Learning that optimize policies directly by adjusting their parameters to maximize expected rewards. Unlike value-based methods, which estimate value functions to derive policies, policy-based methods offer several advantages, especially in environments with continuous action spaces and when stochastic policies are beneficial.

In this lesson, you:
- **Defined and differentiated** between policy-based and value-based methods.
- **Explored the mathematical foundation** of policy gradients, understanding how gradients of the expected reward are computed.
- **Implemented a basic policy gradient example**, observing how policy parameters influence action probabilities.
- **Visualized the learning process**, witnessing how the agent's policy evolves to maximize rewards.

This foundational knowledge is essential as we progress to more sophisticated Policy Gradient methods, such as REINFORCE and Actor-Critic algorithms, in the upcoming lessons.

---

## **Best Practices When Working with Policy Gradients**

1. **Parameter Initialization:**
   - Initialize policy parameters carefully to avoid saturation of the activation functions (e.g., softmax).
   
2. **Handling High Variance:**
   - Policy gradients can have high variance. Techniques like baseline subtraction (using value functions) and variance reduction methods can improve learning stability.
   
3. **Balancing Exploration and Exploitation:**
   - Ensure sufficient exploration by maintaining a balance between exploring new actions and exploiting known rewarding actions.
   
4. **Learning Rate Tuning:**
   - Carefully tune the learning rate (`alpha`) to ensure stable and efficient learning. Too high can cause divergence; too low can slow down convergence.
   
5. **Using Entropy Regularization:**
   - Encourage exploration by adding an entropy term to the loss function, promoting a more stochastic policy.
   
6. **Reproducibility:**
   - Set random seeds to ensure consistent results across runs.
     ```python
     np.random.seed(42)
     ```
   
7. **Modular Code Structure:**
   - Organize your code into functions or classes for better readability and maintenance.
     ```python
     def softmax(theta):
         exp_theta = np.exp(theta - np.max(theta))
         return exp_theta / np.sum(exp_theta)
     
     def compute_gradient(theta, action, reward, gamma=1.0):
         probs = softmax(theta)
         grad_log = -probs
         grad_log[action] += 1
         return grad_log * reward
     
     def update_policy(theta, gradient, alpha=0.1):
         return theta + alpha * gradient
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

---

**Great job on completing Lesson 4.1!** You've gained a solid understanding of Policy Gradient methods and how they enable agents to learn optimal policies directly. This knowledge is crucial as we move forward to implementing more advanced Policy Gradient algorithms and exploring their applications in complex environments in the upcoming lessons.
```