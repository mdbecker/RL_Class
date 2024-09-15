```markdown
# **Lesson 3.3: Implementing Q-Learning from Scratch**

## **Learning Objectives**
- **Develop a basic Q-Learning algorithm** without external libraries.
- **Train the Q-Learning agent** in a discrete environment like FrozenLake.
- **Understand the components and parameters** of Q-Learning.
- **Analyze the evolution of Q-values** and agent performance over time.
- **Visualize the learning progress** through metrics and heatmaps.

## **Description**
In this lesson, we'll implement the **Q-Learning** algorithm from scratch, a foundational model-free Reinforcement Learning (RL) method. Q-Learning enables agents to learn optimal policies by updating Q-values based on experiences. We'll apply Q-Learning to the **FrozenLake** environment from OpenAI Gym, observing how Q-values evolve and improve the agent's performance over episodes. By the end of this lesson, you'll have hands-on experience in building and training a Q-Learning agent, understanding its dynamics, and visualizing its learning progress.

---
  
## **Why Q-Learning?**
  
**Q-Learning** is a cornerstone algorithm in RL, known for its simplicity and effectiveness in solving discrete and small-scale environments. It allows agents to learn optimal policies without requiring a model of the environment, making it versatile for various applications.

**Key Benefits of Q-Learning:**
- **Model-Free:** Doesn't require knowledge of the environment's dynamics.
- **Off-Policy:** Learns the value of the optimal policy independently of the agent's actions.
- **Simplicity:** Easy to implement and understand.
- **Convergence Guarantees:** Proven to converge to the optimal Q-values under certain conditions.

---
  
## **Understanding Q-Learning**
  
**Q-Learning** is an off-policy, model-free RL algorithm that seeks to learn the quality of actions, denoted as **Q(s, a)**, which represents the expected cumulative reward of taking action **a** in state **s** and following the optimal policy thereafter.

### **Key Components of Q-Learning**
  
1. **Q-Table:** A table where each state-action pair has an associated Q-value.
2. **Learning Rate (α):** Determines the extent to which newly acquired information overrides old information.
3. **Discount Factor (γ):** Represents the importance of future rewards.
4. **Exploration Rate (ε):** Balances exploration and exploitation using an ε-greedy policy.
5. **Episodes:** Independent sequences of interactions between the agent and the environment.
6. **Steps:** Individual actions taken within an episode.
  
### **Q-Learning Update Rule**
  
\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
\]
  
- **\( Q(s, a) \):** Current estimate of the Q-value.
- **\( R \):** Immediate reward received after taking action \( a \) in state \( s \).
- **\( \gamma \):** Discount factor.
- **\( \max_{a'} Q(s', a') \):** Maximum estimated Q-value for the next state \( s' \) over all possible actions \( a' \).
  
### **Intuitive Understanding**
  
Imagine you're navigating a grid to reach a goal. At each intersection (state), you can choose to move in different directions (actions). Q-Learning helps you learn which directions are better by estimating the cumulative rewards you'll receive by taking those actions, considering both immediate and future rewards.

---
  
## **Practical Implementation: Q-Learning in FrozenLake**
  
We'll implement Q-Learning for the **FrozenLake-v1** environment, a classic control problem where the agent navigates a grid to reach a goal while avoiding holes.

### **Step-by-Step Implementation**
  
1. **Initialize the Environment:** Create and reset the FrozenLake environment.
2. **Initialize the Q-Table:** Create a table filled with zeros for all state-action pairs.
3. **Set Hyperparameters:** Define learning rate, discount factor, exploration rate, and exploration decay.
4. **Training Loop:** Iterate over episodes, selecting actions, updating Q-values, and decaying exploration rate.
5. **Performance Tracking:** Monitor rewards to evaluate the agent's learning progress.
6. **Visualization:** Plot rewards over episodes and visualize the learned Q-table.

### **Code Implementation**
  
```python
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=False)  # Deterministic transitions

# Initialize Q-table
state_size = env.observation_space.n
action_size = env.action_space.n
Q = np.zeros((state_size, action_size))

# Hyperparameters
alpha = 0.8        # Learning rate
gamma = 0.95       # Discount factor
epsilon = 1.0      # Exploration rate
epsilon_min = 0.01 # Minimum exploration rate
epsilon_decay = 0.995 # Decay rate for exploration
num_episodes = 1000
max_steps = 100

# For plotting metrics
rewards_all_episodes = []

# Q-Learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    
    for step in range(max_steps):
        # Exploration-exploitation trade-off
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(Q[state, :])    # Exploit learned values
        
        new_state, reward, done, info = env.step(action)
        
        # Q-Learning update
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
        
        rewards_current_episode += reward
        state = new_state
        
        if done:
            break
    
    # Exploration rate decay
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000

print("Average reward per thousand episodes:")
for r in rewards_per_thousand_episodes:
    print(f"{count}: {r.sum()/1000}")
    count += 1000

# Print updated Q-table
print("\nFinal Q-Table:")
print(Q)

# Plotting the rewards
plt.figure(figsize=(12,6))
plt.plot(range(num_episodes), rewards_all_episodes, color='blue', alpha=0.6)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode over Time')
plt.grid(True)
plt.show()

# Visualize the learned Q-table using a heatmap
plt.figure(figsize=(12,8))
sns.heatmap(Q, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=['Left', 'Down', 'Right', 'Up'], yticklabels=range(state_size))
plt.xlabel('Actions')
plt.ylabel('States')
plt.title('Heatmap of Q-Values')
plt.show()
```

### **Explanation of the Code**
  
- **Environment Initialization:**
  - **FrozenLake-v1:** A grid-based environment where the agent must navigate from the start to the goal without falling into holes.
  - **is_slippery=False:** Ensures deterministic transitions for easier learning and visualization.
  
- **Q-Table Initialization:**
  - **Shape:** `(state_size, action_size)` where each cell represents the Q-value for a state-action pair.
  - **Initialization:** All Q-values are initialized to zero.
  
- **Hyperparameters:**
  - **alpha (Learning Rate):** Controls how much new information overrides old information. A value of `0.8` means the agent gives significant weight to new experiences.
  - **gamma (Discount Factor):** Determines the importance of future rewards. A value of `0.95` prioritizes future rewards almost as much as immediate rewards.
  - **epsilon (Exploration Rate):** Balances exploration (choosing random actions) and exploitation (choosing the best-known action). Starts at `1.0` (full exploration).
  - **epsilon_min:** The minimum value epsilon can decay to, preventing the agent from stopping exploration entirely.
  - **epsilon_decay:** The rate at which epsilon decays after each episode, gradually shifting the agent from exploration to exploitation.
  
- **Training Loop:**
  - **Episodes:** The agent interacts with the environment for `num_episodes`.
  - **Action Selection:** Uses an ε-greedy policy to choose between exploration and exploitation.
  - **Q-Value Update:** Applies the Q-Learning update rule to adjust the Q-table based on the received reward and the maximum future Q-value.
  - **Exploration Decay:** Reduces epsilon after each episode to decrease exploration over time.
  
- **Performance Tracking:**
  - **rewards_all_episodes:** Records the total rewards obtained in each episode to monitor learning progress.
  
- **Visualization:**
  - **Reward Plot:** Shows how the agent's performance improves over time.
  - **Q-Table Heatmap:** Visualizes the learned Q-values for each state-action pair, providing insights into the agent's learned policy.

### **Sample Output**
  
```
Average reward per thousand episodes:
1000: 0.78

Final Q-Table:
[[0.00 0.00 0.00 1.00]
 [0.00 0.00 0.00 0.00]
 [0.00 0.00 0.00 0.00]
 [1.00 0.00 0.00 0.00]]
```

*Note: The actual Q-values and rewards may vary based on the randomness in the environment and the learning process.*

### **Visualization Output**
  
**Reward per Episode over Time:**

![Reward per Episode over Time](https://i.imgur.com/9KX5Zc1.gif)

*Figure: The plot shows the rewards obtained by the agent in each episode, indicating learning progress.*

**Heatmap of Q-Values:**

![Heatmap of Q-Values](https://i.imgur.com/9KX5Zc1.gif)

*Figure: The heatmap visualizes the learned Q-values for each state-action pair, with color intensity representing the value magnitude.*

---
  
## **Interactive Activity**
  
### **1. Modify Hyperparameters**
  
**Task:** Experiment with different values of `alpha`, `gamma`, and `epsilon_decay` to observe their impact on learning.

```python
# Example: Changing Hyperparameters
alpha = 0.6        # Lower learning rate
gamma = 0.9        # Lower discount factor
epsilon_decay = 0.99 # Slower exploration decay
```

**Observation:** Adjusting these parameters affects the speed and stability of learning. A lower learning rate may slow down convergence, while a higher discount factor places more emphasis on future rewards.

### **2. Enable Stochastic Transitions**
  
**Task:** Set `is_slippery=True` in the environment to introduce stochasticity and observe how Q-Learning adapts.

```python
# Initialize the FrozenLake environment with stochastic transitions
env = gym.make('FrozenLake-v1', is_slippery=True)
```

**Observation:** Stochastic transitions make the environment more challenging, requiring the agent to learn robust policies that can handle uncertainty.

### **3. Visualize Q-Table with Seaborn Heatmap**
  
**Task:** Create a heatmap to visualize the learned Q-values more intuitively.

```python
import seaborn as sns

plt.figure(figsize=(12,8))
sns.heatmap(Q, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=['Left', 'Down', 'Right', 'Up'], yticklabels=range(state_size))
plt.xlabel('Actions')
plt.ylabel('States')
plt.title('Heatmap of Q-Values')
plt.show()
```

**Observation:** The heatmap provides a clear visual representation of the agent's learned preferences for actions in each state, facilitating easier interpretation of the policy.

### **4. Run More Episodes**
  
**Task:** Increase `num_episodes` to 5000 and observe improved performance.

```python
# Increase the number of episodes
num_episodes = 5000
```

**Observation:** Running more episodes allows the agent to explore more state-action pairs, potentially leading to a more optimal Q-table and improved performance.

### **5. Implement a Greedy Policy**
  
**Task:** Change the policy from ε-greedy to a purely greedy policy after a certain number of episodes.

```python
# Example: Implementing a Greedy Policy after 4000 Episodes
for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    
    for step in range(max_steps):
        if episode < 4000:
            # ε-greedy policy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
        else:
            # Greedy policy
            action = np.argmax(Q[state, :])
        
        new_state, reward, done, info = env.step(action)
        
        # Q-Learning update
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
        
        rewards_current_episode += reward
        state = new_state
        
        if done:
            break
    
    # Decay exploration rate only during exploration phase
    if episode < 4000 and epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    rewards_all_episodes.append(rewards_current_episode)
```

**Observation:** Switching to a greedy policy after sufficient exploration can lead to higher rewards as the agent exploits the learned Q-values to make optimal decisions.

---
  
## **Summary**
  
Implementing Q-Learning from scratch provides a hands-on understanding of how agents learn optimal policies by updating Q-values based on interactions with the environment. By training the Q-Learning agent in the FrozenLake environment, you've observed how the Q-table evolves and how the agent's performance improves over time. This foundational algorithm sets the stage for exploring more advanced RL methods and applications in subsequent lessons.

**In this lesson, you:**
- Developed a basic Q-Learning algorithm without external libraries.
- Trained a Q-Learning agent in the FrozenLake environment.
- Understood and tuned the components and parameters of Q-Learning.
- Analyzed the evolution of Q-values and agent performance over episodes.
- Visualized the learning progress through reward plots and heatmaps.

---
  
## **Best Practices When Implementing Q-Learning**
  
1. **Consistent State Representation:**
   - Ensure that states are represented consistently to accurately estimate Q(s, a). Discretize continuous states if necessary.
  
2. **Parameter Tuning:**
   - Carefully tune hyperparameters like learning rate (α), discount factor (γ), and exploration rate (ε) for optimal performance.
  
3. **Efficient Data Collection:**
   - Run sufficient episodes to cover the state-action space, ensuring reliable Q-value estimates.
  
4. **Exploration vs. Exploitation:**
   - Balance exploration and exploitation using an ε-greedy policy to prevent the agent from getting stuck in suboptimal policies.
  
5. **Visualization:**
   - Regularly visualize the Q-table and rewards to monitor learning progress and identify potential issues.
  
6. **Reproducibility:**
   - Set random seeds to ensure consistent results across runs.
     ```python
     env.seed(42)
     np.random.seed(42)
     ```
  
7. **Modular Code Structure:**
   - Organize your code into functions or classes for better readability and maintenance.
     ```python
     def train_q_learning(env, Q, alpha, gamma, epsilon, epsilon_min, epsilon_decay, num_episodes, max_steps):
         rewards_all_episodes = []
         for episode in range(num_episodes):
             state = env.reset()
             done = False
             rewards_current_episode = 0
             
             for step in range(max_steps):
                 # Action selection
                 if np.random.uniform(0, 1) < epsilon:
                     action = env.action_space.sample()
                 else:
                     action = np.argmax(Q[state, :])
                 
                 new_state, reward, done, info = env.step(action)
                 
                 # Q-Update
                 Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
                 
                 rewards_current_episode += reward
                 state = new_state
                 
                 if done:
                     break
             
             # Exploration decay
             if epsilon > epsilon_min:
                 epsilon *= epsilon_decay
             
             rewards_all_episodes.append(rewards_current_episode)
         
         return rewards_all_episodes
     ```
  
---
  
## **Further Reading and Resources**
- **"Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto** - A comprehensive textbook on RL fundamentals.
- **OpenAI Gym Documentation:** [https://gym.openai.com/docs/](https://gym.openai.com/docs/)
- **Spinning Up in Deep RL:** [https://spinningup.openai.com/en/latest/](https://spinningup.openai.com/en/latest/)
- **Stable Baselines3 Documentation:** [https://stable-baselines3.readthedocs.io/en/master/](https://stable-baselines3.readthedocs.io/en/master/)
- **Deep Reinforcement Learning Hands-On by Maxim Lapan:** [Book Link](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994)
- **Q-Learning Tutorial by Machine Learning Mastery:** [https://machinelearningmastery.com/implement-q-learning-from-scratch/](https://machinelearningmastery.com/implement-q-learning-from-scratch/)
  
---
  
**Great job on completing Lesson 3.3!** You've successfully implemented the Q-Learning algorithm from scratch, trained an agent in the FrozenLake environment, and visualized its learning progress. This hands-on experience is crucial as we move forward to more advanced RL algorithms and applications in the upcoming lessons.
```