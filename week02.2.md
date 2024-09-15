```markdown
# **Lesson 2.2: Implementing a Random Agent**

## **Learning Objectives**
- **Create a simple agent** that selects actions randomly.
- **Visualize the agent's performance** in an environment.
- **Establish a baseline** for evaluating more advanced agents.
- **Understand the concept of episodes and steps** in Gym environments.
- **Analyze the performance metrics** of a Random Agent.

## **Description**
In this lesson, we'll build and run a **Random Agent** that selects actions uniformly at random within an environment. This agent serves as a fundamental baseline, allowing us to compare the performance of more sophisticated RL algorithms in subsequent lessons. We'll visualize the agent's performance, track rewards, and gain insights into the dynamics of different Gym environments.

---

## **Why Implement a Random Agent?**

Before diving into complex algorithms, it's crucial to establish a performance baseline. A Random Agent, despite its simplicity, provides valuable insights into:
- **Baseline Performance:** Understanding the minimum performance expectations.
- **Environment Dynamics:** Observing how environments respond to random actions.
- **Algorithm Evaluation:** Measuring improvements achieved by advanced agents against the random baseline.

---

## **Setting Up the Environment**

Ensure you are in the `rl_course_week2` Conda environment with the necessary packages installed. If not, refer to **Lesson 2.1** for setup instructions.

### **Step 1: Activate the Environment**

```bash
# Activate the 'rl_course_week2' environment
conda activate rl_course_week2
```

### **Step 2: Launch JupyterLab**

```bash
# Launch JupyterLab
jupyter lab
```

*JupyterLab will open in your default web browser, providing an interactive environment for coding and visualization.*

---

## **Implementing a Random Agent**

### **What is a Random Agent?**

A **Random Agent** selects actions randomly from the available action space without considering the current state. While it doesn't learn or optimize behavior, it provides a performance baseline for evaluating more advanced RL algorithms.

**Key Characteristics:**
- **No Learning:** Actions are selected without any strategy.
- **Baseline Reference:** Serves as a point of comparison for more intelligent agents.
- **Simple Implementation:** Easy to implement and understand.

### **Practical Example: Random Agent in CartPole**

Let's implement a Random Agent in the **CartPole-v1** environment and visualize its performance.

#### **Step-by-Step Implementation**

1. **Environment Initialization:** Create and reset the environment.
2. **Episode Loop:** Run multiple episodes to observe agent behavior across different trials.
3. **Step Loop:** Within each episode, the agent takes random actions up to a maximum number of steps or until the episode ends (`done`).
4. **Reward Tracking:** Accumulate rewards per episode to evaluate performance.
5. **Rendering:** Visualize the agent's actions and the environment's response.
6. **Plotting:** Display the total rewards per episode for analysis.

```python
import gym
import matplotlib.pyplot as plt

# Initialize the CartPole environment
env = gym.make('CartPole-v1')

# Parameters
num_episodes = 10        # Number of episodes to run
max_steps = 200           # Maximum steps per episode

# Store rewards for each episode
rewards_per_episode = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    for step in range(max_steps):
        env.render()  # Render the environment
        
        action = env.action_space.sample()  # Random action
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode {episode + 1} finished after {step + 1} steps with total reward {total_reward}")
            break
    
    rewards_per_episode.append(total_reward)

env.close()

# Plotting rewards
plt.figure(figsize=(10,5))
plt.plot(range(1, num_episodes + 1), rewards_per_episode, marker='o', linestyle='-', color='b')
plt.title('Random Agent Performance in CartPole-v1')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.xticks(range(1, num_episodes + 1))
plt.grid(True)
plt.show()
```

#### **Explanation of the Code**

- **Environment Creation:** `gym.make('CartPole-v1')` initializes the CartPole environment.
- **Reset:** `env.reset()` resets the environment and returns the initial state.
- **Render:** `env.render()` visualizes the current state of the environment.
- **Action Sampling:** `env.action_space.sample()` selects a random action from the action space.
- **Step:** `env.step(action)` applies the action, returning:
  - **next_state:** The state after the action.
  - **reward:** Immediate reward received after the action.
  - **done:** Boolean indicating if the episode has ended.
  - **info:** Additional diagnostic information (can be ignored for basic tasks).
- **Close:** `env.close()` properly closes the environment to free up resources.
- **Plotting:** Visualizes the total rewards per episode to analyze performance trends.

### **Visualization**

![Random Agent Performance](https://i.imgur.com/9KX5Zc1.gif)

*Figure: CartPole Environment - The pole remains balanced as the cart moves left or right with random actions.*

---

## **Interactive Activity**

### **1. Change the Number of Episodes**

**Task:** Modify `num_episodes` to run more episodes and observe variations in performance.

```python
# Example: Increase number of episodes to 20
num_episodes = 20
```

**Expected Outcome:** More data points in the reward plot, allowing for a better understanding of performance variability.

### **2. Adjust Maximum Steps**

**Task:** Change `max_steps` to limit or extend the duration of each episode.

```python
# Example: Increase maximum steps to 300
max_steps = 300
```

**Observation:** Allows the agent more time to balance the pole, potentially increasing total rewards per episode.

### **3. Compare Different Environments**

**Task:** Replace `'CartPole-v1'` with another environment (e.g., `'MountainCar-v0'`) and implement the Random Agent.

```python
import gym
import matplotlib.pyplot as plt

# Initialize the MountainCar environment
env = gym.make('MountainCar-v0')

# Parameters
num_episodes = 10
max_steps = 200

# Store rewards for each episode
rewards_per_episode = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    for step in range(max_steps):
        env.render()  # Render the environment
        
        action = env.action_space.sample()  # Random action
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"Episode {episode + 1} finished after {step + 1} steps with total reward {total_reward}")
            break
    
    rewards_per_episode.append(total_reward)

env.close()

# Plotting rewards
plt.figure(figsize=(10,5))
plt.plot(range(1, num_episodes + 1), rewards_per_episode, marker='o', linestyle='-', color='g')
plt.title('Random Agent Performance in MountainCar-v0')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.xticks(range(1, num_episodes + 1))
plt.grid(True)
plt.show()
```

**Expected Outcome:** Observe how the Random Agent performs in a different environment, highlighting environment-specific challenges.

---

## **Observations**

- **Performance Variability:** The Random Agent exhibits inconsistent performance across episodes due to the stochastic nature of action selection.
- **Environment Differences:** Different environments present varying levels of difficulty for the Random Agent. For example:
  - **CartPole-v1:** Balancing the pole can be challenging with random actions, leading to episodes ending quickly.
  - **MountainCar-v0:** The agent needs to build momentum to reach the goal, which is difficult with random actions.

---

## **Summary**

By implementing a Random Agent, you've established a **performance baseline** for your RL experiments. This agent provides a reference point to evaluate the effectiveness of more advanced algorithms, such as Q-Learning or Policy Gradients, in future lessons. Understanding the Random Agent's limitations underscores the need for intelligent decision-making in RL, highlighting the improvements achievable through learning-based approaches.

---

## **Best Practices When Implementing RL Agents**

1. **Consistent Environment Initialization:**
   - Always reset the environment at the start of each episode to ensure consistency.
   
2. **Resource Management:**
   - Properly close the environment using `env.close()` to free up resources, especially when rendering.
   
3. **Tracking Metrics:**
   - Keep track of rewards and steps to analyze agent performance over time.
   
4. **Visualization:**
   - Use rendering and plotting to gain intuitive insights into agent behavior and performance trends.
   
5. **Reproducibility:**
   - Set random seeds if you need reproducible results.
   - **Example:**
     ```python
     env.seed(42)
     ```
   
6. **Modular Code Structure:**
   - Organize your code into functions or classes for better readability and maintenance.
   
   ```python
   def run_random_agent(env_name, num_episodes=10, max_steps=200):
       env = gym.make(env_name)
       rewards = []
       
       for episode in range(num_episodes):
           state = env.reset()
           total_reward = 0
           done = False
           
           for step in range(max_steps):
               action = env.action_space.sample()
               next_state, reward, done, info = env.step(action)
               total_reward += reward
               
               if done:
                   print(f"Episode {episode + 1} finished after {step + 1} steps with total reward {total_reward}")
                   break
           
           rewards.append(total_reward)
       
       env.close()
       return rewards
   ```

---

## **Further Reading and Resources**
- **OpenAI Gym Documentation:** [https://gym.openai.com/docs/](https://gym.openai.com/docs/)
- **Gym Environments List:** [https://gym.openai.com/envs/](https://gym.openai.com/envs/)
- **Stable Baselines3 Documentation:** [https://stable-baselines3.readthedocs.io/en/master/](https://stable-baselines3.readthedocs.io/en/master/)
- **Spinning Up in Deep RL:** [https://spinningup.openai.com/en/latest/](https://spinningup.openai.com/en/latest/)
- **Deep Reinforcement Learning Hands-On by Maxim Lapan:** [Book Link](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994)

---

**Excellent work on completing Lesson 2.2!** You've successfully implemented a Random Agent, established a performance baseline, and gained hands-on experience interacting with different Gym environments. This foundational understanding is essential as we progress to more sophisticated RL agents and algorithms in the upcoming lessons.
```