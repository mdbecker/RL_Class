```markdown
## **Lesson 1.1: What is Reinforcement Learning?**

### **Learning Objectives**
- **Define Reinforcement Learning (RL)** and distinguish it from supervised and unsupervised learning.
- **Identify real-world applications** of RL.
- **Understand the basic framework** of RL through practical examples.
- **Explore historical milestones** and significant breakthroughs in RL.
- **Implement a basic RL interaction** with an environment to solidify understanding.

### **Description**
Welcome to the first lesson of our Reinforcement Learning (RL) course! In this session, we'll explore what RL is, how it differs from other machine learning paradigms, and where it can be effectively applied. We'll also delve into the history and key milestones that have shaped the field of RL. By the end of this lesson, you'll have a foundational understanding of RL and gain hands-on experience interacting with a simple RL environment.

### **What is Reinforcement Learning?**
Reinforcement Learning is a type of machine learning where an **agent** learns to make decisions by performing **actions** in an **environment** to achieve maximum cumulative **reward**. Unlike supervised learning, where the model learns from labeled data, RL relies on the agent's interactions with the environment to learn optimal behaviors.

![Reinforcement Learning Cycle](https://miro.medium.com/max/1400/1*Z8JsJcTiJfZjC8kFjz8agQ.png)

*Figure: The Reinforcement Learning Loop - Agent interacts with Environment, receives Reward and next State.*

### **RL vs. Supervised and Unsupervised Learning**
- **Supervised Learning:**
  - **Objective:** Learn a mapping from inputs to outputs based on labeled data.
  - **Example:** Image classification where each image is labeled with the correct category.
  - **Feedback:** Direct and explicit through labeled examples.

- **Unsupervised Learning:**
  - **Objective:** Discover hidden patterns or intrinsic structures in input data without labeled responses.
  - **Example:** Clustering customers based on purchasing behavior.
  - **Feedback:** Implicit through the data's structure.

- **Reinforcement Learning:**
  - **Objective:** Learn to make sequences of decisions by interacting with an environment to maximize cumulative rewards.
  - **Example:** Training a game-playing agent to win by learning from gameplay outcomes.
  - **Feedback:** Delayed and based on the consequences of actions.

### **Historical Milestones in RL**
- **1950s-1960s:** 
  - Early concepts and foundational theories laid the groundwork for RL.
  - **Key Figure:** **Richard Bellman** introduced the Bellman Equation, a cornerstone of RL.
  
- **1980s:** 
  - Introduction of **Temporal Difference (TD) Learning** by **Richard Sutton**.
  - Development of **SARSA** and other TD-based algorithms.
  
- **1990s:** 
  - **Q-Learning** algorithm introduced by **Chris Watkins**, enabling agents to learn optimal policies without a model of the environment.
  - Emergence of **function approximation** methods to handle larger state spaces.
  
- **2010s:** 
  - Breakthroughs with **Deep Reinforcement Learning**, notably **DeepMind's AlphaGo** defeating the world champion in Go.
  - Integration of deep neural networks with RL algorithms, enabling learning in complex environments.

### **Real-World Applications of RL**
RL has been successfully applied in various domains, including:
- **Gaming:** Training agents to play games like Go, Chess, and video games (e.g., AlphaGo, OpenAI Five).
- **Robotics:** Enabling robots to perform tasks like walking, grasping, and navigation.
- **Finance:** Optimizing trading strategies and portfolio management.
- **Healthcare:** Personalizing treatment plans and managing healthcare resources.
- **Autonomous Vehicles:** Enhancing decision-making for self-driving cars.
- **Recommendation Systems:** Improving personalized content delivery based on user interactions.

### **Practical Example: Training a Simple Agent**
Let's dive into a hands-on example to illustrate the basics of RL using OpenAI Gym's **CartPole** environment. We'll create a simple agent that takes random actions to interact with the environment.

#### **Step 1: Setting Up the Environment**
First, ensure you have the necessary libraries installed. You can install OpenAI Gym using pip if you haven't already:

```python
!pip install gym
```

#### **Step 2: Interacting with the Environment**
We'll create the CartPole environment, reset it to start, take a random action, and observe the results.

```python
import gym

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Reset the environment to start
state = env.reset()

# Render the initial state
env.render()

# Take a random action
action = env.action_space.sample()

# Apply the action to the environment
next_state, reward, done, info = env.step(action)

print(f"Next State: {next_state}")
print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Info: {info}")

# Close the environment
env.close()
```

#### **Step 3: Understanding the Code**
- **Environment:** The `CartPole-v1` environment simulates a pole balanced on a cart. The goal is to prevent the pole from falling by moving the cart left or right.
- **State:** Represents the current situation of the environment, including:
  - Cart position
  - Cart velocity
  - Pole angle
  - Pole velocity at the tip
- **Action:** The possible moves the agent can take:
  - `0`: Push cart to the left
  - `1`: Push cart to the right
- **Reward:** Feedback from the environment based on the action taken. Typically, +1 for every step the pole remains upright.
- **Done:** Indicates whether the episode has ended (e.g., the pole has fallen or the cart has moved out of bounds).
- **Info:** Additional diagnostic information (can be ignored for basic RL tasks).

### **Extending the Example: Running a Random Policy**
To better understand how actions affect the environment over time, let's run a loop where the agent takes random actions until the episode ends.

```python
import gym

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Reset the environment to start
state = env.reset()

total_reward = 0
steps = 0
done = False

while not done:
    env.render()  # Render the environment
    action = env.action_space.sample()  # Take a random action
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    steps += 1

print(f"Episode finished after {steps} steps with total reward {total_reward}")

# Close the environment
env.close()
```

*Run the above code cell to see the CartPole in action with a random policy.*

### **Explanation**
In this extended example:
- The agent interacts with the environment by taking random actions until the episode ends.
- **Rendering** visualizes the environment, allowing you to see the cart and pole.
- **Total Reward** accumulates the rewards received in each step, giving an indication of how long the pole was balanced.
- **Steps:** Counts the number of actions taken before the episode ended.

### **Interactive Discussion**
- **Question:** How does RL differ fundamentally from supervised learning in terms of feedback?
  
  **Answer:** In supervised learning, the model receives direct and explicit feedback through labeled input-output pairs, allowing it to learn the mapping between them. In contrast, RL involves learning through interaction, where the agent receives delayed and indirect feedback in the form of rewards or penalties based on its actions, without explicit labels for each action.

- **Activity:** Identify an RL application in your field of interest and discuss how the agent interacts with its environment.
  
  **Example Response:** *In finance, an RL agent can be used to optimize trading strategies. The environment consists of the financial market, where the agent takes actions such as buying or selling assets. The rewards are based on the profitability of these trades over time.*

### **Quick Exercise: Modify the Random Agent**
Try modifying the random agent to take a fixed action (e.g., always push right) and observe how it affects the environment's behavior.

```python
import gym

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Reset the environment to start
state = env.reset()

total_reward = 0
steps = 0
done = False

while not done:
    env.render()  # Render the environment
    action = 1  # Always push right
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    steps += 1

print(f"Episode finished after {steps} steps with total reward {total_reward}")

# Close the environment
env.close()
```

*Observe how a fixed action policy performs compared to a random policy.*

### **Further Reading and Resources**
- **"Reinforcement Learning: An Introduction" by Richard S. Sutton and Andrew G. Barto** - A comprehensive textbook on RL fundamentals.
- **OpenAI Gym Documentation:** [https://gym.openai.com/docs/](https://gym.openai.com/docs/)
- **DeepMind's RL Courses:** [https://deepmind.com/learning-resources](https://deepmind.com/learning-resources)

---

**Great job on completing Lesson 1.1!** You've laid the groundwork for understanding what RL is and how it differs from other machine learning paradigms. In the next lesson, we'll delve deeper into the key components of RL and how they interact to enable learning.

```