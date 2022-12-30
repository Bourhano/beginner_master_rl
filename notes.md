# Notes

# MDP: Markov Decision Process

decision process : Discrete-time, stochastic (S_t+1 depends only partially on taken actions A_t), control process
variables: (S, A, R, P) to describe a control task
            Space
            Action
            Reward
            Possible transitions (Propabilities)
        The process has no memory, only depends on the current state.
    
    Types:
        - Finite MDP
        - Infinite MDP
    Episodic vs Continuing --> whether it hasn ending condition or no, in this course YES: Episodic

Actions can be deterministic or probabilistic

Trajectory and Episode:
    - Trajectory: a sequence of State action (moving from a state to another) 
    - Episode: a trajectory starting from the initial state and ending in the terminal state?

Reward and Return:
    - Reward: immediate return that an action r produces
    - Return: the cumulative reward untill time t, the goal is to maximise this return with the decisions taken
        G_t = R_t + R_(t+1) + ... + R_T

Discount factor: a factor in (0, 1) multiplied by the reward at each instant:
        **Discounted Return**: G_0 = R_1 + \gammaR_2 + \gamma^2R_3 + ... + \gamma^(T-t-1)R_T
            default value: 0.99, how much the agent is blinded from the future.
        \gamma = 0 --> GREEDY
        \gamma = 1 --> OPTIMAL DECISION

Policy:
    Determines what actions we are willing to take: \Pi
        \Pi* is the optimal policy thta is going to maximize the return of a task.

State Value and Action Value:
    - State Value: v_pi(s) = |E [G_t|S_t=s], following policy \pi
    - Action Value: q-value of a state-action pair
        q_pi(s, a) = |E [G_t |S_t=s, A_t=a], following policy \pi

Bellman equation for v(s):
    v_pi(s) = |E[G_t|S_t=s]
            = |E[R_t+1 + \gamma G_(t+1) |S_t=s]
            = \sum_a pi(a|s) \sum_(s', r) p(s', r|s, a) [r + \gamma v_pi(s')]
        we represent a state value in function of another state value.
        which is an expansion that can be optimized to get the optimal policy.

Bellman equation for q(s, a):
    q_pi(s, a) = |E[G_t|S_t=s, A_t=a]
            = |E[R_t+1 + \gamma G_(t+1) |S_t=s, A_t=a]
            = \sum_(s', r) p(s', r|s, a) [r + \gamma \sum_a' pi(a'|s') q_pi(s', a')]

## 3 Dynamic Programming

Finding the optimal policy: \pi*

We have overlapping solutions ( they are mutually dependant ), hence we use the Bellman equations to solve these equations dynamically with an update rule until reaching a better estimate.

Problem: we need access to the transition probabilities to start with a certzin value (initial state)

### Value Iteration

We have to find the optimal policy, we do so with an update rule:
    We update the estimated value of our state: v(s) 
    repeat until the delta is small enough.

### Policy Iteration

Improve alternately the estimated values and policy.

## 4 Monte Carlo Methods

Learn optimal values v*(s) or q*(s, a) based on samples (sampling)
LLN: the more we sample, the more accurate the results are.

On-policy and off-policy methods

epsilon-greedy policy (PageRank like jumps)
MonteCarlo control

Off-policy: keep track of a learning policy b to find optimal policy Pi

## 5 Temporal Difference Methods

Combination of MC methods and dynamic programming.

Early decisions affect later ones, unlike Monte Carlo methods.

SARSA: on-policy 

Q-learning: off-policy

Advantages: Faster and calculates qvalues in live time

## 6 n-step bootstrapping (Temporal Difference Method)

This method does the estimation by cosidering n-steps in the future.

SARSA is a 1-step bootstrapping method

n-step bootstrapping is a contrast between temporal difference methods that only consider the very next step (t+1) and between the Monte Carlo methods that take into considerations all the future possibilities until the end of the episode.

Bias-Variance Tradeoff:
the higher the n the more the variance and less bias

Proposition: n-step SARSA

# Classic Control Tasks

CartPole: Keep the tip of the pole straight.
Acrobot: Swing the bar up to a certain height.
MountainCar: Reach the goal from the bottom of the valley.
Pendulum: swing it and keep it upright

## Discrete Control Tasks

## Continuous Control Tasks

### State Aggregation
Classical RL algorithm that consists of discretizing a continuous space.
SARSA cannot be applied by default: solution is to discretize the space then carry on with it.


### Tile Coding
An aggregation of multiple state aggragations
Advantage: smoothens the discrete policies

## 8 Neural Networks as Function Estimators

## 9 Deep SARSA
On-Policy algo

## 10 Deep Q-Learning (DQN: Deep Q-Network)
Off-Policy algo

## 11 reinforce
Policy gradient methods: 
Use a function estimator to predict the probabilities of executing the next action. The neural network is the POLICY.

Value-based methods cannot represent stochastic policies unlinke policy gradient methods.