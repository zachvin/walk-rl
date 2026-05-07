# Outline

One network class
One agent class
One environment class

Main file with imports and A2C implementation

A2C
- Play N steps; for each step: get state, action, reward
- Set R <-- 0 at end of episode and work backward
    - Save reward and accumulate policy and value gradients (accumulate = loss function)