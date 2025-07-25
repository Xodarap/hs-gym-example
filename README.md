# Haskell Deep Q-Network (DQN)

A complete Deep Q-Network implementation in Haskell using backprop for automatic differentiation and gym-hs for OpenAI Gymnasium environment integration.

## Overview

This project implements a Deep Q-Network agent that learns to play CartPole-v1 through reinforcement learning. The implementation features experience replay, target networks, epsilon-greedy exploration, and proper Q-learning with temporal difference updates.

## Installation

1. **Install Python dependencies**:
   ```bash
   pip3 install gymnasium
   ```

2. **Build the project**:
   ```bash
   cabal build
   ```

## Usage

Run the DQN training:

```bash
cabal run
```

This will:
1. Initialize a DQN with Glorot weight initialization
2. Show initial performance (before training)
3. Train for 500 episodes with experience replay and target network updates
4. Display final performance and improvement metrics

## Example Output

```
Initializing DQN with Glorot initialization...

=== INITIAL PERFORMANCE (BEFORE TRAINING) ===
Initial average trajectory length: 10 steps
Initial trajectory lengths: [9,10,9,10,9,9,9,10,10,11]
Initial Q-values: [0.033,-0.462]

=== DQN TRAINING ===
Training with experience replay, target network updates, and epsilon decay
Episode 10, Q-values: [0.751,0.519], Trajectory length: 10, Buffer size: 1000
Episode 50, Q-values: [1.295,1.255], Trajectory length: 14, Buffer size: 5000
Updating target network at episode 50
...
Episode 500, Q-values: [8.817,8.683], Trajectory length: 46, Buffer size: 10000

DQN training completed!

=== FINAL PERFORMANCE (AFTER DQN TRAINING) ===
Final average trajectory length: 42 steps
Final trajectory lengths: [59,30,44,26,27,37,41,28,97,28]
Final Q-values: [8.817,8.683]

=== TRAINING SUMMARY ===
Performance improvement: 32 steps
Performance improvement: 334% (10 → 42 steps)
Q-value changes:
  Action 0: 0.033 → 8.817
  Action 1: -0.462 → 8.683
```
