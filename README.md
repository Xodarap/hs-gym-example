# gym-test

A Haskell testing project for [gym-hs](../gym-hs), demonstrating reinforcement learning environment integration without the heavy hasktorch dependency.

## Overview

This project provides a lightweight way to test and experiment with OpenAI Gymnasium environments from Haskell. It uses the local `gym-hs` library to interface with Python's Gymnasium, enabling reinforcement learning experiments in pure Haskell.

## Features

- **No hasktorch dependency**: Removed heavy tensor library requirements
- **gym-hs integration**: Direct interface to OpenAI Gymnasium environments
- **CartPole-v1 demo**: Working example with the classic CartPole balancing task
- **Simple API**: Easy-to-understand code structure for learning and experimentation

## Prerequisites

- GHC 9.6.7 or compatible
- Python 3.8+ with gymnasium installed
- Cabal 3.0+

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

Run the CartPole-v1 demonstration:

```bash
cabal run
```

This will:
1. Create a CartPole-v1 environment
2. Reset the environment and show initial observation
3. Run 10 steps with alternating actions (left/right)
4. Display rewards, termination, and truncation status

## Example Output

```
Testing gym-hs library integration...
Creating CartPole-v1 environment...
Environment created successfully!
Resetting environment...
Initial observation: Array [Number (-3.4837662242352962e-3),Number (-2.2283699363470078e-3),Number 2.6785125955939293e-2,Number 2.8203139081597328e-2]
Running 10 random steps...
Step 1: action=Number 0.0, reward=1.0, terminated=False, truncated=False
Step 2: action=Number 1.0, reward=1.0, terminated=False, truncated=False
...
Step 10: action=Number 1.0, reward=1.0, terminated=False, truncated=False
gym-hs test completed successfully!
```

## Code Structure

### Main Components

- **`app/Main.hs`**: Main executable demonstrating gym-hs usage
- **`gym-test.cabal`**: Project configuration with gym-hs dependency
- **`cabal.project`**: Multi-package configuration linking to ../gym-hs

### Key Functions

```haskell
-- Create and manage environments
makeEnv :: Text -> IO (Either GymError Environment)
closeEnv :: Environment -> IO ()

-- Environment interaction
reset :: Environment -> IO (Either GymError Observation)
step :: Environment -> Action -> IO (Either GymError StepResult)
```

## CartPole Environment

The demonstration uses CartPole-v1, where:
- **Observation**: 4 values [cart position, cart velocity, pole angle, pole angular velocity]
- **Actions**: 0 (push left) or 1 (push right)
- **Reward**: 1.0 for each step the pole remains upright
- **Termination**: Episode ends when pole falls too far or cart moves too far

## Dependencies

This project depends on:
- `base >= 4.16.4.0 && < 5`
- `aeson >= 2.0` (for JSON communication)
- `gym-hs` (local dependency from ../gym-hs)

## Extending the Project

To test other Gymnasium environments:

1. Replace `"CartPole-v1"` with your desired environment name
2. Adjust the action space in `runSteps` function
3. Handle different observation and action formats as needed

Popular environments to try:
- `"MountainCar-v0"`
- `"Acrobot-v1"`
- `"LunarLander-v3"`

## Troubleshooting

**"No module named 'gymnasium'"**: Install with `pip3 install gymnasium`

**Build errors**: Ensure you're in the gym-test directory and ../gym-hs exists

**Runtime hangs**: Check that Python 3 is available and gymnasium is properly installed

## Related Projects

- [gym-hs](../gym-hs): The Haskell bindings library this project uses
- [OpenAI Gymnasium](https://gymnasium.farama.org/): The Python RL environment library

## License

BSD-3-Clause