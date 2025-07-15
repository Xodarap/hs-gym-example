{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.Aeson (Value(Number, Array))
import qualified Data.Vector as V
import Control.Exception (bracket)
import Control.Monad (when)
import System.Random
import Data.List (maximumBy)
import Data.Function (on)
import qualified Data.Map as Map
import Data.Map (Map)

import Gym.Environment
import Gym.Core

-- | State discretization for tabular Q-learning
-- CartPole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
data DiscreteState = DiscreteState Int Int Int Int deriving (Show, Eq, Ord)

-- | Action space for CartPole (0 = left, 1 = right)
type QLAction = Int

-- | Q-Table: maps (state, action) pairs to Q-values
type QTable = Map (DiscreteState, QLAction) Double

-- | Q-Learning agent configuration
data QLearningConfig = QLearningConfig
  { qLearningRate :: Double    -- α (alpha)
  , qDiscount :: Double        -- γ (gamma) 
  , qEpsilon :: Double         -- ε (epsilon) for ε-greedy
  , qEpsilonDecay :: Double    -- ε decay rate
  , qEpsilonMin :: Double      -- minimum ε
  } deriving (Show)

-- | Q-Learning agent state
data QLearningAgent = QLearningAgent
  { qTable :: QTable
  , qConfig :: QLearningConfig
  } deriving (Show)

-- | Default Q-Learning configuration
defaultQLearningConfig :: QLearningConfig
defaultQLearningConfig = QLearningConfig
  { qLearningRate = 0.1
  , qDiscount = 0.95
  , qEpsilon = 1.0
  , qEpsilonDecay = 0.995
  , qEpsilonMin = 0.01
  }

-- | Initialize Q-Learning agent
initQLearningAgent :: QLearningConfig -> QLearningAgent
initQLearningAgent config = QLearningAgent Map.empty config

-- | Discretize continuous CartPole state to discrete bins
-- This is crucial for tabular Q-learning to work effectively
discretizeState :: [Double] -> DiscreteState
discretizeState [cart_pos, cart_vel, pole_angle, pole_vel] =
  let -- Discretize cart position (-2.4 to 2.4) -> 10 bins
      pos_bin = max 0 $ min 9 $ floor ((cart_pos + 2.4) / 4.8 * 10)
      
      -- Discretize cart velocity (-3.0 to 3.0) -> 10 bins  
      vel_bin = max 0 $ min 9 $ floor ((cart_vel + 3.0) / 6.0 * 10)
      
      -- Discretize pole angle (-0.5 to 0.5 radians) -> 10 bins
      angle_bin = max 0 $ min 9 $ floor ((pole_angle + 0.5) / 1.0 * 10)
      
      -- Discretize pole angular velocity (-2.0 to 2.0) -> 10 bins
      ang_vel_bin = max 0 $ min 9 $ floor ((pole_vel + 2.0) / 4.0 * 10)
      
  in DiscreteState pos_bin vel_bin angle_bin ang_vel_bin
discretizeState _ = error "Expected 4 state values for CartPole"

-- | Parse observation from gym-hs to state list
parseObservationQL :: Value -> Maybe [Double]
parseObservationQL (Array arr) = 
  let values = V.toList arr
      doubles = mapM parseNumber values
  in doubles
  where
    parseNumber (Number n) = Just (realToFrac n)
    parseNumber _ = Nothing
parseObservationQL _ = Nothing

-- | Get Q-value for a state-action pair (returns 0 if not in table)
getQValue :: QLearningAgent -> DiscreteState -> QLAction -> Double
getQValue agent state action = 
  Map.findWithDefault 0.0 (state, action) (qTable agent)

-- | Get best action for a state (greedy policy)
getBestAction :: QLearningAgent -> DiscreteState -> QLAction
getBestAction agent state =
  let q0 = getQValue agent state 0
      q1 = getQValue agent state 1
  in if q0 > q1 then 0 else 1

-- | Choose action using ε-greedy policy
chooseActionQL :: QLearningAgent -> DiscreteState -> IO QLAction
chooseActionQL agent state = do
  prob <- randomRIO (0.0, 1.0)
  if prob < qEpsilon (qConfig agent)
    then randomRIO (0, 1)  -- Random action (exploration)
    else return $ getBestAction agent state  -- Greedy action (exploitation)

-- | Update Q-value using Q-learning update rule:
-- Q(s,a) = Q(s,a) + α[r + γ*max_a'(Q(s',a')) - Q(s,a)]
updateQValue :: QLearningAgent -> DiscreteState -> QLAction -> Double -> DiscreteState -> QLearningAgent
updateQValue agent state action reward nextState =
  let config = qConfig agent
      currentQ = getQValue agent state action
      
      -- Find maximum Q-value for next state
      maxNextQ = max (getQValue agent nextState 0) (getQValue agent nextState 1)
      
      -- Q-learning update rule
      target = reward + qDiscount config * maxNextQ
      newQ = currentQ + qLearningRate config * (target - currentQ)
      
      -- Update Q-table
      newTable = Map.insert (state, action) newQ (qTable agent)
      
  in agent { qTable = newTable }

-- | Update Q-value for terminal states (no next state)
updateQValueTerminal :: QLearningAgent -> DiscreteState -> QLAction -> Double -> QLearningAgent
updateQValueTerminal agent state action reward =
  let config = qConfig agent
      currentQ = getQValue agent state action
      
      -- For terminal states, there's no next state, so target is just the reward
      target = reward
      newQ = currentQ + qLearningRate config * (target - currentQ)
      
      -- Update Q-table
      newTable = Map.insert (state, action) newQ (qTable agent)
      
  in agent { qTable = newTable }

-- | Decay epsilon (reduce exploration over time)
decayEpsilon :: QLearningAgent -> QLearningAgent
decayEpsilon agent =
  let config = qConfig agent
      newEpsilon = max (qEpsilonMin config) 
                      (qEpsilon config * qEpsilonDecay config)
      newConfig = config { qEpsilon = newEpsilon }
  in agent { qConfig = newConfig }

-- | Run a single episode with Q-learning
runEpisodeQL :: QLearningAgent -> Environment -> IO (QLearningAgent, Double, Int)
runEpisodeQL agent env = do
  resetResult <- reset env
  case resetResult of
    Left err -> do
      putStrLn $ "Reset error: " ++ show err
      return (agent, 0, 0)
    Right (Observation obs) -> do
      case parseObservationQL obs of
        Nothing -> do
          putStrLn "Failed to parse initial observation"
          return (agent, 0, 0)
        Just stateValues -> do
          let initialState = discretizeState stateValues
          (finalAgent, totalReward, steps) <- runStepsQL agent env initialState 0 0
          return (finalAgent, totalReward, steps)

-- | Run steps within an episode
runStepsQL :: QLearningAgent -> Environment -> DiscreteState -> Double -> Int -> IO (QLearningAgent, Double, Int)
runStepsQL agent env currentState totalReward stepCount = do
  action <- chooseActionQL agent currentState
  stepResult <- Gym.Environment.step env (Action (Number (fromIntegral action)))
  
  case stepResult of
    Left err -> do
      putStrLn $ "Step error: " ++ show err
      return (agent, totalReward, stepCount)
    Right result -> do
      let reward = stepReward result
          done = stepTerminated result || stepTruncated result
          newTotalReward = totalReward + reward
          newStepCount = stepCount + 1
      
      if done || newStepCount >= 500  -- Max steps per episode
        then do
          -- Terminal state - update Q-value
          let updatedAgent = updateQValueTerminal agent currentState action reward
          return (updatedAgent, newTotalReward, newStepCount)
        else do
          -- Non-terminal state - get next state and update Q-value
          let (Observation nextObs) = stepObservation result
          case parseObservationQL nextObs of
            Nothing -> return (agent, newTotalReward, newStepCount)
            Just nextStateValues -> do
              let nextState = discretizeState nextStateValues
                  updatedAgent = updateQValue agent currentState action reward nextState
              runStepsQL updatedAgent env nextState newTotalReward newStepCount

-- | Training loop for Q-learning
trainQLearningAgent :: QLearningAgent -> Environment -> Int -> IO QLearningAgent
trainQLearningAgent agent env numEpisodes = go agent 1
  where
    go currentAgent episode
      | episode > numEpisodes = return currentAgent
      | otherwise = do
          (newAgent, reward, steps) <- runEpisodeQL currentAgent env
          
          when (episode `mod` 50 == 0) $
            putStrLn $ "Episode " ++ show episode ++ 
                      ": Reward = " ++ show reward ++ 
                      ", Steps = " ++ show steps ++ 
                      ", Epsilon = " ++ show (qEpsilon $ qConfig newAgent) ++
                      ", Q-table size = " ++ show (Map.size $ qTable newAgent)
          
          let trainedAgent = decayEpsilon newAgent
          go trainedAgent (episode + 1)

-- | Test the trained Q-learning agent
testQLearningAgent :: QLearningAgent -> Environment -> Int -> IO ()
testQLearningAgent agent env numEpisodes = go 1
  where
    go episode
      | episode > numEpisodes = return ()
      | otherwise = do
          -- Test with no exploration (epsilon = 0)
          let testAgent = agent { qConfig = (qConfig agent) { qEpsilon = 0.0 } }
          (_, reward, steps) <- runEpisodeQL testAgent env
          putStrLn $ "Test Episode " ++ show episode ++ 
                    ": Reward = " ++ show reward ++ 
                    ", Steps = " ++ show steps
          go (episode + 1)

-- | Main function for Q-learning CartPole
main :: IO ()
main = do
    putStrLn "Tabular Q-Learning for CartPole with gym-hs"
    putStrLn "============================================="
    
    putStrLn "Creating CartPole-v1 environment..."
    result <- makeEnv "CartPole-v1"
    case result of
        Left err -> putStrLn $ "Error creating environment: " ++ show err
        Right env -> bracket (return env) closeEnv $ \env -> do
            putStrLn "Environment created successfully!"
            
            putStrLn "Initializing Q-Learning agent..."
            let agent = initQLearningAgent defaultQLearningConfig
            putStrLn "Agent initialized!"
            
            putStrLn "Starting Q-learning training..."
            trainedAgent <- trainQLearningAgent agent env 500
            
            putStrLn $ "\nTraining completed! Final Q-table size: " ++ 
                      show (Map.size $ qTable trainedAgent)
            putStrLn "Testing agent..."
            testQLearningAgent trainedAgent env 5
            
            putStrLn "\nQ-learning training and testing completed!"