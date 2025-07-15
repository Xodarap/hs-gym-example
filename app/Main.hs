{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.Aeson (Value(Number, Array))
import qualified Data.Vector as V
import Control.Exception (bracket)
import Control.Monad (when)
import System.Random
import Data.List (maximumBy)
import Data.Function (on)
import qualified Data.Sequence as Seq
import Data.Sequence (Seq)
import qualified Data.Map as Map
import Data.Map (Map)

import Numeric.LinearAlgebra hiding (Vector, (!))
import qualified Numeric.LinearAlgebra as LA

import Gym.Environment
import Gym.Core

-- | Simple neural network using hmatrix
data NeuralNetwork = NeuralNetwork
  { w1 :: Matrix Double  -- Input to hidden layer weights (4x24)
  , b1 :: LA.Vector Double  -- Hidden layer biases (24)
  , w2 :: Matrix Double  -- Hidden to hidden layer weights (24x24)
  , b2 :: LA.Vector Double  -- Hidden layer biases (24)
  , w3 :: Matrix Double  -- Hidden to output layer weights (24x2)
  , b3 :: LA.Vector Double  -- Output layer biases (2)
  } deriving (Show)

-- | Experience replay buffer entry
data Experience = Experience
  { expState :: LA.Vector Double
  , expAction :: Int
  , expReward :: Double
  , expNextState :: Maybe (LA.Vector Double)
  , expDone :: Bool
  } deriving (Show)

-- | DQN Agent configuration
data DQNConfig = DQNConfig
  { dqnLearningRate :: Double
  , dqnEpsilon :: Double
  , dqnEpsilonDecay :: Double
  , dqnEpsilonMin :: Double
  , dqnGamma :: Double
  , dqnBatchSize :: Int
  , dqnMemorySize :: Int
  } deriving (Show)

-- | DQN Agent state
data DQNAgent = DQNAgent
  { agentNetwork :: NeuralNetwork
  , agentMemory :: Seq Experience
  , agentConfig :: DQNConfig
  } deriving (Show)

-- | Default DQN configuration
defaultDQNConfig :: DQNConfig
defaultDQNConfig = DQNConfig
  { dqnLearningRate = 0.001
  , dqnEpsilon = 1.0
  , dqnEpsilonDecay = 0.995
  , dqnEpsilonMin = 0.01
  , dqnGamma = 0.95
  , dqnBatchSize = 32
  , dqnMemorySize = 2000
  }

-- | Tanh activation function
tanh' :: Matrix Double -> Matrix Double
tanh' = cmap tanh

-- | ReLU activation function
relu :: Matrix Double -> Matrix Double
relu = cmap (max 0)

-- | Initialize random neural network
randomNetwork :: IO NeuralNetwork
randomNetwork = do
  gen <- newStdGen
  let (w1, gen1) = randomMatrix gen 24 4 (-0.5, 0.5)
      (b1, gen2) = randomVec gen1 24 (-0.5, 0.5)
      (w2, gen3) = randomMatrix gen2 24 24 (-0.5, 0.5)
      (b2, gen4) = randomVec gen3 24 (-0.5, 0.5)
      (w3, gen5) = randomMatrix gen4 2 24 (-0.5, 0.5)
      (b3, _) = randomVec gen5 2 (-0.5, 0.5)
  return $ NeuralNetwork w1 b1 w2 b2 w3 b3

-- | Generate random matrix
randomMatrix :: StdGen -> Int -> Int -> (Double, Double) -> (Matrix Double, StdGen)
randomMatrix gen rows cols range =
  let (values, newGen) = randomList gen (rows * cols) range
      matrix = (rows >< cols) values
  in (matrix, newGen)

-- | Generate random vector
randomVec :: StdGen -> Int -> (Double, Double) -> (LA.Vector Double, StdGen)
randomVec gen size range =
  let (values, newGen) = randomList gen size range
      vector = fromList values
  in (vector, newGen)

-- | Generate list of random doubles
randomList :: StdGen -> Int -> (Double, Double) -> ([Double], StdGen)
randomList gen 0 _ = ([], gen)
randomList gen n range =
  let (value, newGen) = randomR range gen
      (rest, finalGen) = randomList newGen (n-1) range
  in (value : rest, finalGen)

-- | Forward pass through neural network
forwardPass :: NeuralNetwork -> LA.Vector Double -> LA.Vector Double
forwardPass net input =
  let hidden1Vec = (w1 net) #> input + (b1 net)
      hidden1 = cmap tanh hidden1Vec
      hidden2Vec = (w2 net) #> hidden1 + (b2 net)
      hidden2 = cmap tanh hidden2Vec
      output = (w3 net) #> hidden2 + (b3 net)
  in output

-- | Initialize a new DQN agent
initDQNAgent :: DQNConfig -> IO DQNAgent
initDQNAgent config = do
  network <- randomNetwork
  return $ DQNAgent network Seq.empty config

-- | Parse observation from gym-hs to vector
parseObservation :: Value -> Maybe (LA.Vector Double)
parseObservation (Array arr) = 
  let values = V.toList arr
      doubles = mapM parseNumber values
  in fmap (fromList . V.toList . V.fromList) doubles
  where
    parseNumber (Number n) = Just (realToFrac n)
    parseNumber _ = Nothing
parseObservation _ = Nothing

-- | Choose action using epsilon-greedy policy
chooseAction :: DQNAgent -> LA.Vector Double -> IO Int
chooseAction agent state = do
  prob <- randomRIO (0.0, 1.0)
  if prob < dqnEpsilon (agentConfig agent)
    then randomRIO (0, 1)  -- Random action
    else do
      let qValues = forwardPass (agentNetwork agent) state
          action = if qValues LA.! 0 > qValues LA.! 1 then 0 else 1
      return action

-- | Add experience to replay buffer
addExperience :: DQNAgent -> Experience -> DQNAgent
addExperience agent exp =
  let memory = agentMemory agent
      config = agentConfig agent
      newMemory = if Seq.length memory >= dqnMemorySize config
                  then Seq.drop 1 memory Seq.|> exp
                  else memory Seq.|> exp
  in agent { agentMemory = newMemory }

-- | Update epsilon (exploration rate)
updateEpsilon :: DQNAgent -> DQNAgent
updateEpsilon agent =
  let config = agentConfig agent
      newEpsilon = max (dqnEpsilonMin config) 
                      (dqnEpsilon config * dqnEpsilonDecay config)
      newConfig = config { dqnEpsilon = newEpsilon }
  in agent { agentConfig = newConfig }

-- | Run a single episode
runEpisode :: DQNAgent -> Environment -> IO (DQNAgent, Double, Int)
runEpisode agent env = do
  resetResult <- reset env
  case resetResult of
    Left err -> do
      putStrLn $ "Reset error: " ++ show err
      return (agent, 0, 0)
    Right (Observation obs) -> do
      case parseObservation obs of
        Nothing -> do
          putStrLn "Failed to parse initial observation"
          return (agent, 0, 0)
        Just state -> do
          (finalAgent, totalReward, steps) <- runSteps agent env state 0 0
          return (finalAgent, totalReward, steps)
  where
    runSteps :: DQNAgent -> Environment -> LA.Vector Double -> Double -> Int -> IO (DQNAgent, Double, Int)
    runSteps currentAgent env currentState totalReward stepCount = do
      action <- chooseAction currentAgent currentState
      stepResult <- Gym.Environment.step env (Action (Number (fromIntegral action)))
      
      case stepResult of
        Left err -> do
          putStrLn $ "Step error: " ++ show err
          return (currentAgent, totalReward, stepCount)
        Right result -> do
          let reward = stepReward result
              done = stepTerminated result || stepTruncated result
              newTotalReward = totalReward + reward
              newStepCount = stepCount + 1
          
          nextState <- if done
            then return Nothing
            else let (Observation obsValue) = stepObservation result
                 in case parseObservation obsValue of
                      Nothing -> return Nothing
                      Just s -> return (Just s)
          
          let experience = Experience currentState action reward nextState done
              updatedAgent = addExperience currentAgent experience
          
          if done || newStepCount >= 500  -- Max steps per episode
            then return (updatedAgent, newTotalReward, newStepCount)
            else case nextState of
              Nothing -> return (updatedAgent, newTotalReward, newStepCount)
              Just ns -> runSteps updatedAgent env ns newTotalReward newStepCount

-- | Training loop
trainAgent :: DQNAgent -> Environment -> Int -> IO DQNAgent
trainAgent agent env numEpisodes = go agent 1
  where
    go currentAgent episode
      | episode > numEpisodes = return currentAgent
      | otherwise = do
          (newAgent, reward, steps) <- runEpisode currentAgent env
          
          when (episode `mod` 10 == 0) $
            putStrLn $ "Episode " ++ show episode ++ 
                      ": Reward = " ++ show reward ++ 
                      ", Steps = " ++ show steps ++ 
                      ", Epsilon = " ++ show (dqnEpsilon $ agentConfig newAgent)
          
          let trainedAgent = updateEpsilon newAgent
          go trainedAgent (episode + 1)

-- | Test the trained agent
testAgent :: DQNAgent -> Environment -> Int -> IO ()
testAgent agent env numEpisodes = go 1
  where
    go episode
      | episode > numEpisodes = return ()
      | otherwise = do
          let testAgent = agent { agentConfig = (agentConfig agent) { dqnEpsilon = 0.0 } }
          (_, reward, steps) <- runEpisode testAgent env
          putStrLn $ "Test Episode " ++ show episode ++ 
                    ": Reward = " ++ show reward ++ 
                    ", Steps = " ++ show steps
          go (episode + 1)

main :: IO ()
main = do
    putStrLn "Deep Q-Network Training with hmatrix and gym-hs"
    putStrLn "==============================================="
    
    putStrLn "Creating CartPole-v1 environment..."
    result <- makeEnv "CartPole-v1"
    case result of
        Left err -> putStrLn $ "Error creating environment: " ++ show err
        Right env -> bracket (return env) closeEnv $ \env -> do
            putStrLn "Environment created successfully!"
            
            putStrLn "Initializing DQN agent..."
            agent <- initDQNAgent defaultDQNConfig
            putStrLn "Agent initialized!"
            
            putStrLn "Starting training..."
            trainedAgent <- trainAgent agent env 1000
            
            putStrLn "\nTraining completed! Testing agent..."
            testAgent trainedAgent env 5
            
            putStrLn "\nDQN training and testing completed!"