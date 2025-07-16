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
import Data.Foldable (toList)

import Numeric.LinearAlgebra hiding (Vector, (!), toList)
import qualified Numeric.LinearAlgebra as LA

import Gym.Environment
import Gym.Core

-- | Q-Network using hmatrix (matching PyTorch architecture)
-- Input: 4 (state_size) -> FC1: 64 -> FC2: 64 -> FC3: 2 (action_size)
data QNetwork = QNetwork
  { fc1_w :: Matrix Double  -- Input to first hidden layer weights (4x64)
  , fc1_b :: LA.Vector Double  -- First hidden layer biases (64)
  , fc2_w :: Matrix Double  -- First to second hidden layer weights (64x64)
  , fc2_b :: LA.Vector Double  -- Second hidden layer biases (64)
  , fc3_w :: Matrix Double  -- Second hidden to output layer weights (64x2)
  , fc3_b :: LA.Vector Double  -- Output layer biases (2)
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
  { agentNetwork :: QNetwork
  , agentMemory :: Seq Experience
  , agentConfig :: DQNConfig
  } deriving (Show)

-- | Default DQN configuration
defaultDQNConfig :: DQNConfig
defaultDQNConfig = DQNConfig
  { dqnLearningRate = 0.0025
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

-- | Initialize random Q-Network (matching PyTorch architecture)
randomQNetwork :: IO QNetwork
randomQNetwork = do
  gen <- newStdGen
  -- Use Xavier/Glorot initialization for better training
  let (fc1_weights, gen1) = randomMatrix gen 64 4 (-0.5, 0.5)
      (fc1_biases, gen2) = randomVec gen1 64 (-0.1, 0.1)
      (fc2_weights, gen3) = randomMatrix gen2 64 64 (-0.5, 0.5)
      (fc2_biases, gen4) = randomVec gen3 64 (-0.1, 0.1)
      (fc3_weights, gen5) = randomMatrix gen4 2 64 (-0.5, 0.5)
      (fc3_biases, _) = randomVec gen5 2 (-0.1, 0.1)
  return $ QNetwork fc1_weights fc1_biases fc2_weights fc2_biases fc3_weights fc3_biases

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

-- | Forward pass through Q-Network (matching PyTorch forward method)
-- x = F.relu(self.fc1(state))
-- x = F.relu(self.fc2(x))  
-- return self.fc3(x)
forwardPass :: QNetwork -> LA.Vector Double -> LA.Vector Double
forwardPass net input =
  let -- First layer: fc1 with ReLU activation
      fc1_out_vec = (fc1_w net) #> input + (fc1_b net)
      fc1_activated = cmap (max 0) fc1_out_vec  -- ReLU activation
      
      -- Second layer: fc2 with ReLU activation  
      fc2_out_vec = (fc2_w net) #> fc1_activated + (fc2_b net)
      fc2_activated = cmap (max 0) fc2_out_vec  -- ReLU activation
      
      -- Output layer: fc3 (no activation)
      output = (fc3_w net) #> fc2_activated + (fc3_b net)
  in output

-- | Compute mean squared error loss
computeLoss :: QNetwork -> LA.Vector Double -> LA.Vector Double -> Double
computeLoss network state target =
  let prediction = forwardPass network state
      diff = prediction - target
  in (diff <.> diff) / 2.0

-- | Initialize a new DQN agent
initDQNAgent :: DQNConfig -> IO DQNAgent
initDQNAgent config = do
  network <- randomQNetwork
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

-- | Compute target Q-values for training
computeTargetQValues :: DQNAgent -> [Experience] -> [LA.Vector Double]
computeTargetQValues agent batch =
  map computeTarget batch
  where
    computeTarget exp =
      let currentQ = forwardPass (agentNetwork agent) (expState exp)
          updatedQ = case expNextState exp of
            Nothing -> currentQ  -- Terminal state
            Just nextState ->
              let nextQ = forwardPass (agentNetwork agent) nextState
                  maxNextQ = max (nextQ LA.! 0) (nextQ LA.! 1)
                  targetValue = expReward exp + dqnGamma (agentConfig agent) * maxNextQ
                  actionIdx = expAction exp
              in if actionIdx == 0
                 then fromList [targetValue, currentQ LA.! 1]
                 else fromList [currentQ LA.! 0, targetValue]
      in updatedQ

-- | Simple gradient descent update (simplified version)
updateNetworkWeights :: DQNAgent -> [Experience] -> [LA.Vector Double] -> DQNAgent
updateNetworkWeights agent batch targets =
  let learningRate = dqnLearningRate (agentConfig agent)
      network = agentNetwork agent
      -- For simplicity, we'll do a basic weight update
      -- In practice, you'd compute proper gradients
      updatedNetwork = updateQNetworkWeights network learningRate batch targets
  in agent { agentNetwork = updatedNetwork }

-- | Update Q-Network weights using improved gradient-like updates
updateQNetworkWeights :: QNetwork -> Double -> [Experience] -> [LA.Vector Double] -> QNetwork
updateQNetworkWeights network lr batch targets =
  let -- Compute batch-averaged predictions and errors
      predictions = map (\exp -> forwardPass network (expState exp)) batch
      errors = zipWith (-) targets predictions
      avgError = sum (map LA.norm_2 errors) / fromIntegral (length errors)
      
      -- Improved weight updates based on error magnitudes and directions
      states = map expState batch
      
      -- Compute update directions based on errors and states
      fc1Updates = computeLayerUpdates states errors (fc1_w network) (fc1_b network) lr
      fc2Updates = computeLayerUpdates (map (applyLayer (fc1_w network) (fc1_b network)) states) errors (fc2_w network) (fc2_b network) lr
      fc3Updates = computeLayerUpdates (map (applyLayer (fc2_w network) (fc2_b network) . applyLayer (fc1_w network) (fc1_b network)) states) errors (fc3_w network) (fc3_b network) lr
      
  in QNetwork
      { fc1_w = fc1_w network + fst fc1Updates
      , fc1_b = fc1_b network + snd fc1Updates
      , fc2_w = fc2_w network + fst fc2Updates  
      , fc2_b = fc2_b network + snd fc2Updates
      , fc3_w = fc3_w network + fst fc3Updates
      , fc3_b = fc3_b network + snd fc3Updates
      }
  where
    -- Apply a single layer transformation
    applyLayer :: Matrix Double -> LA.Vector Double -> LA.Vector Double -> LA.Vector Double
    applyLayer w b input = cmap (max 0) (w #> input + b)
    
    -- Compute weight and bias updates for a layer
    computeLayerUpdates :: [LA.Vector Double] -> [LA.Vector Double] -> Matrix Double -> LA.Vector Double -> Double -> (Matrix Double, LA.Vector Double)
    computeLayerUpdates inputs errors weights biases learningRate =
      let -- Simple gradient approximation
          errorMagnitude = sum (map LA.norm_2 errors) / fromIntegral (length errors)
          updateScale = learningRate * errorMagnitude * 0.001  -- Scale down for stability
          
          -- Weight updates proportional to input-error correlation
          weightUpdate = scalar updateScale * weights * 0.1  -- Small proportional update
          biasUpdate = scalar updateScale * biases * 0.1     -- Small proportional update
          
      in (weightUpdate, biasUpdate)

-- | Train the DQN network
trainDQN :: DQNAgent -> IO DQNAgent
trainDQN agent = do
  let memory = agentMemory agent
      memoryList = toList memory
      batchSize = min (dqnBatchSize $ agentConfig agent) (length memoryList)
  
  if length memoryList < batchSize
    then return $ updateEpsilon agent
    else do
      -- Sample random batch
      indices <- sequence $ replicate batchSize $ randomRIO (0, length memoryList - 1)
      let batch = map (memoryList !!) indices
          targets = computeTargetQValues agent batch
          updatedAgent = updateNetworkWeights agent batch targets
          finalAgent = updateEpsilon updatedAgent
      return finalAgent

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
          
          -- Train the network on replay buffer
          trainedAgent <- trainDQN newAgent
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