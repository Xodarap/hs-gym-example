{-# LANGUAGE DataKinds                   #-}
{-# LANGUAGE FlexibleContexts            #-}
{-# LANGUAGE OverloadedStrings           #-}
{-# LANGUAGE ScopedTypeVariables         #-}
{-# LANGUAGE TypeApplications            #-}

module Main where

import NeuralNetwork
import System.Random
import Gym.Environment
import Gym.Core
import qualified System.Random.MWC as MWC
import Numeric.LinearAlgebra.Static
import Data.Aeson (Value(Number, Array))
import qualified Data.Vector as V
import qualified Numeric.LinearAlgebra as LA
import Lens.Micro
import Numeric.Backprop
import GHC.TypeLits
import Control.Monad (foldM, replicateM)
import Data.Proxy

-- ============================================================================
-- TYPE DEFINITIONS
-- ============================================================================

type DQNNet = Network 4 64 64 2
type DQNState = R 4
type Reward = Double
type Trajectory = [(DQNState, Action, Reward)]
newtype DiscountedTrajectory = DT Trajectory

-- Experience replay buffer
type Experience = (DQNState, Action, Reward, DQNState, Bool)  -- (state, action, reward, nextState, done)
type ReplayBuffer = [Experience]

-- Add experience to replay buffer with maximum size
addExperience :: Int -> Experience -> ReplayBuffer -> ReplayBuffer
addExperience maxSize exp buffer = 
  let newBuffer = exp : buffer
  in take maxSize newBuffer

-- Sample random batch from replay buffer
sampleBatch :: Int -> ReplayBuffer -> IO [Experience]
sampleBatch batchSize buffer = do
  let bufferSize = length buffer
  if bufferSize < batchSize
    then return buffer
    else do
      indices <- replicateM batchSize (randomRIO (0, bufferSize - 1))
      return $ map (buffer !!) indices

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

randomAction :: IO Action
randomAction = Action . Number . fromIntegral <$> randomRIO (0 :: Int, 1)

vectorFromList :: [Double] -> R 4
vectorFromList [a, b, c, d] = vector [a, b, c, d]
vectorFromList _ = vector [0.0, 0.0, 0.0, 0.0]

-- ============================================================================
-- NETWORK EXECUTION
-- ============================================================================

-- ReLU activation function
reluVec :: (KnownNat n) => R n -> R n
reluVec v = dvmap (\x -> max 0 x) v

-- Softmax activation function
softmaxVec :: (KnownNat n) => R n -> R n
softmaxVec v = 
  let expVec = dvmap exp v
      sumExp = LA.sumElements (extract expVec)
  in dvmap (/ sumExp) expVec

-- Generic network execution with ReLU activation
runDQNNetwork :: DQNNet -> DQNState -> R 2
runDQNNetwork net input = runLayerNormal (net ^. nLayer3)
                        . reluVec
                        . runLayerNormal (net ^. nLayer2)
                        . reluVec
                        . runLayerNormal (net ^. nLayer1)
                        $ input

-- Epsilon-greedy action selection
selectAction :: DQNNet -> DQNState -> Double -> IO Action
selectAction net input epsilon = do
  rand <- randomRIO (0.0, 1.0)
  if rand < epsilon
    then randomAction
    else do
      let qValues = runDQNNetwork net input
      let qList = extract qValues
      let bestAction = if (qList LA.! 0) > (qList LA.! 1) then 0 else 1
      return $ Action $ Number $ fromIntegral bestAction

-- ============================================================================
-- ACTION SELECTION
-- ============================================================================

makeTransition :: Environment -> Action -> IO (DQNState, Reward, Bool)
makeTransition env action = do
    stepResult <- Gym.Environment.step env action
    case stepResult of
      Left err -> do
        putStrLn $ "Step error: " ++ show err
        return (vector [0.0, 0.0, 0.0, 0.0], 0.0, True)
      Right result -> do
        case parseObservation $ stepObservation result of
          Nothing -> return (vector [0.0, 0.0, 0.0, 0.0], stepReward result, stepTerminated result || stepTruncated result)
          Just state -> return (vectorFromList state, stepReward result, stepTerminated result || stepTruncated result)

-- ============================================================================
-- TRAJECTORY SAMPLING
-- ============================================================================

-- Generic trajectory sampling with custom action selection
sampleTrajectoryWith :: (DQNState -> IO Action) -> DQNState -> (Action -> IO (DQNState, Reward, Bool)) -> IO Trajectory
sampleTrajectoryWith actionSelector input transition = do
  action <- actionSelector input
  (nextState, reward, done) <- transition action
  if done
    then return [(input, action, reward)]
    else do
      nextTrajectory <- sampleTrajectoryWith actionSelector nextState transition
      return ((input, action, reward) : nextTrajectory)

-- Trajectory sampling with epsilon-greedy policy
sampleTrajectoryEpsilonGreedy :: DQNNet -> Double -> DQNState -> (Action -> IO (DQNState, Reward, Bool)) -> IO Trajectory
sampleTrajectoryEpsilonGreedy net epsilon = sampleTrajectoryWith (\state -> selectAction net state epsilon)

makeDiscountedTrajectory :: Double -> Trajectory -> DiscountedTrajectory
makeDiscountedTrajectory gamma trajectory = 
  let inner ((ti, to, treward):u:us) = let r@((_, _, ur):_) = inner (u:us) in (ti, to, gamma * ur + treward) : r
      inner [x] = [x]
      inner [] = []
  in DT $ inner trajectory


-- ============================================================================
-- OBSERVATION PARSING AND CONVERSION
-- ============================================================================

parseObservation :: Observation -> Maybe [Double]
parseObservation (Observation (Array arr)) = 
  let values = V.toList arr
      doubles = mapM parseNumber values
  in doubles
  where
    parseNumber (Number n) = Just (realToFrac n)
    parseNumber _ = Nothing
parseObservation _ = Nothing

actionToInt :: Action -> Int
actionToInt (Action (Number n)) = round $ realToFrac n
actionToInt _ = 0

-- ============================================================================
-- TRAJECTORY ANALYSIS
-- ============================================================================

trajectoryLength :: Trajectory -> Int
trajectoryLength = length

averageTrajectoryLength :: [Trajectory] -> Double
averageTrajectoryLength trajectories = 
  if null trajectories 
    then 0.0 
    else fromIntegral (sum (map trajectoryLength trajectories)) / fromIntegral (length trajectories)

sampleMultipleTrajectories :: Int -> Environment -> DQNNet -> Double -> IO [Trajectory]
sampleMultipleTrajectories n envHandle dqnNet epsilon = do
  replicateM n (trajectoryFromEnv dqnNet epsilon envHandle)

-- ============================================================================
-- TARGET CALCULATION
-- ============================================================================

-- Calculate Q-targets using temporal difference learning
calculateQTargets :: DQNNet -> DQNNet -> Double -> Trajectory -> [(DQNState, Action, R 2)]
calculateQTargets qNet targetNet gamma trajectory = 
  let transitions = zip trajectory (tail trajectory ++ [(vector [0,0,0,0], Action (Number 0), 0)])
  in map (\((state, action, reward), (nextState, _, _)) -> 
      let currentQ = runDQNNetwork qNet state
          nextQ = runDQNNetwork targetNet nextState
          maxNextQ = LA.maxElement (extract nextQ)
          targetValue = reward + gamma * maxNextQ
          actionIdx = actionToInt action
          currentExtracted = extract currentQ
          target = if actionIdx == 0
                   then vector [targetValue, currentExtracted LA.! 1]
                   else vector [currentExtracted LA.! 0, targetValue]
      in (state, action, target)) (init transitions)

-- ============================================================================
-- BACKPROPAGATION SUPPORT
-- ============================================================================

-- ReLU activation for backpropagation
reluBackpropVec :: (KnownNat n, Reifies s W) => BVar s (R n) -> BVar s (R n)
reluBackpropVec = liftOp1 . op1 $ \v -> 
  let reluVec = dvmap (\x -> max 0 x) v
      grad g = dvmap (\x -> if x > 0 then 1 else 0) v * g
  in (reluVec, grad)

-- Network execution for backpropagation
runNetworkForQ :: (Reifies s W) => BVar s DQNNet -> R 4 -> BVar s (R 2)
runNetworkForQ n input = runLayer (n ^^. nLayer3)
                       . reluBackpropVec
                       . runLayer (n ^^. nLayer2)
                       . reluBackpropVec
                       . runLayer (n ^^. nLayer1)
                       . constVar
                       $ input

-- MSE loss function for backpropagation
mseErr :: (Reifies s W) => R 4 -> R 2 -> BVar s DQNNet -> BVar s Double
mseErr input target net = 
  let predicted = runNetworkForQ net input
      diff = predicted - constVar target
  in 0.5 * (diff <.>! diff)

-- ============================================================================
-- TRAINING FUNCTIONS
-- ============================================================================

-- MSE-based training step for Q-value regression
trainStepMSE :: Double -> DQNNet -> DQNState -> R 2 -> DQNNet
trainStepMSE learningRate net input target = 
  net - realToFrac learningRate * gradBP (mseErr input target) net

-- Train DQN network using experience replay
trainOnExperiences :: DQNNet -> DQNNet -> Double -> Double -> [Experience] -> DQNNet
trainOnExperiences qNet targetNet learningRate gamma experiences = 
  let trainingPairs = map experienceToTarget experiences
      newNet = foldl (\n (state, target) -> trainStepMSE learningRate n state target) qNet trainingPairs
  in newNet
  where
    experienceToTarget (state, action, reward, nextState, done) =
      let currentQ = runDQNNetwork qNet state
          nextQ = runDQNNetwork targetNet nextState
          maxNextQ = if done then 0.0 else LA.maxElement (extract nextQ)
          targetValue = reward + gamma * maxNextQ
          actionIdx = actionToInt action
          currentExtracted = extract currentQ
          target = if actionIdx == 0
                   then vector [targetValue, currentExtracted LA.! 1]
                   else vector [currentExtracted LA.! 0, targetValue]
      in (state, target)

-- Train DQN network on trajectory with temporal difference learning (legacy)
trainOnTrajectory :: DQNNet -> DQNNet -> Double -> Double -> Trajectory -> DQNNet
trainOnTrajectory qNet targetNet learningRate gamma trajectory = 
  let trainingPairs = calculateQTargets qNet targetNet gamma trajectory
      newNet = foldl (\n (state, _, target) -> trainStepMSE learningRate n state target) qNet trainingPairs
  in newNet

-- ============================================================================
-- ENVIRONMENT INTERACTION
-- ============================================================================

-- Generate experiences for replay buffer
generateExperiences :: DQNNet -> Double -> Int -> Environment -> ReplayBuffer -> IO ReplayBuffer
generateExperiences net epsilon numSteps envHandle buffer = do
  initialState <- Gym.Environment.reset envHandle
  case initialState of
    Left err -> do
      putStrLn $ "Reset error: " ++ show err
      return buffer
    Right obs -> do
      case parseObservation obs of
        Nothing -> do
          putStrLn $ "Parsing returned nothing"
          return buffer
        Just state -> do
          let stateVec = vectorFromList state
          collectExperiences net epsilon numSteps envHandle stateVec buffer

-- Collect experiences step by step
collectExperiences :: DQNNet -> Double -> Int -> Environment -> DQNState -> ReplayBuffer -> IO ReplayBuffer
collectExperiences _ _ 0 _ _ buffer = return buffer
collectExperiences net epsilon steps envHandle currentState buffer = do
  action <- selectAction net currentState epsilon
  (nextState, reward, done) <- makeTransition envHandle action
  let experience = (currentState, action, reward, nextState, done)
  let newBuffer = addExperience 10000 experience buffer  -- Max buffer size of 10k
  
  if done
    then do
      -- Reset environment and continue
      resetResult <- Gym.Environment.reset envHandle
      case resetResult of
        Left _ -> return newBuffer
        Right obs -> case parseObservation obs of
          Nothing -> return newBuffer
          Just state -> collectExperiences net epsilon (steps - 1) envHandle (vectorFromList state) newBuffer
    else collectExperiences net epsilon (steps - 1) envHandle nextState newBuffer

-- Generate trajectory from environment with epsilon-greedy policy  
trajectoryFromEnv :: DQNNet -> Double -> Environment -> IO Trajectory
trajectoryFromEnv net epsilon envHandle = do
  initialState <- Gym.Environment.reset envHandle
  case initialState of
    Left err -> do
      putStrLn $ "Reset error: " ++ show err
      return []
    Right obs -> do
      case parseObservation obs of
        Nothing -> do
          putStrLn $ "Parsing returned nothing"
          return []
        Just state -> do
          let stateVec = vectorFromList state
          sampleTrajectoryEpsilonGreedy net epsilon stateVec (makeTransition envHandle)

-- DQN training with experience replay and target network updates
trainDQN :: DQNNet -> DQNNet -> Double -> Double -> Double -> Int -> Int -> Int -> Environment -> ReplayBuffer -> IO (DQNNet, DQNNet)
trainDQN qNet targetNet learningRate gamma epsilon episodes targetUpdateFreq batchSize envHandle buffer = 
  trainDQNEpisodes qNet targetNet learningRate gamma epsilon episodes targetUpdateFreq batchSize envHandle buffer 0

trainDQNEpisodes :: DQNNet -> DQNNet -> Double -> Double -> Double -> Int -> Int -> Int -> Environment -> ReplayBuffer -> Int -> IO (DQNNet, DQNNet)
trainDQNEpisodes qNet targetNet _ _ _ 0 _ _ _ buffer _ = return (qNet, targetNet)
trainDQNEpisodes qNet targetNet learningRate gamma epsilon episodes targetUpdateFreq batchSize envHandle buffer episodeCount = do
  -- Collect experiences
  newBuffer <- generateExperiences qNet epsilon 100 envHandle buffer
  
  -- Train on batch if buffer has enough experiences
  newQNet <- if length newBuffer >= batchSize
    then do
      batch <- sampleBatch batchSize newBuffer
      return $ trainOnExperiences qNet targetNet learningRate gamma batch
    else return qNet
  
  -- Update target network periodically
  newTargetNet <- if (episodeCount + 1) `mod` targetUpdateFreq == 0
    then do
      putStrLn $ "Updating target network at episode " ++ show (episodeCount + 1)
      return newQNet
    else return targetNet
  
  -- Report progress
  if (episodeCount + 1) `mod` 10 == 0
    then do
      let sampleQ = runDQNNetwork newQNet (vector [0.0, 0.0, 0.0, 0.0])
      trajectory <- trajectoryFromEnv newQNet 0.1 envHandle  -- Low epsilon for evaluation
      let trajectoryLength = length trajectory
      putStrLn $ "Episode " ++ show (episodeCount + 1) ++ 
                  ", Q-values: " ++ show (extract sampleQ) ++
                  ", Trajectory length: " ++ show trajectoryLength ++
                  ", Buffer size: " ++ show (length newBuffer)
    else return ()
  
  -- Decay epsilon
  let newEpsilon = max 0.01 (epsilon * 0.995)
  
  trainDQNEpisodes newQNet newTargetNet learningRate gamma newEpsilon (episodes - 1) targetUpdateFreq batchSize envHandle newBuffer (episodeCount + 1)





-- ============================================================================
-- GLOROT INITIALIZATION
-- ============================================================================

-- | Glorot (Xavier) initialization for a layer
-- For a layer with fanIn inputs and fanOut outputs, 
-- weights are initialized from uniform distribution [-limit, limit]
-- where limit = sqrt(6 / (fanIn + fanOut))
glorotInitLayer :: forall i o. (KnownNat i, KnownNat o) => IO (Layer i o)
glorotInitLayer = do
  let fanIn = fromIntegral $ natVal (Proxy @i)
  let fanOut = fromIntegral $ natVal (Proxy @o)
  let limit = sqrt (6.0 / (fanIn + fanOut)) :: Double
  
  gen <- MWC.createSystemRandom
  -- Generate uniform [0,1] and scale to [-limit, limit]
  uniformWeights <- MWC.uniform gen :: IO (L o i)
  uniformBiases <- MWC.uniform gen :: IO (R o)
  
  let weights = uniformWeights * konst (2 * limit) - konst limit  -- Scale to [-limit, limit]
  let biases = uniformBiases * konst (2 * limit) - konst limit    -- Scale to [-limit, limit]
  
  return $ Layer weights biases

-- | Glorot initialization for the entire DQN network
glorotInitDQNNetwork :: IO DQNNet
glorotInitDQNNetwork = do
  layer1 <- glorotInitLayer @4 @64
  layer2 <- glorotInitLayer @64 @64
  layer3 <- glorotInitLayer @64 @2
  return $ Net layer1 layer2 layer3



main :: IO ()
main = do  
  putStrLn "Initializing DQN with Glorot initialization..."
  qNet0 <- glorotInitDQNNetwork
  targetNet0 <- glorotInitDQNNetwork  -- Initialize target network
  env <- Gym.Environment.makeEnv "CartPole-v1"
  case env of
    Left err -> do
      putStrLn $ "Environment error: " ++ show err
      return ()
    Right envHandle -> do
      putStrLn "\n=== INITIAL PERFORMANCE (BEFORE TRAINING) ==="
      -- Test initial performance with low epsilon
      initialTestTrajectories <- sampleMultipleTrajectories 10 envHandle qNet0 0.1
      let initialAvgLength = averageTrajectoryLength initialTestTrajectories
      let initialLengths = map trajectoryLength initialTestTrajectories
      putStrLn $ "Initial average trajectory length: " ++ show (round initialAvgLength) ++ " steps"
      putStrLn $ "Initial trajectory lengths: " ++ show initialLengths
      
      -- Show initial Q-values
      let sampleState = vector [0.0, 0.0, 0.0, 0.0]
      let initialQ = runDQNNetwork qNet0 sampleState
      putStrLn $ "Initial Q-values: " ++ show (extract initialQ)
      
      -- DQN Training
      putStrLn "\n=== DQN TRAINING ==="
      putStrLn "Training with experience replay, target network updates, and epsilon decay"
      let learningRate = 0.001
      let gamma = 0.99
      let initialEpsilon = 1.0
      let episodes = 500
      let targetUpdateFreq = 50  -- Update target network every 50 episodes
      let batchSize = 32
      let initialBuffer = []
      
      (finalQNet, finalTargetNet) <- trainDQN qNet0 targetNet0 learningRate gamma initialEpsilon episodes targetUpdateFreq batchSize envHandle initialBuffer
      putStrLn "\nDQN training completed!"
      
      -- Evaluate final performance
      putStrLn "\n=== FINAL PERFORMANCE (AFTER DQN TRAINING) ==="
      finalTestTrajectories <- sampleMultipleTrajectories 10 envHandle finalQNet 0.05  -- Very low epsilon for evaluation
      let finalAvgLength = averageTrajectoryLength finalTestTrajectories
      let finalLengths = map trajectoryLength finalTestTrajectories
      putStrLn $ "Final average trajectory length: " ++ show (round finalAvgLength) ++ " steps"
      putStrLn $ "Final trajectory lengths: " ++ show finalLengths
      
      -- Show final Q-values
      let finalQ = runDQNNetwork finalQNet sampleState
      putStrLn $ "Final Q-values: " ++ show (extract finalQ)
      
      -- Calculate improvement
      let performanceImprovement = finalAvgLength - initialAvgLength
      let performanceImprovementPct = if initialAvgLength /= 0 then (performanceImprovement / initialAvgLength) * 100 else 0
      
      putStrLn $ "\n=== TRAINING SUMMARY ==="
      putStrLn $ "Performance improvement: " ++ show (round performanceImprovement) ++ " steps"
      putStrLn $ "Performance improvement: " ++ show (round performanceImprovementPct) ++ "% (" ++ 
                 show (round initialAvgLength) ++ " → " ++ show (round finalAvgLength) ++ " steps)"
      
      -- Show Q-value changes
      let initialQList = extract initialQ
      let finalQList = extract finalQ
      putStrLn $ "Q-value changes:"
      putStrLn $ "  Action 0: " ++ show (initialQList LA.! 0) ++ " → " ++ show (finalQList LA.! 0)
      putStrLn $ "  Action 1: " ++ show (initialQList LA.! 1) ++ " → " ++ show (finalQList LA.! 1)
      
      closeEnv envHandle
      return ()
        
