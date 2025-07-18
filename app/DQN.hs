{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE BangPatterns                #-}
{-# LANGUAGE DataKinds                   #-}
{-# LANGUAGE DeriveGeneric               #-}
{-# LANGUAGE FlexibleContexts            #-}
{-# LANGUAGE GADTs                       #-}
{-# LANGUAGE ScopedTypeVariables         #-}
{-# LANGUAGE TemplateHaskell             #-}
{-# LANGUAGE TypeApplications            #-}
{-# LANGUAGE ViewPatterns                #-}
{-# LANGUAGE AllowAmbiguousTypes         #-}
{-# OPTIONS_GHC -Wno-orphans             #-}

module Main where

import NeuralNetwork
import System.Random
-- import GHC.Generics
import Gym.Environment
import Gym.Core
import qualified System.Random.MWC as MWC
import           Numeric.LinearAlgebra.Static
import Data.Aeson (Value(Number, Array))
import qualified Data.Vector as V
import qualified Numeric.LinearAlgebra as LA
import Lens.Micro
import Numeric.Backprop
import GHC.TypeLits
import Control.Monad (foldM, replicateM)
import Debug.Trace (trace)
import Data.Proxy
import SyntheticData

-- DQN Network Architecture
data DQNSpec = DQNSpec { inputSize :: Int, hiddenSize :: Int, outputSize :: Int }
  deriving (Show, Eq)


type DQNNet o = Network 4 64 64 o
type DQNCritic = DQNNet 1
type DQNActor = DQNNet 2
type DQNState = R 4
type DQNOutput = R 1
type Reward = Double
type Trajectory = [(DQNState, Action, Reward)]
newtype DiscountedTrajectory = DT Trajectory

-- Gym.Environment.step 
-- :: Environment -> Action -> IO (Either GymError StepResult)

-- Since we only have one output, we'll use random action selection
randomAction :: IO Int
randomAction = randomRIO (0, 1)

vectorFromList :: [Double] -> R 4
vectorFromList [a, b, c, d] = vector [a, b, c, d]
vectorFromList _ = vector [0.0, 0.0, 0.0, 0.0]

-- ReLU activation function for static vectors
reluVec :: (KnownNat n) => R n -> R n
reluVec v = dvmap (\x -> max 0 x) v

-- Softmax activation function for static vectors
softmaxVec :: (KnownNat n) => R n -> R n
softmaxVec v = 
  let expVec = dvmap exp v
      sumExp = LA.sumElements (extract expVec)
  in dvmap (/ sumExp) expVec

-- Run network to get Q-value prediction (no softmax, using ReLU)
runNetForQ :: forall o. KnownNat o => DQNNet o -> DQNState -> R o
runNetForQ net input = runLayerNormal (net ^. nLayer3)
                     . reluVec
                     . runLayerNormal (net ^. nLayer2)
                     . reluVec
                     . runLayerNormal (net ^. nLayer1)
                     $ input

runNetForActor :: DQNActor -> DQNState -> R 2
runNetForActor net input = runLayerNormal (net ^. nLayer3)
                     . reluVec
                     . runLayerNormal (net ^. nLayer2)
                     . reluVec
                     . runLayerNormal (net ^. nLayer1)
                     $ input

-- Get action probabilities from actor using softmax
getActionProbs :: DQNActor -> DQNState -> R 2
getActionProbs net input = softmaxVec (runNetForActor net input)

-- Sample action from actor probabilities
sampleActionFromActor :: DQNActor -> DQNState -> IO Action
sampleActionFromActor net input = do
  let probs = getActionProbs net input
  let probList = extract probs
  rand <- randomRIO (0.0, 1.0)
  let action = if rand < (probList LA.! 0) then 0 else 1
  return $ Action $ Number $ fromIntegral action

-- Generic action function for critic (uses random actions for exploration)
getAction :: forall o. KnownNat o => DQNNet o -> DQNState -> IO Action
getAction net input = do
  let output = runNetForQ net input
  -- Always use random action since we're training a critic, not a policy
  randomActionInt <- randomAction
  return $ Action $ Number $ fromIntegral randomActionInt

-- Action function specifically for actor networks (uses policy)
getActionFromActor :: DQNActor -> DQNState -> IO Action
getActionFromActor = sampleActionFromActor

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

sampleTrajectory :: forall o. KnownNat o => DQNNet o -> DQNState -> (Action -> IO (DQNState, Reward, Bool)) -> IO Trajectory
sampleTrajectory net input transition = do
  action <- getAction net input
  (nextState, reward, done) <- transition action
  if done
    then return [(input, action, reward)]
    else do
      nextTrajectory <- sampleTrajectory net nextState transition
      return ((input, action, reward) : nextTrajectory)

-- Sample trajectory using actor network
sampleTrajectoryWithActor :: DQNActor -> DQNState -> (Action -> IO (DQNState, Reward, Bool)) -> IO Trajectory
sampleTrajectoryWithActor net input transition = do
  action <- getActionFromActor net input
  (nextState, reward, done) <- transition action
  if done
    then return [(input, action, reward)]
    else do
      nextTrajectory <- sampleTrajectoryWithActor net nextState transition
      return ((input, action, reward) : nextTrajectory)

-- Sample trajectory for critic using actor's policy for action selection
sampleTrajectoryWithActorPolicy :: forall o. KnownNat o => DQNNet o -> DQNActor -> DQNState -> (Action -> IO (DQNState, Reward, Bool)) -> IO Trajectory
sampleTrajectoryWithActorPolicy criticNet actorNet input transition = do
  action <- getActionFromActor actorNet input  -- Use actor's policy
  (nextState, reward, done) <- transition action
  if done
    then return [(input, action, reward)]
    else do
      nextTrajectory <- sampleTrajectoryWithActorPolicy criticNet actorNet nextState transition
      return ((input, action, reward) : nextTrajectory)

makeDiscountedTrajectory :: Double -> Trajectory -> DiscountedTrajectory
makeDiscountedTrajectory gamma trajectory = 
  let inner ((ti, to, treward):u:us) = let r@((_, _, ur):_) = inner (u:us) in (ti, to, gamma * ur + treward) : r
      inner [x] = [x]
      inner [] = []
  in DT $ inner trajectory

showDiscountedTrajectory :: DiscountedTrajectory -> String
showDiscountedTrajectory (DT t) = unlines $ zipWith showDiscountedStep [(1 :: Int)..] t
  where
    showDiscountedStep stepNum (state, action, discountedReward) = 
      "Step " ++ show stepNum ++ ": " ++
      "State=" ++ show (extract state) ++ ", " ++
      "Action=" ++ show action ++ ", " ++
      "Discounted Reward=" ++ show discountedReward
showTrajectory :: Trajectory -> String
showTrajectory trajectory = unlines $ zipWith showStep [(1 :: Int)..] trajectory
  where
    showStep stepNum (state, action, reward) = 
      "Step " ++ show stepNum ++ ": " ++
      "State=" ++ show (extract state) ++ ", " ++
      "Action=" ++ show (action) ++ ", " ++
      "Reward=" ++ show reward

-- Calculate trajectory length
trajectoryLength :: Trajectory -> Int
trajectoryLength = length

-- Calculate average trajectory length from multiple trajectories
averageTrajectoryLength :: [Trajectory] -> Double
averageTrajectoryLength trajectories = 
  if null trajectories 
    then 0.0 
    else fromIntegral (sum (map trajectoryLength trajectories)) / fromIntegral (length trajectories)

-- Sample multiple trajectories for performance evaluation
sampleMultipleTrajectories :: Int -> Environment -> DQNActor -> IO [Trajectory]
sampleMultipleTrajectories n envHandle actorNet = do
  replicateM n (trajectoryFromEnvWithActor envHandle actorNet)

parseObservation :: Observation -> Maybe [Double]
parseObservation (Observation (Array arr)) = 
  let values = V.toList arr
      doubles = mapM parseNumber values
  in doubles
  where
    parseNumber (Number n) = Just (realToFrac n)
    parseNumber _ = Nothing
parseObservation _ = Nothing

-- Convert action to integer for Q-value indexing
actionToInt :: Action -> Int
actionToInt (Action (Number n)) = round $ realToFrac n
actionToInt _ = 0

-- Calculate Q-value targets from discounted trajectory
calculateQTargets :: forall o. KnownNat o => Double -> Trajectory -> [(DQNState, R o)]
calculateQTargets gamma trajectory = 
  let (DT discountedTrajectory) = makeDiscountedTrajectory gamma trajectory
  in map (\(state, action, discountedReward) -> 
      -- The network now predicts a single Q-value for the action taken
      -- The target is the discounted reward
      let target = vector [discountedReward]
      in (state, target)) discountedTrajectory

calculateActionQTargets :: DQNActor -> Double -> Trajectory -> [(DQNState, R 2)]
calculateActionQTargets net gamma trajectory = 
  let (DT discountedTrajectory) = makeDiscountedTrajectory gamma trajectory
  in map (\(state, action, discountedReward) -> 
      let predicted = runNetForActor net state
          extracted = extract predicted
          target = (vector $ if (actionToInt action) == 0 then [discountedReward, extracted LA.! 1] else [extracted LA.! 0, discountedReward]) :: R 2
      in (state, target)) discountedTrajectory

-- Calculate average MSE loss for trajectory
averageMSELoss :: forall o. KnownNat o => DQNNet o -> Double -> Trajectory -> Double
averageMSELoss net gamma trajectory = 
  let (DT discountedTrajectory) = makeDiscountedTrajectory gamma trajectory
      losses = map calculateLoss discountedTrajectory
  in sum losses / fromIntegral (length losses)
  where
    calculateLoss (state, action, discountedReward) = 
      let predictedQ = runNetForQ net state
          predictedValue = (extract predictedQ) LA.! 0  -- Single output
          target = discountedReward
          err = predictedValue - target
      in err * err  -- MSE = (predicted - target)^2

-- Calculate average MSE loss for actor trajectory
averageActorMSELoss :: DQNActor -> Double -> Trajectory -> Double
averageActorMSELoss net gamma trajectory = 
  let trainingPairs = calculateActionQTargets net gamma trajectory
      losses = map calculateLoss trainingPairs
  in sum losses / fromIntegral (length losses)
  where
    calculateLoss (state, target) = 
      let predicted = runNetForActor net state
          diff = extract predicted - extract target
          squaredDiff = LA.toList diff
      in sum (map (^2) squaredDiff) / 2.0  -- MSE = 0.5 * ||predicted - target||^2

-- Show predicted vs actual Q-values with MSE loss
showQValueComparison :: forall o. KnownNat o =>  DQNNet o -> Double -> Trajectory -> String
showQValueComparison net gamma trajectory = 
  let (DT discountedTrajectory) = makeDiscountedTrajectory gamma trajectory
      comparisons = zipWith showComparison [(1 :: Int)..] discountedTrajectory
      avgMSELoss = averageMSELoss net gamma trajectory
  in unlines $ ("Q-Value Predictions vs Targets (Average MSE Loss: " ++ show avgMSELoss ++ "):") : comparisons
  where
    showComparison stepNum (state, action, discountedReward) = 
      let predictedQ = runNetForQ net state
          actionIdx = actionToInt action
          predictedValue = (extract predictedQ) LA.! 0  -- Single output -- Todo
          target = discountedReward
          err = predictedValue - target
          mseErr = err * err
      in "Step " ++ show stepNum ++ ": " ++
         "Action=" ++ show actionIdx ++ ", " ++
         "Predicted Q=" ++ show predictedValue ++ ", " ++
         "Target Q=" ++ show target ++ ", " ++
         "MSE Loss=" ++ show mseErr


-- ReLU activation function for backpropagation
reluBackpropVec :: (KnownNat n, Reifies s W) => BVar s (R n) -> BVar s (R n)
reluBackpropVec = liftOp1 . op1 $ \v -> 
  let reluVec = dvmap (\x -> max 0 x) v
      grad g = dvmap (\x -> if x > 0 then 1 else 0) v * g
  in (reluVec, grad)

-- Run network for Q-value prediction with backprop (no softmax, using ReLU)
runNetworkForQ :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o, Reifies s W)
               => BVar s (Network i h1 h2 o)
               -> R i
               -> BVar s (R o)
runNetworkForQ n input = runLayer (n ^^. nLayer3)
                       . reluBackpropVec
                       . runLayer (n ^^. nLayer2)
                       . reluBackpropVec
                       . runLayer (n ^^. nLayer1)
                       . constVar
                       $ input

-- MSE loss for Q-value regression
mseErr :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o, Reifies s W)
       => R i -> R o -> BVar s (Network i h1 h2 o) -> BVar s Double
mseErr input target net = 
  let predicted = runNetworkForQ net input
      diff = predicted - constVar target
  in 0.5 * (diff <.>! diff)  -- MSE = 0.5 * ||predicted - target||^2

-- MSE-based training step for Q-value regression
trainStepMSE :: forall o. KnownNat o => Double -> DQNNet o -> DQNState -> R o -> DQNNet o
trainStepMSE learningRate net input target = 
  let predicted = evalBP (\netVar -> runNetworkForQ netVar input) net
      err = evalBP (\netVar -> mseErr input target netVar) net
      retVal = net - realToFrac learningRate * gradBP (mseErr input target) net
      debugInfo = "Predicted: " ++ (show predicted) ++ " Actual: " ++ (show target) ++ " Error: " ++ (show $ err)
  in retVal --trace debugInfo retVal

trainStepActorMSEAct :: Double -> DQNActor -> DQNState -> Double -> Action -> DQNActor
trainStepActorMSEAct learningRate net input target action =
  let predicted = evalBP (\netVar -> runNetworkForQ netVar input) net
      extracted = extract predicted
      realTarget = (vector $ if (actionToInt action) == 0 then [target, extracted LA.! 1] else [extracted LA.! 0, target]) :: R 2
  in trainStepActorMSE learningRate net input realTarget

trainStepActorMSE :: Double -> DQNActor -> DQNState -> R 2 -> DQNActor
trainStepActorMSE learningRate net input target = 
  let predicted = evalBP (\netVar -> runNetworkForQ netVar input) net
      err = evalBP (\netVar -> mseErr input target netVar) net
      retVal = net - realToFrac learningRate * gradBP (mseErr input target) net
      debugInfo = "Predicted: " ++ (show predicted) ++ " Actual: " ++ (show target) ++ " Error: " ++ (show $ err)
  in retVal --trace debugInfo retVal

-- Train network on trajectory with MSE loss
trainOnTrajectory :: forall o. KnownNat o => DQNNet o -> Double -> Double -> Trajectory -> DQNNet o
trainOnTrajectory net learningRate gamma trajectory = 
  let trainingPairs = calculateQTargets gamma trajectory
      -- Use MSE loss for Q-value regression
      newNet = foldl (\n (state, target) -> trainStepMSE learningRate n state target) net trainingPairs
  in newNet

trainActorOnTrajectory :: DQNActor -> Double -> Double -> Trajectory -> DQNActor
trainActorOnTrajectory net learningRate gamma trajectory = 
  let trainingPairs = calculateActionQTargets net gamma trajectory
      newNet = foldl (\n (state, target) -> trainStepActorMSE learningRate n state target) net trainingPairs
  in newNet

trajectoryFromEnv :: forall o. KnownNat o => Environment -> DQNNet o -> IO (Trajectory)
trajectoryFromEnv envHandle net = do
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
          sampleTrajectory net stateVec (makeTransition envHandle)

-- Trajectory from environment using actor
trajectoryFromEnvWithActor :: Environment -> DQNActor -> IO (Trajectory)
trajectoryFromEnvWithActor envHandle net = do
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
          sampleTrajectoryWithActor net stateVec (makeTransition envHandle)

-- Trajectory from environment for critic using actor's policy
trajectoryFromEnvWithActorPolicy :: forall o. KnownNat o => Environment -> DQNNet o -> DQNActor -> IO (Trajectory)
trajectoryFromEnvWithActorPolicy envHandle criticNet actorNet = do
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
          sampleTrajectoryWithActorPolicy criticNet actorNet stateVec (makeTransition envHandle)

trainForEpochs :: forall o. KnownNat o => DQNNet o -> Double -> Double -> Int -> Environment -> IO (DQNNet o)
trainForEpochs net _ _ 0 _ = return net
trainForEpochs net learningRate gamma epochs envHandle = do
  trajectory <- trajectoryFromEnv envHandle net
  let newNet = trainOnTrajectory net learningRate gamma trajectory
  
  -- Report progress every 10 epochs
  if (101 - epochs) `mod` 10 == 0
    then do
      let sampleQ = runNetForQ newNet (vector [0.0, 0.0, 0.0, 0.0])
      let mseLoss = averageMSELoss newNet gamma trajectory
      putStrLn $ "Epoch " ++ show (101 - epochs) ++ 
                  ", Sample Q-value: " ++ show (extract sampleQ) ++
                  ", MSE Loss: " ++ show mseLoss
    else return ()
  
  trainForEpochs newNet learningRate gamma (epochs - 1) envHandle

-- Training function for actor
trainActorForEpochs :: DQNActor -> Double -> Double -> Int -> Environment -> IO DQNActor
trainActorForEpochs net _ _ 0 _ = return net
trainActorForEpochs net learningRate gamma epochs envHandle = do
  trajectory <- trajectoryFromEnvWithActor envHandle net
  let newNet = trainActorOnTrajectory net learningRate gamma trajectory
  
  -- Report progress every 10 epochs
  if (101 - epochs) `mod` 10 == 0
    then do
      let sampleProbs = getActionProbs newNet (vector [0.0, 0.0, 0.0, 0.0])
      let actorLoss = averageActorMSELoss newNet gamma trajectory
      putStrLn $ "Actor Epoch " ++ show (101 - epochs) ++ 
                  ", Sample Action Probs: " ++ show (extract sampleProbs) ++
                  ", Actor MSE Loss: " ++ show actorLoss
    else return ()
  
  trainActorForEpochs newNet learningRate gamma (epochs - 1) envHandle

-- Combined training function for both critic and actor
trainBothNetworks :: DQNCritic -> DQNActor -> Double -> Double -> Int -> Environment -> IO (DQNCritic, DQNActor)
trainBothNetworks criticNet actorNet _ _ 0 _ = return (criticNet, actorNet)
trainBothNetworks criticNet actorNet learningRate gamma epochs envHandle = do
  -- Train critic with random exploration
  criticTrajectory <- trajectoryFromEnv envHandle criticNet
  let newCriticNet = trainOnTrajectory criticNet learningRate gamma criticTrajectory
  
  -- Train actor with its own policy
  actorTrajectory <- trajectoryFromEnvWithActor envHandle actorNet
  let newActorNet = trainActorOnTrajectory actorNet learningRate gamma actorTrajectory
  
  -- Report progress every 10 epochs
  if (epochs `mod` 10 == 0) || (epochs <= 10)
    then do
      let sampleQ = runNetForQ newCriticNet (vector [0.0, 0.0, 0.0, 0.0])
      let criticLoss = averageMSELoss newCriticNet gamma criticTrajectory
      let sampleProbs = getActionProbs newActorNet (vector [0.0, 0.0, 0.0, 0.0])
      let actorLoss = averageActorMSELoss newActorNet gamma actorTrajectory
      putStrLn $ "Epoch " ++ show epochs ++ 
                  " - Critic Q: " ++ show (extract sampleQ) ++
                  ", Critic Loss: " ++ show criticLoss ++
                  " | Actor Probs: " ++ show (extract sampleProbs) ++
                  ", Actor Loss: " ++ show actorLoss
    else return ()
  
  trainBothNetworks newCriticNet newActorNet learningRate gamma (epochs - 1) envHandle

-- Actor-Critic training function: train actor for 100 epochs, then use it for critic training
trainActorCritic :: DQNCritic -> DQNActor -> Double -> Double -> Int -> Environment -> IO (DQNCritic, DQNActor)
trainActorCritic criticNet actorNet _ _ 0 _ = return (criticNet, actorNet)
trainActorCritic criticNet actorNet learningRate gamma cycles envHandle = do
  putStrLn $ "\n=== CYCLE " ++ show (cycles) ++ " ==="
  
  -- Phase 1: Train actor for 100 epochs
  putStrLn "Phase 1: Training Actor for 100 epochs..."
  trainedActor <- trainActorForEpochs actorNet learningRate gamma 100 envHandle
  
  -- Evaluate actor performance with multiple trajectories
  putStrLn "Evaluating Actor performance..."
  testTrajectories <- sampleMultipleTrajectories 10 envHandle trainedActor
  let avgLength = averageTrajectoryLength testTrajectories
  let lengths = map trajectoryLength testTrajectories
  putStrLn $ "  Average trajectory length: " ++ show (round avgLength) ++ " steps"
  putStrLn $ "  Trajectory lengths: " ++ show lengths
  
  -- Phase 2: Train critic using the trained actor's policy for 100 epochs
  putStrLn "Phase 2: Training Critic using Actor's policy for 100 epochs..."
  trainedCritic <- trainCriticWithActorPolicy criticNet trainedActor learningRate gamma 100 envHandle
  
  -- Show cycle results
  let sampleQ = runNetForQ trainedCritic (vector [0.0, 0.0, 0.0, 0.0])
  let sampleProbs = getActionProbs trainedActor (vector [0.0, 0.0, 0.0, 0.0])
  putStrLn $ "Cycle " ++ show cycles ++ " completed:"
  putStrLn $ "  Critic Q-prediction: " ++ show (extract sampleQ)
  putStrLn $ "  Actor action probs: " ++ show (extract sampleProbs)
  putStrLn $ "  Performance: " ++ show (round avgLength) ++ " steps average"
  
  trainActorCritic trainedCritic trainedActor learningRate gamma (cycles - 1) envHandle

-- Train critic using actor's policy for action selection
trainCriticWithActorPolicy :: DQNCritic -> DQNActor -> Double -> Double -> Int -> Environment -> IO DQNCritic
trainCriticWithActorPolicy criticNet _ _ _ 0 _ = return criticNet
trainCriticWithActorPolicy criticNet actorNet learningRate gamma epochs envHandle = do
  trajectory <- trajectoryFromEnvWithActorPolicy envHandle criticNet actorNet
  let newCriticNet = trainOnTrajectory criticNet learningRate gamma trajectory
  
  -- Report progress every 25 epochs
  if epochs `mod` 25 == 0
    then do
      let sampleQ = runNetForQ newCriticNet (vector [0.0, 0.0, 0.0, 0.0])
      let criticLoss = averageMSELoss newCriticNet gamma trajectory
      putStrLn $ "  Critic Epoch " ++ show (101 - epochs) ++ 
                  " - Q: " ++ show (extract sampleQ) ++
                  ", Loss: " ++ show criticLoss
    else return ()
  
  trainCriticWithActorPolicy newCriticNet actorNet learningRate gamma (epochs - 1) envHandle

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

-- | Glorot initialization for the entire network
glorotInitNetwork :: forall o. KnownNat o => IO (DQNNet o)
glorotInitNetwork = do
  layer1 <- glorotInitLayer @4 @64
  layer2 <- glorotInitLayer @64 @64
  layer3 <- glorotInitLayer @64 @o
  return $ Net layer1 layer2 layer3


-- ============================================================================
-- UTILITY FUNCTIONS FOR GHCI DEBUGGING
-- ============================================================================

-- | Create a random network for testing with Glorot initialization (use this in ghci)
-- Example: net <- getRandomNet
getRandomNet :: forall o. KnownNat o => IO (DQNNet o)
getRandomNet = glorotInitNetwork

-- | Evaluate critic on a given state
evalCritic :: forall o. KnownNat o => (DQNNet o) -> DQNState -> Double
evalCritic net state = 
  let output = runNetForQ net state
  in (extract output) LA.! 0

-- | Train network for one step on a single (state, target) pair
-- trainOneStep :: DQNNet -> DQNState -> Double -> Double -> DQNNet
-- trainOneStep net state target learningRate =
--   let targetVec = vector [target]
--   in trainStepMSE learningRate net state targetVec

-- trainActorOneStep :: DQNActor -> DQNState -> Double -> Double -> Int -> DQNActor
-- trainActorOneStep net state target learningRate action =
--   let output = runNetForActor net state
--       realTarget = if action == 1 then [(extract output) LA.! 0, target] else [target, (extract output) LA.! 1]
--   in trainStepActorMSE learningRate net state (LA.fromList realTarget)

-- | Compute MSE loss for a single (state, target) pair
computeLoss :: forall o. KnownNat o => DQNNet o -> DQNState -> Double -> Double
computeLoss net state target =
  let predicted = evalCritic net state
      err = predicted - target
  in err * err

-- | Create a sample state for testing
sampleState :: DQNState
sampleState = vector [0.1, 0.2, -0.1, 0.3]

-- | Create another sample state
sampleState2 :: DQNState
sampleState2 = vector [-0.2, 0.4, 0.1, -0.3]

-- | Demo function showing how to use the utilities
demoUtilities :: IO ()
demoUtilities = do
  putStrLn "=== DQN Utilities Demo ==="
  
  -- Get a random network
  net <- getRandomNet :: IO (DQNCritic)
  putStrLn $ "Initial network created"
  
  -- Evaluate critic on sample state
  let initialOutput = evalCritic net sampleState
  putStrLn $ "Initial critic output: " ++ show initialOutput
  
  -- Compute loss with target = 5.0
  let target = 5.0
  let initialLoss = computeLoss net sampleState target
  putStrLn $ "Initial loss (target=" ++ show target ++ "): " ++ show initialLoss
  
  -- Train for one step
  let learningRate = 0.01
  let trainedNet = trainStepMSE learningRate net sampleState (vector [target])
  
  -- Check new output and loss
  let newOutput = evalCritic trainedNet sampleState
  let newLoss = computeLoss trainedNet sampleState target
  
  putStrLn $ "After training - critic output: " ++ show newOutput
  putStrLn $ "After training - loss: " ++ show newLoss
  putStrLn $ "Loss improvement: " ++ show (initialLoss - newLoss)
  
  -- Test on a different state
  let otherOutput = evalCritic trainedNet sampleState2
  putStrLn $ "Output on different state: " ++ show otherOutput

-- | Train network on multiple steps and show loss trajectory
trainMultipleSteps :: forall o. KnownNat o => DQNNet o -> DQNState -> Double -> Double -> Int -> IO ()
trainMultipleSteps net state target learningRate steps = do
  putStrLn $ "=== Training for " ++ show steps ++ " steps ==="
  putStrLn $ "State: " ++ show (extract state)
  putStrLn $ "Target: " ++ show target
  putStrLn $ "Learning rate: " ++ show learningRate
  putStrLn ""
  
  let trainAndPrint n step = do
        let output = evalCritic n state
        let loss = computeLoss n state target
        putStrLn $ "Step " ++ show step ++ ": output=" ++ show output ++ ", loss=" ++ show loss
        return $ trainStepMSE learningRate n state (vector [target]) 
  
  -- Initial state
  let initialOutput = evalCritic net state
  let initialLoss = computeLoss net state target
  putStrLn $ "Step 0: output=" ++ show initialOutput ++ ", loss=" ++ show initialLoss
  
  -- Train step by step
  finalNet <- foldM trainAndPrint net [1..steps]
  
  let finalOutput = evalCritic finalNet state
  let finalLoss = computeLoss finalNet state target
  putStrLn ""
  putStrLn $ "Final: output=" ++ show finalOutput ++ ", loss=" ++ show finalLoss
  putStrLn $ "Total loss improvement: " ++ show (initialLoss - finalLoss)

-- | Quick test function for different targets
testDifferentTargets :: IO ()
testDifferentTargets = do
  net <- getRandomNet :: IO DQNCritic
  let targets = [0.0, 1.0, 5.0, 10.0, -2.0]
  putStrLn "=== Testing different targets ==="
  putStrLn $ "State: " ++ show (extract sampleState)
  putStrLn ""
  
  mapM_ (\target -> do
          let output = evalCritic net sampleState
          let loss = computeLoss net sampleState target
          putStrLn $ "Target " ++ show target ++ ": output=" ++ show output ++ ", loss=" ++ show loss
        ) targets

main :: IO ()
main = do  
  putStrLn "Initializing neural networks with Glorot initialization..."
  criticNet0 <- glorotInitNetwork :: IO DQNCritic
  actorNet0 <- glorotInitNetwork :: IO DQNActor
  env <- Gym.Environment.makeEnv "CartPole-v1"
  case env of
    Left err -> do
      putStrLn $ "Environment error: " ++ show err
      return ()
    Right envHandle -> do
      initialState <- Gym.Environment.reset envHandle
      case initialState of
        Left err -> do
          putStrLn $ "Reset error: " ++ show err
          return ()
        Right obs -> do
          case parseObservation obs of
            Nothing -> do
              putStrLn $ "Parsing returned nothing"
              return ()
            Just state -> do
              let stateVec = vectorFromList state
              
              -- Sample initial trajectories to show before training
              criticTrajectory <- sampleTrajectory criticNet0 stateVec (makeTransition envHandle)
              actorTrajectory <- trajectoryFromEnvWithActor envHandle actorNet0
              
              -- Evaluate initial actor performance
              putStrLn "\n=== INITIAL PERFORMANCE (BEFORE TRAINING) ==="
              initialTestTrajectories <- sampleMultipleTrajectories 10 envHandle actorNet0
              let initialAvgLength = averageTrajectoryLength initialTestTrajectories
              let initialLengths = map trajectoryLength initialTestTrajectories
              putStrLn $ "Initial average trajectory length: " ++ show (round initialAvgLength) ++ " steps"
              putStrLn $ "Initial trajectory lengths: " ++ show initialLengths
              
              putStrLn "\n=== INITIAL LOSSES (BEFORE TRAINING) ==="
              let initialCriticLoss = averageMSELoss criticNet0 0.99 criticTrajectory
              let initialActorLoss = averageActorMSELoss actorNet0 0.99 actorTrajectory
              putStrLn $ "Initial Critic MSE Loss: " ++ show initialCriticLoss
              putStrLn $ "Initial Actor MSE Loss: " ++ show initialActorLoss
              
              -- Show initial predictions
              let sampleState = vector [0.0, 0.0, 0.0, 0.0]
              let initialCriticQ = runNetForQ criticNet0 sampleState
              let initialActorProbs = getActionProbs actorNet0 sampleState
              putStrLn $ "Initial Critic Q-value: " ++ show (extract initialCriticQ)
              putStrLn $ "Initial Actor action probs: " ++ show (extract initialActorProbs)
              
              -- Actor-Critic training: 3 cycles of (train actor 100 epochs, then use it for critic 100 epochs)
              putStrLn "\n=== ACTOR-CRITIC TRAINING (3 CYCLES) ==="
              putStrLn "Each cycle: Train Actor (100 epochs) → Train Critic using Actor's policy (100 epochs)"
              (finalCritic, finalActor) <- trainActorCritic criticNet0 actorNet0 0.001 0.99 3 envHandle
              putStrLn "\nActor-Critic training completed!"
              
              -- Evaluate final actor performance
              putStrLn "\n=== FINAL PERFORMANCE (AFTER ACTOR-CRITIC TRAINING) ==="
              finalTestTrajectories <- sampleMultipleTrajectories 10 envHandle finalActor
              let finalAvgLength = averageTrajectoryLength finalTestTrajectories
              let finalLengths = map trajectoryLength finalTestTrajectories
              putStrLn $ "Final average trajectory length: " ++ show (round finalAvgLength) ++ " steps"
              putStrLn $ "Final trajectory lengths: " ++ show finalLengths
              
              -- Sample final trajectories using the trained networks
              finalCriticTrajectory <- trajectoryFromEnvWithActorPolicy envHandle finalCritic finalActor
              finalActorTrajectory <- trajectoryFromEnvWithActor envHandle finalActor
              
              putStrLn "\n=== FINAL COMPARISON (AFTER ACTOR-CRITIC TRAINING) ==="
              let finalCriticLoss = averageMSELoss finalCritic 0.99 finalCriticTrajectory
              let finalActorLoss = averageActorMSELoss finalActor 0.99 finalActorTrajectory
              let criticImprovement = initialCriticLoss - finalCriticLoss
              let actorImprovement = initialActorLoss - finalActorLoss
              let performanceImprovement = finalAvgLength - initialAvgLength
              
              putStrLn $ "Final Critic MSE Loss: " ++ show finalCriticLoss
              putStrLn $ "Critic Loss Improvement: " ++ show criticImprovement
              putStrLn $ "Final Actor MSE Loss: " ++ show finalActorLoss
              putStrLn $ "Actor Loss Improvement: " ++ show actorImprovement
              putStrLn $ "Performance improvement: " ++ show (round performanceImprovement) ++ " steps"
              
              -- Show final predictions
              let finalCriticQ = runNetForQ finalCritic sampleState
              let finalActorProbs = getActionProbs finalActor sampleState
              putStrLn $ "\nFinal Critic Q-value prediction: " ++ show (extract finalCriticQ)
              putStrLn $ "Final Actor action probabilities: " ++ show (extract finalActorProbs)
              
              -- Show improvement percentages
              let criticImprovementPct = if initialCriticLoss /= 0 then (criticImprovement / initialCriticLoss) * 100 else 0
              let actorImprovementPct = if initialActorLoss /= 0 then (actorImprovement / initialActorLoss) * 100 else 0
              let performanceImprovementPct = if initialAvgLength /= 0 then (performanceImprovement / initialAvgLength) * 100 else 0
              putStrLn $ "\nImprovement Summary:"
              putStrLn $ "Critic: " ++ show (round criticImprovementPct) ++ "% improvement"
              putStrLn $ "Actor: " ++ show (round actorImprovementPct) ++ "% improvement"
              putStrLn $ "Performance: " ++ show (round performanceImprovementPct) ++ "% improvement (" ++ 
                         show (round initialAvgLength) ++ " → " ++ show (round finalAvgLength) ++ " steps)"
              
              closeEnv envHandle
              return ()
        
