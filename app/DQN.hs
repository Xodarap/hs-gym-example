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
import Control.Monad (foldM)
import Debug.Trace (trace)

-- DQN Network Architecture
data DQNSpec = DQNSpec { inputSize :: Int, hiddenSize :: Int, outputSize :: Int }
  deriving (Show, Eq)


type DQNNet = Network 4 64 64 1
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

-- Run network to get Q-value prediction (no softmax)
runNetForQ :: DQNNet -> DQNState -> R 1
runNetForQ net input = runLayerNormal (net ^. nLayer3)
                     . logistic
                     . runLayerNormal (net ^. nLayer2)
                     . logistic
                     . runLayerNormal (net ^. nLayer1)
                     $ input

getAction :: DQNNet -> DQNState -> IO Action
getAction net input = do
  let output = runNetForQ net input
  -- Always use random action since we're training a critic, not a policy
  randomActionInt <- randomAction
  return $ Action $ Number $ fromIntegral randomActionInt

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

sampleTrajectory :: DQNNet -> DQNState -> (Action -> IO (DQNState, Reward, Bool)) -> IO Trajectory
sampleTrajectory net input transition = do
  action <- getAction net input
  (nextState, reward, done) <- transition action
  if done
    then return [(input, action, reward)]
    else do
      nextTrajectory <- sampleTrajectory net nextState transition
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
calculateQTargets :: DQNNet -> Double -> Trajectory -> [(DQNState, R 1)]
calculateQTargets net gamma trajectory = 
  let (DT discountedTrajectory) = makeDiscountedTrajectory gamma trajectory
  in map (\(state, action, discountedReward) -> 
      -- The network now predicts a single Q-value for the action taken
      -- The target is the discounted reward
      let target = vector [discountedReward]
      in (state, target)) discountedTrajectory

-- Calculate average MSE loss for trajectory
averageMSELoss :: DQNNet -> Double -> Trajectory -> Double
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

-- Show predicted vs actual Q-values with MSE loss
showQValueComparison :: DQNNet -> Double -> Trajectory -> String
showQValueComparison net gamma trajectory = 
  let (DT discountedTrajectory) = makeDiscountedTrajectory gamma trajectory
      comparisons = zipWith showComparison [(1 :: Int)..] discountedTrajectory
      avgMSELoss = averageMSELoss net gamma trajectory
  in unlines $ ("Q-Value Predictions vs Targets (Average MSE Loss: " ++ show avgMSELoss ++ "):") : comparisons
  where
    showComparison stepNum (state, action, discountedReward) = 
      let predictedQ = runNetForQ net state
          actionIdx = actionToInt action
          predictedValue = (extract predictedQ) LA.! 0  -- Single output
          target = discountedReward
          err = predictedValue - target
          mseErr = err * err
      in "Step " ++ show stepNum ++ ": " ++
         "Action=" ++ show actionIdx ++ ", " ++
         "Predicted Q=" ++ show predictedValue ++ ", " ++
         "Target Q=" ++ show target ++ ", " ++
         "MSE Loss=" ++ show mseErr

-- Check if network output has NaN values
hasNaN :: DQNNet -> Bool
hasNaN net = 
  let testOutput = runNetForQ net (vector [0.0, 0.0, 0.0, 0.0])
  in any isNaN (LA.toList $ extract testOutput)


-- Run network for Q-value prediction with backprop (no softmax)
runNetworkForQ :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o, Reifies s W)
               => BVar s (Network i h1 h2 o)
               -> R i
               -> BVar s (R o)
runNetworkForQ n input = runLayer (n ^^. nLayer3)
                       . logistic
                       . runLayer (n ^^. nLayer2)
                       . logistic
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
trainStepMSE :: Double -> DQNNet -> DQNState -> R 1 -> DQNNet
trainStepMSE learningRate net input target = 
  let predicted = evalBP (\netVar -> runNetworkForQ netVar input) net
      err = evalBP (\netVar -> mseErr input target netVar) net
      retVal = net - realToFrac learningRate * gradBP (mseErr input target) net
      debugInfo = "Predicted: " ++ (show predicted) ++ " Actual: " ++ (show target) ++ " Error: " ++ (show $ err)
  in trace debugInfo retVal

-- Train network on trajectory with MSE loss
trainOnTrajectory :: DQNNet -> Double -> Double -> Trajectory -> DQNNet
trainOnTrajectory net learningRate gamma trajectory = 
  let trainingPairs = calculateQTargets net gamma trajectory
      -- Use MSE loss for Q-value regression
      newNet = foldl (\n (state, target) -> trainStepMSE learningRate n state target) net trainingPairs
  in if hasNaN newNet
     then net  -- Return original network if NaN detected
     else newNet

trajectoryFromEnv :: Environment -> DQNNet -> IO (Trajectory)
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

trainForEpochs :: DQNNet -> Double -> Double -> Int -> Environment -> IO (DQNNet)
trainForEpochs net _ _ 0 _ = return net
trainForEpochs net learningRate gamma epochs envHandle = do
  trajectory <- trajectoryFromEnv envHandle net
  let newNet = trainOnTrajectory net learningRate gamma trajectory
  
  -- Check for NaN and report progress
  if hasNaN newNet
    then do
      putStrLn $ "Warning: NaN detected at epoch " ++ show (101 - epochs) ++ ", stopping training"
      return net
    else do
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

-- ============================================================================
-- UTILITY FUNCTIONS FOR GHCI DEBUGGING
-- ============================================================================

-- | Create a random network for testing (use this in ghci)
-- Example: net <- getRandomNet
getRandomNet :: IO DQNNet
getRandomNet = do
  g <- MWC.createSystemRandom
  MWC.uniformR @(Network 4 64 64 1) (-0.5, 0.5) g

-- | Evaluate critic on a given state
evalCritic :: DQNNet -> DQNState -> Double
evalCritic net state = 
  let output = runNetForQ net state
  in (extract output) LA.! 0

-- | Train network for one step on a single (state, target) pair
trainOneStep :: DQNNet -> DQNState -> Double -> Double -> DQNNet
trainOneStep net state target learningRate =
  let targetVec = vector [target]
  in trainStepMSE learningRate net state targetVec

-- | Compute MSE loss for a single (state, target) pair
computeLoss :: DQNNet -> DQNState -> Double -> Double
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
  net <- getRandomNet
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
  let trainedNet = trainOneStep net sampleState target learningRate
  
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
trainMultipleSteps :: DQNNet -> DQNState -> Double -> Double -> Int -> IO ()
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
        return $ trainOneStep n state target learningRate
  
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
  net <- getRandomNet
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
  -- First run the utilities demo
  demoUtilities
  
  -- Then run the full training
  MWC.withSystemRandom $ \g -> do
    putStrLn "Initializing neural network..."
    net0 <- MWC.uniformR @(Network 4 64 64 1) (-0.5, 0.5) g
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
                -- Sample initial trajectory to show before training
                trajectory <- sampleTrajectory net0 stateVec (makeTransition envHandle)
                
                -- putStrLn "\n=== INITIAL TRAJECTORY (BEFORE TRAINING) ==="
                -- putStrLn $ showTrajectory trajectory
                
                -- putStrLn "\n=== DISCOUNTED TRAJECTORY ==="
                -- putStrLn $ showDiscountedTrajectory (makeDiscountedTrajectory 0.99 trajectory)
                
                -- putStrLn "\n=== Q-VALUE COMPARISON (BEFORE TRAINING) ==="
                -- putStrLn $ showQValueComparison net0 0.99 trajectory
                
                -- Train network for 100 epochs with reduced learning rate
                putStrLn "\n=== TRAINING FOR 100 EPOCHS ==="
                trainedNet <- trainForEpochs net0 0.001 0.99 3 envHandle
                putStrLn "Training completed!"
                
                -- Sample trajectory with trained network
                finalTrajectory <- trajectoryFromEnv envHandle trainedNet
                
                -- putStrLn "\n=== FINAL TRAJECTORY (AFTER TRAINING) ==="
                -- putStrLn $ showTrajectory finalTrajectory
                
                putStrLn "\n=== Q-VALUE COMPARISON (AFTER TRAINING) ==="
                putStrLn $ showQValueComparison trainedNet 0.99 finalTrajectory
                
                -- Show MSE loss improvement on original trajectory
                let originalMSELoss = averageMSELoss net0 0.99 trajectory
                let trainedMSELoss = averageMSELoss trainedNet 0.99 trajectory
                putStrLn $ "\nOriginal MSE loss: " ++ show originalMSELoss
                putStrLn $ "Trained MSE loss: " ++ show trainedMSELoss
                putStrLn $ "Loss improvement: " ++ show (originalMSELoss - trainedMSELoss)
                
                closeEnv envHandle
                return ()
        
