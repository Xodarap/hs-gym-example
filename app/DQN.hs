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

-- DQN Network Architecture
data DQNSpec = DQNSpec { inputSize :: Int, hiddenSize :: Int, outputSize :: Int }
  deriving (Show, Eq)


type DQNNet = Network 4 64 64 2
type DQNState = R 4
type DQNOutput = R 2
type Reward = Double
type Trajectory = [(DQNState, Action, Reward)]
newtype DiscountedTrajectory = DT Trajectory

-- Gym.Environment.step 
-- :: Environment -> Action -> IO (Either GymError StepResult)

argmax :: R 2 -> Int
argmax v = 
  let vals = extract v
  in if vals LA.! 0 > vals LA.! 1 then 0 else 1

vectorFromList :: [Double] -> R 4
vectorFromList [a, b, c, d] = vector [a, b, c, d]
vectorFromList _ = vector [0.0, 0.0, 0.0, 0.0]

-- Run network without softmax for Q-values
runNetForQ :: DQNNet -> DQNState -> R 2
runNetForQ net input = runLayerNormal (net ^. nLayer3)
                     . logistic
                     . runLayerNormal (net ^. nLayer2)
                     . logistic
                     . runLayerNormal (net ^. nLayer1)
                     $ input

getAction :: DQNNet -> DQNState -> IO Action
getAction net input = do
  let output = runNetForQ net input
  doRandom <- randomRIO (0.0, 1.0 :: Double)
  if doRandom < 0.1
    then do
      randomAction <- randomRIO (0, 1 :: Int)
      return $ Action $ Number $ fromIntegral randomAction
    else return $ Action $ Number $ fromIntegral $ argmax output

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
calculateQTargets :: DQNNet -> Double -> Trajectory -> [(DQNState, R 2)]
calculateQTargets net gamma trajectory = 
  let (DT discountedTrajectory) = makeDiscountedTrajectory gamma trajectory
      updateQValue :: R 2 -> Int -> Double -> R 2
      updateQValue qvals idx target = 
        let vals = extract qvals
            newVals = [if i == idx then target else vals LA.! i | i <- [0,1]]
        in vector newVals
  in map (\(state, action, discountedReward) -> 
      let currentQ = runNetForQ net state
          actionIdx = actionToInt action
          -- Use discounted reward as target (no need to add gamma * next_q since it's already discounted)
          updatedQ = updateQValue currentQ actionIdx discountedReward
      in (state, updatedQ)) discountedTrajectory

-- Show predicted vs actual Q-values
showQValueComparison :: DQNNet -> Double -> Trajectory -> String
showQValueComparison net gamma trajectory = 
  let (DT discountedTrajectory) = makeDiscountedTrajectory gamma trajectory
      comparisons = zipWith showComparison [(1 :: Int)..] discountedTrajectory
  in unlines $ "Q-Value Predictions vs Targets:" : comparisons
  where
    showComparison stepNum (state, action, discountedReward) = 
      let predictedQ = runNetForQ net state
          actionIdx = actionToInt action
          predictedValue = (extract predictedQ) LA.! actionIdx
          target = discountedReward
      in "Step " ++ show stepNum ++ ": " ++
         "Action=" ++ show actionIdx ++ ", " ++
         "Predicted Q=" ++ show predictedValue ++ ", " ++
         "Target Q=" ++ show target ++ ", " ++
         "Error=" ++ show (abs (predictedValue - target))

-- Train network on trajectory
trainOnTrajectory :: DQNNet -> Double -> Double -> Trajectory -> DQNNet
trainOnTrajectory net learningRate gamma trajectory = 
  let trainingPairs = calculateQTargets net gamma trajectory
  in trainList learningRate trainingPairs net

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
  trainForEpochs newNet learningRate gamma (epochs - 1) envHandle

main :: IO ()
main = MWC.withSystemRandom $ \g -> do
  putStrLn "Initializing neural network..."
  net0 <- MWC.uniformR @(Network 4 64 64 2) (-0.5, 0.5) g
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
              
              putStrLn "\n=== INITIAL TRAJECTORY (BEFORE TRAINING) ==="
              putStrLn $ showTrajectory trajectory
              
              putStrLn "\n=== DISCOUNTED TRAJECTORY ==="
              putStrLn $ showDiscountedTrajectory (makeDiscountedTrajectory 0.99 trajectory)
              
              putStrLn "\n=== Q-VALUE COMPARISON (BEFORE TRAINING) ==="
              putStrLn $ showQValueComparison net0 0.99 trajectory
              
              -- Train network for 100 epochs
              putStrLn "\n=== TRAINING FOR 100 EPOCHS ==="
              trainedNet <- trainForEpochs net0 0.01 0.99 10 envHandle
              putStrLn "Training completed!"
              
              -- Sample trajectory with trained network
              finalTrajectory <- trajectoryFromEnv envHandle trainedNet
              
              putStrLn "\n=== FINAL TRAJECTORY (AFTER TRAINING) ==="
              putStrLn $ showTrajectory finalTrajectory
              
              putStrLn "\n=== Q-VALUE COMPARISON (AFTER TRAINING) ==="
              putStrLn $ showQValueComparison trainedNet 0.99 finalTrajectory
              
              -- Show loss improvement on original trajectory
              let originalLoss = averageLoss (calculateQTargets net0 0.99 trajectory) net0
              let trainedLoss = averageLoss (calculateQTargets trainedNet 0.99 trajectory) trainedNet
              putStrLn $ "\nOriginal loss: " ++ show originalLoss
              putStrLn $ "Trained loss: " ++ show trainedLoss
              putStrLn $ "Loss improvement: " ++ show (originalLoss - trainedLoss)
              
              closeEnv envHandle
              return ()
        
