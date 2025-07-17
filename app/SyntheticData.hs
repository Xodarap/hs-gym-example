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

module SyntheticData (
    -- * Synthetic functions
    generateSumSquares,
    generateLinearCombo,
    generateQuadratic,
    
    -- * Data generation
    generateTrainingData,
    
    -- * Test patterns
    testPatterns,
    linearComboPatterns,
    
    -- * Training and testing
    trainOnSyntheticData,
    testOnPatterns,
    testFunction,
    testSyntheticLearning
) where

import NeuralNetwork
import System.Random
import qualified System.Random.MWC as MWC
import           Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra as LA
import Control.Monad (foldM, replicateM)

-- Import types from DQN module
type DQNNet = Network 4 64 64 1
type DQNState = R 4

-- ============================================================================
-- SYNTHETIC DATASET FOR TESTING
-- ============================================================================

-- | Simple synthetic dataset where target = sum of squares of inputs
-- f(x1, x2, x3, x4) = x1^2 + x2^2 + x3^2 + x4^2
generateSumSquares :: DQNState -> Double
generateSumSquares state = 
  let vals = LA.toList $ extract state
  in sum $ map (\x -> x * x) vals

-- | Linear combination: f(x1, x2, x3, x4) = 2*x1 + 3*x2 - x3 + 0.5*x4
generateLinearCombo :: DQNState -> Double
generateLinearCombo state =
  let [x1, x2, x3, x4] = LA.toList $ extract state
  in 2*x1 + 3*x2 - x3 + 0.5*x4

-- | Quadratic with cross terms: f(x1, x2, x3, x4) = x1*x2 + x3*x4 + x1^2
generateQuadratic :: DQNState -> Double
generateQuadratic state =
  let [x1, x2, x3, x4] = LA.toList $ extract state
  in x1*x2 + x3*x4 + x1*x1

-- | Generate random training data
generateTrainingData :: Int -> (DQNState -> Double) -> IO [(DQNState, Double)]
generateTrainingData numSamples targetFunc = do
  states <- replicateM numSamples $ do
    x1 <- randomRIO (-2.0, 2.0)
    x2 <- randomRIO (-2.0, 2.0) 
    x3 <- randomRIO (-2.0, 2.0)
    x4 <- randomRIO (-2.0, 2.0)
    return $ vector [x1, x2, x3, x4]
  let targets = map targetFunc states
  return $ zip states targets

-- | Test specific patterns for sum of squares function
testPatterns :: [(DQNState, Double, String)]
testPatterns = 
  [ (vector [1.0, 0.0, 0.0, 0.0], 1.0, "unit_x1")
  , (vector [0.0, 1.0, 0.0, 0.0], 1.0, "unit_x2") 
  , (vector [0.0, 0.0, 1.0, 0.0], 1.0, "unit_x3")
  , (vector [0.0, 0.0, 0.0, 1.0], 1.0, "unit_x4")
  , (vector [1.0, 1.0, 1.0, 1.0], 4.0, "all_ones")
  , (vector [2.0, 0.0, 0.0, 0.0], 4.0, "double_x1")
  , (vector [0.0, 0.0, 0.0, 0.0], 0.0, "zeros")
  , (vector [-1.0, 1.0, -1.0, 1.0], 4.0, "alternating")
  ]

-- | Test patterns for linear combination function
linearComboPatterns :: [(DQNState, Double, String)]
linearComboPatterns = 
  [ (vector [1.0, 0.0, 0.0, 0.0], 2.0, "unit_x1_*2")
  , (vector [0.0, 1.0, 0.0, 0.0], 3.0, "unit_x2_*3") 
  , (vector [0.0, 0.0, 1.0, 0.0], -1.0, "unit_x3_*(-1)")
  , (vector [0.0, 0.0, 0.0, 1.0], 0.5, "unit_x4_*0.5")
  , (vector [1.0, 1.0, 1.0, 1.0], 4.5, "all_ones")
  , (vector [0.0, 0.0, 0.0, 0.0], 0.0, "zeros")
  , (vector [2.0, 1.0, -1.0, 2.0], 9.0, "mixed")
  ]

-- ============================================================================
-- TRAINING AND TESTING FUNCTIONS
-- ============================================================================

-- | Train network on synthetic dataset
trainOnSyntheticData :: (DQNNet -> DQNState -> Double -> Double -> DQNNet) -- trainOneStep
                    -> (DQNNet -> DQNState -> Double -> Double) -- computeLoss  
                    -> DQNNet -> Double -> Int -> [(DQNState, Double)] -> IO DQNNet
trainOnSyntheticData trainOneStep computeLoss net learningRate epochs trainingData = do
  putStrLn $ "Training on " ++ show (length trainingData) ++ " samples for " ++ show epochs ++ " epochs"
  
  let trainEpoch currentNet epochNum = do
        -- Shuffle and train on all data
        let trainedNet = foldl (\n (state, target) -> 
              trainOneStep n state target learningRate) currentNet trainingData
        
        -- Report progress every 50 epochs
        if epochNum `mod` 50 == 0 then do
          let avgLoss = sum [computeLoss trainedNet state target | (state, target) <- trainingData] 
                       / fromIntegral (length trainingData)
          putStrLn $ "Epoch " ++ show epochNum ++ ", Average Loss: " ++ show avgLoss
        else return ()
        
        return trainedNet
  
  foldM trainEpoch net [1..epochs]

-- | Test network on specific patterns
testOnPatterns :: (DQNNet -> DQNState -> Double) -- evalCritic
              -> DQNNet -> [(DQNState, Double, String)] -> IO ()
testOnPatterns evalCritic net patterns = do
  putStrLn "\n=== Testing on Specific Patterns ==="
  mapM_ testPattern patterns
  where
    testPattern (state, expected, name) = do
      let predicted = evalCritic net state
      let error = abs (predicted - expected)
      let relativeError = if expected /= 0 then error / abs expected * 100 else error * 100
      putStrLn $ name ++ ": expected=" ++ show expected ++ 
                ", predicted=" ++ show predicted ++ 
                ", error=" ++ show error ++
                ", rel_error=" ++ show relativeError ++ "%"

-- | Test a specific function
testFunction :: (IO DQNNet) -- getRandomNet
            -> (DQNNet -> DQNState -> Double) -- evalCritic
            -> (DQNNet -> DQNState -> Double -> Double -> DQNNet) -- trainOneStep
            -> (DQNNet -> DQNState -> Double -> Double) -- computeLoss
            -> String -> (DQNState -> Double) -> [(DQNState, Double, String)] -> IO ()
testFunction getRandomNet evalCritic trainOneStep computeLoss funcName targetFunc patterns = do
  putStrLn $ "\n--- Testing " ++ funcName ++ " ---"
  
  -- Generate training data
  trainingData <- generateTrainingData 1000 targetFunc
  testData <- generateTrainingData 100 targetFunc
  
  -- Initialize network
  net0 <- getRandomNet
  
  -- Test before training
  putStrLn "\nBefore training:"
  testOnPatterns evalCritic net0 patterns
  
  -- Train network
  trainedNet <- trainOnSyntheticData trainOneStep computeLoss net0 0.01 200 trainingData
  
  -- Test after training
  putStrLn "\nAfter training:"
  testOnPatterns evalCritic trainedNet patterns
  
  -- Test on unseen data
  putStrLn "\n=== Testing on Unseen Data ==="
  let avgTestLoss = sum [computeLoss trainedNet state target | (state, target) <- testData] 
                   / fromIntegral (length testData)
  putStrLn $ "Average test loss: " ++ show avgTestLoss
  
  -- Show some test examples
  putStrLn "\nSample predictions on test data:"
  mapM_ (\(state, expected) -> do
          let predicted = evalCritic trainedNet state
          let vals = LA.toList $ extract state
          putStrLn $ "Input: " ++ show vals ++ " → Expected: " ++ show expected ++ 
                     ", Predicted: " ++ show predicted ++ 
                     ", Error: " ++ show (abs (predicted - expected))
        ) (take 3 testData)

-- | Comprehensive test of synthetic learning
testSyntheticLearning :: (IO DQNNet) -- getRandomNet
                     -> (DQNNet -> DQNState -> Double) -- evalCritic
                     -> (DQNNet -> DQNState -> Double -> Double -> DQNNet) -- trainOneStep
                     -> (DQNNet -> DQNState -> Double -> Double) -- computeLoss
                     -> IO ()
testSyntheticLearning getRandomNet evalCritic trainOneStep computeLoss = do
  putStrLn "=== Testing DQN Learning on Synthetic Data ==="
  
  -- Test sum of squares function (nonlinear)
  testFunction getRandomNet evalCritic trainOneStep computeLoss "Sum of Squares: f(x) = x1² + x2² + x3² + x4²" generateSumSquares testPatterns
  
  -- Test linear combination (should be easier)
  testFunction getRandomNet evalCritic trainOneStep computeLoss "Linear Combination: f(x) = 2*x1 + 3*x2 - x3 + 0.5*x4" generateLinearCombo linearComboPatterns