{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE TypeApplications    #-}

module Main where

import           NeuralNetwork
import           Numeric.LinearAlgebra.Static
import           GHC.TypeLits
import qualified System.Random.MWC as MWC
import           Lens.Micro
import           Lens.Micro.TH

-- | Simple ReLU activation function
relu :: (KnownNat i) => R i -> R i
relu x = dvmap (max 0) x

-- | Example network running function using ReLU instead of logistic
runNetworkReLU :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o)
               => Network i h1 h2 o
               -> R i
               -> R o
runNetworkReLU n = softMaxNormal
                 . runLayerNormal (n ^. nLayer3)
                 . relu
                 . runLayerNormal (n ^. nLayer2)
                 . relu
                 . runLayerNormal (n ^. nLayer1)

-- | Main function demonstrating the neural network library
main :: IO ()
main = MWC.withSystemRandom $ \g -> do
  putStrLn "Initializing neural network..."
--   net0 <- uniformRNetwork @4 @64 @64 @2 (-0.5, 0.5) g
--   putStrLn "Network initialized."
  
--   -- Example input (4 features)
--   let input = vector [1.0, 0.5, -0.3, 2.1]
  
--   -- Run network
--   let output = runNetworkReLU net0 input
--   putStrLn $ "Input: " ++ show (extract input)
--   putStrLn $ "Output: " ++ show (extract output)
  
--   -- Example training on dummy data
--   let trainingData = [
--         (vector [1.0, 0.0, 0.0, 0.0], vector [1.0, 0.0]),
--         (vector [0.0, 1.0, 0.0, 0.0], vector [0.0, 1.0]),
--         (vector [0.0, 0.0, 1.0, 0.0], vector [1.0, 0.0]),
--         (vector [0.0, 0.0, 0.0, 1.0], vector [0.0, 1.0])]
  
--   putStrLn "\nTraining network..."
--   let trainedNet = trainList 0.1 trainingData net0
  
--   putStrLn "Testing trained network:"
--   let testAccuracy = testNet trainingData trainedNet
--   putStrLn $ "Test accuracy: " ++ show (testAccuracy * 100) ++ "%"
  
--   -- Test individual predictions
--   putStrLn "\nIndividual predictions:"
--   mapM_ (\(input', expected) -> do
--     let predicted = runNetworkReLU trainedNet input'
--     putStrLn $ "Input: " ++ show (extract input') ++ 
--                " -> Expected: " ++ show (extract expected) ++ 
--                " -> Predicted: " ++ show (extract predicted)
--   ) trainingData