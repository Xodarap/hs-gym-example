{-# LANGUAGE OverloadedStrings #-}
module Main where

import Data.Aeson (Value(Number))
import Control.Exception (bracket)
import Control.Monad (replicateM_)

import Gym.Environment
import Gym.Core

main :: IO ()
main = do
    putStrLn "Testing gym-hs library integration..."
    
    putStrLn "Creating CartPole-v1 environment..."
    result <- makeEnv "CartPole-v1"
    case result of
        Left err -> putStrLn $ "Error creating environment: " ++ show err
        Right env -> bracket (return env) closeEnv $ \env -> do
            putStrLn "Environment created successfully!"
            
            putStrLn "Resetting environment..."
            resetResult <- reset env
            case resetResult of
                Left err -> putStrLn $ "Reset error: " ++ show err
                Right (Observation obs) -> do
                    putStrLn $ "Initial observation: " ++ show obs
                    
                    putStrLn "Running 10 random steps..."
                    runSteps env 10
                    
                    putStrLn "gym-hs test completed successfully!"

runSteps :: Environment -> Int -> IO ()
runSteps env 0 = return ()
runSteps env n = do
    -- Take a random action (0 or 1 for CartPole)
    let action = if n `mod` 2 == 0 then Number 0 else Number 1
    stepResult <- step env (Action action)
    case stepResult of
        Left err -> putStrLn $ "Step error: " ++ show err
        Right result -> do
            putStrLn $ "Step " ++ show (11 - n) ++ ": action=" ++ show action ++ 
                       ", reward=" ++ show (stepReward result) ++ 
                       ", terminated=" ++ show (stepTerminated result) ++
                       ", truncated=" ++ show (stepTruncated result)
            if stepTerminated result || stepTruncated result
                then do
                    putStrLn "Episode ended, resetting..."
                    resetResult <- reset env
                    case resetResult of
                        Left err -> putStrLn $ "Reset error: " ++ show err
                        Right _ -> runSteps env (n - 1)
                else runSteps env (n - 1)