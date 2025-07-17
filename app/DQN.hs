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
import Control.Monad (replicateM, when)
import Data.List (maximumBy)
import Data.Ord (comparing)
import System.Random
-- import GHC.Generics
import Gym.Environment
import Gym.Core
import qualified System.Random.MWC as MWC
import           Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra               as HM
import Data.Aeson (Value(Number, Array))
import qualified Data.Vector as V
import qualified Numeric.LinearAlgebra as LA

-- DQN Network Architecture
data DQNSpec = DQNSpec { inputSize :: Int, hiddenSize :: Int, outputSize :: Int }
  deriving (Show, Eq)


type DQNNet = Network 4 64 64 2
type DQNState = R 4
type DQNOutput = R 2
type Reward = Double
type Trajectory = [(DQNState, DQNOutput, Reward)]
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

getAction :: DQNNet -> DQNState -> IO Action
getAction net input = do
  let output = runNetNormal net input
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
  let output = runNetNormal net input
  action <- getAction net input
  (nextState, reward, done) <- transition action
  if done
    then return [(input, output, reward)]
    else do
      nextTrajectory <- sampleTrajectory net nextState transition
      return ((input, output, reward) : nextTrajectory)

makeDiscountedTrajectory :: Double -> Trajectory -> DiscountedTrajectory
makeDiscountedTrajectory gamma trajectory = DT $ reverse $ snd $ foldl (\(d, l) (i, o, r) -> (d * gamma, (i, o, d * r) : l)) (1,[]) trajectory

parseObservation :: Observation -> Maybe [Double]
parseObservation (Observation (Array arr)) = 
  let values = V.toList arr
      doubles = mapM parseNumber values
  in doubles
  where
    parseNumber (Number n) = Just (realToFrac n)
    parseNumber _ = Nothing
parseObservation _ = Nothing

-- parseObservation :: Observation -> [Double]
-- parseObservation (Observation (Array arr)) = (\(Number n) -> toRealFloat n) <$> arr
-- parseObservation _ = []

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
              trajectory <- sampleTrajectory net0 stateVec (makeTransition envHandle)
              putStrLn $ "Trajectory: " ++ show trajectory
              return ()
        

-- loss :: DQNNet -> DQNInput -> DQNOutput -> Double
-- loss net input target = sum (zipWith (\a b -> (a - b) ^ 2) (extract output) (extract target)) / fromIntegral (length output)
--   where output = runNetNormal net input



-- instance T.Randomizable DQNSpec DQN where
--   sample DQNSpec{..} = DQN 
--     <$> T.sample (T.LinearSpec inputSize hiddenSize)
--     <*> T.sample (T.LinearSpec hiddenSize hiddenSize)
--     <*> T.sample (T.LinearSpec hiddenSize outputSize)

-- instance T.Parameterized DQN

-- forward :: DQN -> T.Tensor -> T.Tensor
-- forward DQN{..} x = 
--   let h1 = F.relu (T.linear fc1 x)
--       h2 = F.relu (T.linear fc2 h1)
--   in T.linear fc3 h2

-- -- Experience Replay Buffer
-- data Experience = Experience
--   { state :: [Float]
--   , action :: Int
--   , reward :: Float
--   , nextState :: [Float]
--   , done :: Bool
--   } deriving (Show)

-- data ReplayBuffer = ReplayBuffer
--   { buffer :: [Experience]
--   , capacity :: Int
--   , position :: Int
--   } deriving (Show)

-- createReplayBuffer :: Int -> ReplayBuffer
-- createReplayBuffer cap = ReplayBuffer [] cap 0

-- addExperience :: ReplayBuffer -> Experience -> ReplayBuffer
-- addExperience rb@ReplayBuffer{..} exp =
--   if length buffer < capacity
--     then rb { buffer = buffer ++ [exp] }
--     else rb { buffer = take position buffer ++ [exp] ++ drop (position + 1) buffer
--             , position = (position + 1) `mod` capacity }

-- sampleBatch :: ReplayBuffer -> Int -> IO [Experience]
-- sampleBatch ReplayBuffer{..} batchSize = do
--   indices <- replicateM batchSize (randomRIO (0, length buffer - 1))
--   return $ map (buffer !!) indices

-- -- DQN Agent
-- data DQNAgent = DQNAgent
--   { qNetwork :: DQN
--   , targetNetwork :: DQN
--   , optimizer :: Optim.Adam
--   , replayBuffer :: ReplayBuffer
--   , epsilon :: Float
--   , epsilonMin :: Float
--   , epsilonDecay :: Float
--   , learningRate :: Float
--   , gamma :: Float
--   , batchSize :: Int
--   , targetUpdateFreq :: Int
--   , stepCount :: Int
--   }

-- createDQNAgent :: DQNSpec -> Float -> IO DQNAgent
-- createDQNAgent spec lr = do
--   qNet <- T.sample spec
--   targetNet <- T.sample spec
--   opt <- Optim.adam 0 0.9 0.999 (T.flattenParameters qNet)
--   return $ DQNAgent
--     { qNetwork = qNet
--     , targetNetwork = targetNet
--     , optimizer = opt
--     , replayBuffer = createReplayBuffer 10000
--     , epsilon = 1.0
--     , epsilonMin = 0.01
--     , epsilonDecay = 0.995
--     , learningRate = lr
--     , gamma = 0.99
--     , batchSize = 32
--     , targetUpdateFreq = 100
--     , stepCount = 0
--     }

-- -- Action Selection
-- selectAction :: DQNAgent -> [Float] -> IO Int
-- selectAction agent@DQNAgent{..} state = do
--   r <- randomRIO (0.0, 1.0)
--   if r < epsilon
--     then randomRIO (0, 1) -- CartPole has 2 actions
--     else do
--       let stateTensor = T.asTensor state
--       let qValues = forward qNetwork stateTensor
--       let qList = T.asValue qValues :: [Float]
--       return $ snd $ maximumBy (comparing fst) (zip qList [0..])

-- -- Training Step
-- trainStep :: DQNAgent -> IO DQNAgent
-- trainStep agent@DQNAgent{..} = do
--   if length (buffer replayBuffer) < batchSize
--     then return agent
--     else do
--       batch <- sampleBatch replayBuffer batchSize
      
--       let states = map (T.asTensor . state) batch
--       let actions = map action batch
--       let rewards = map reward batch
--       let nextStates = map (T.asTensor . nextState) batch
--       let dones = map done batch
      
--       let stateBatch = T.stack states 0
--       let nextStateBatch = T.stack nextStates 0
      
--       let currentQValues = forward qNetwork stateBatch
--       let nextQValues = forward targetNetwork nextStateBatch
      
--       let targetQValues = zipWith3 (\r nextQ isDone -> if isDone then r else r + gamma * (maximum $ T.asValue nextQ :: [Float])) rewards (T.unbind nextQValues 0) dones
      
--       let loss = F.mseLoss currentQValues (T.asTensor targetQValues)
      
--       (newOpt, _) <- Optim.runStep optimizer loss learningRate
      
--       let newEpsilon = max epsilonMin (epsilon * epsilonDecay)
--       let newStepCount = stepCount + 1
      
--       let updatedAgent = agent 
--             { optimizer = newOpt
--             , epsilon = newEpsilon
--             , stepCount = newStepCount
--             }
      
--       if newStepCount `mod` targetUpdateFreq == 0
--         then return updatedAgent { targetNetwork = qNetwork }
--         else return updatedAgent

-- -- CartPole Environment Interface
-- data CartPoleState = CartPoleState [Float] deriving (Show)
-- data CartPoleAction = LeftAction | RightAction deriving (Show, Eq)

-- actionToInt :: CartPoleAction -> Int
-- actionToInt LeftAction = 0
-- actionToInt RightAction = 1

-- intToAction :: Int -> CartPoleAction
-- intToAction 0 = LeftAction
-- intToAction 1 = RightAction
-- intToAction _ = LeftAction

-- -- Simple CartPole simulation (placeholder)
-- stepCartPole :: CartPoleState -> CartPoleAction -> IO (CartPoleState, Float, Bool)
-- stepCartPole (CartPoleState state) action = do
--   -- Simplified CartPole dynamics - replace with actual gym interface
--   let newState = map (+0.01) state -- Placeholder
--   let reward = 1.0
--   let done = length newState > 200 -- Placeholder termination condition
--   return (CartPoleState newState, reward, done)

-- resetCartPole :: IO CartPoleState
-- resetCartPole = return $ CartPoleState [0.0, 0.0, 0.0, 0.0]

-- -- Training Loop
-- trainDQN :: Int -> IO ()
-- trainDQN episodes = do
--   let spec = DQNSpec 4 128 2 -- CartPole: 4 state dims, 2 actions
--   agent <- createDQNAgent spec 0.001
  
--   trainLoop agent episodes
--   where
--     trainLoop agent 0 = putStrLn "Training completed!"
--     trainLoop agent eps = do
--       state <- resetCartPole
--       episodeReward <- runEpisode agent state 0
--       putStrLn $ "Episode " ++ show (episodes - eps + 1) ++ ", Reward: " ++ show episodeReward
--       newAgent <- trainStep agent
--       trainLoop newAgent (eps - 1)
    
--     runEpisode agent (CartPoleState state) totalReward = do
--       action <- selectAction agent state
--       (newState@(CartPoleState newStateList), reward, done) <- stepCartPole (CartPoleState state) (intToAction action)
      
--       let experience = Experience state action reward newStateList done
--       let newBuffer = addExperience (replayBuffer agent) experience
--       let updatedAgent = agent { replayBuffer = newBuffer }
      
--       if done
--         then return (totalReward + reward)
--         else runEpisode updatedAgent newState (totalReward + reward)

-- -- Main function for CartPole training
-- main :: IO ()
-- main = do
--   putStrLn "Starting DQN training on CartPole..."
--   trainDQN 1000