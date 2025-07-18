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

type DQNNet o = Network 4 64 64 o
type DQNCritic = DQNNet 1
type DQNActor = DQNNet 2
type DQNState = R 4
type Reward = Double
type Trajectory = [(DQNState, Action, Reward)]
newtype DiscountedTrajectory = DT Trajectory

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

randomAction :: IO Int
randomAction = randomRIO (0, 1)

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
runDQNNetwork :: forall o. KnownNat o => DQNNet o -> DQNState -> R o
runDQNNetwork net input = runLayerNormal (net ^. nLayer3)
                        . reluVec
                        . runLayerNormal (net ^. nLayer2)
                        . reluVec
                        . runLayerNormal (net ^. nLayer1)
                        $ input

-- Convenient aliases for clarity
runNetForQ :: forall o. KnownNat o => DQNNet o -> DQNState -> R o
runNetForQ = runDQNNetwork

runNetForActor :: DQNActor -> DQNState -> R 2
runNetForActor = runDQNNetwork

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

-- ============================================================================
-- ACTION SELECTION
-- ============================================================================

-- Random action selection for critic exploration
getRandomAction :: IO Action
getRandomAction = do
  actionInt <- randomAction
  return $ Action $ Number $ fromIntegral actionInt

-- Policy-based action selection for actor
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

-- Trajectory sampling with random actions (for critic)
sampleTrajectoryRandom :: DQNState -> (Action -> IO (DQNState, Reward, Bool)) -> IO Trajectory
sampleTrajectoryRandom = sampleTrajectoryWith (const getRandomAction)

-- Trajectory sampling with actor policy
sampleTrajectoryWithActor :: DQNActor -> DQNState -> (Action -> IO (DQNState, Reward, Bool)) -> IO Trajectory
sampleTrajectoryWithActor net = sampleTrajectoryWith (getActionFromActor net)

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

sampleMultipleTrajectories :: Int -> Environment -> DQNActor -> IO [Trajectory]
sampleMultipleTrajectories n envHandle actorNet = do
  replicateM n (trajectoryFromEnvWithActor envHandle actorNet)

-- ============================================================================
-- TARGET CALCULATION
-- ============================================================================

calculateQTargets :: forall o. KnownNat o => Double -> Trajectory -> [(DQNState, R o)]
calculateQTargets gamma trajectory = 
  let (DT discountedTrajectory) = makeDiscountedTrajectory gamma trajectory
  in map (\(state, _, discountedReward) -> 
      let target = vector [discountedReward]
      in (state, target)) discountedTrajectory

calculateActionQTargets :: DQNActor -> Double -> Trajectory -> [(DQNState, R 2)]
calculateActionQTargets net gamma trajectory = 
  let (DT discountedTrajectory) = makeDiscountedTrajectory gamma trajectory
  in map (\(state, action, discountedReward) -> 
      let predicted = runNetForActor net state
          extracted = extract predicted
          target = if (actionToInt action) == 0 
                   then vector [discountedReward, extracted LA.! 1] 
                   else vector [extracted LA.! 0, discountedReward]
      in (state, target)) discountedTrajectory

-- ============================================================================
-- LOSS CALCULATION
-- ============================================================================

averageMSELoss :: forall o. KnownNat o => DQNNet o -> Double -> Trajectory -> Double
averageMSELoss net gamma trajectory = 
  let (DT discountedTrajectory) = makeDiscountedTrajectory gamma trajectory
      losses = map calculateLoss discountedTrajectory
  in sum losses / fromIntegral (length losses)
  where
    calculateLoss (state, _, discountedReward) = 
      let predictedQ = runNetForQ net state
          predictedValue = (extract predictedQ) LA.! 0
          err = predictedValue - discountedReward
      in err * err

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
      in sum (map (^2) squaredDiff) / 2.0



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

-- MSE loss function for backpropagation
mseErr :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o, Reifies s W)
       => R i -> R o -> BVar s (Network i h1 h2 o) -> BVar s Double
mseErr input target net = 
  let predicted = runNetworkForQ net input
      diff = predicted - constVar target
  in 0.5 * (diff <.>! diff)

-- ============================================================================
-- TRAINING FUNCTIONS
-- ============================================================================

-- MSE-based training step for Q-value regression
trainStepMSE :: forall o. KnownNat o => Double -> DQNNet o -> DQNState -> R o -> DQNNet o
trainStepMSE learningRate net input target = 
  net - realToFrac learningRate * gradBP (mseErr input target) net

-- MSE-based training step for actor
trainStepActorMSE :: Double -> DQNActor -> DQNState -> R 2 -> DQNActor
trainStepActorMSE learningRate net input target = 
  net - realToFrac learningRate * gradBP (mseErr input target) net

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

-- ============================================================================
-- ENVIRONMENT INTERACTION
-- ============================================================================

-- Generic trajectory generation from environment with custom action selector
trajectoryFromEnvWith :: (DQNState -> IO Action) -> Environment -> IO Trajectory
trajectoryFromEnvWith actionSelector envHandle = do
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
          sampleTrajectoryWith actionSelector stateVec (makeTransition envHandle)

-- Trajectory using random actions (for critic training)
trajectoryFromEnvRandom :: Environment -> IO Trajectory
trajectoryFromEnvRandom = trajectoryFromEnvWith (const getRandomAction)

-- Trajectory using actor policy
trajectoryFromEnvWithActor :: Environment -> DQNActor -> IO Trajectory
trajectoryFromEnvWithActor envHandle net = trajectoryFromEnvWith (getActionFromActor net) envHandle

trainForEpochs :: forall o. KnownNat o => DQNNet o -> Double -> Double -> Int -> Environment -> IO (DQNNet o)
trainForEpochs net _ _ 0 _ = return net
trainForEpochs net learningRate gamma epochs envHandle = do
  trajectory <- trajectoryFromEnvRandom envHandle
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
  criticTrajectory <- trajectoryFromEnvRandom envHandle
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
  trajectory <- trajectoryFromEnvWithActor envHandle actorNet
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
              criticTrajectory <- sampleTrajectoryRandom stateVec (makeTransition envHandle)
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
              finalCriticTrajectory <- trajectoryFromEnvWithActor envHandle finalActor
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
        
