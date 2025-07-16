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

module NeuralNetwork (
    -- * Types
    Layer(..),
    Network(..),
    
    -- * Lenses
    lWeights,
    lBiases,
    nLayer1,
    nLayer2,
    nLayer3,
    
    -- * Backprop operations
    (#>!),
    (<.>!),
    konst',
    sumElements',
    
    -- * Layer operations
    runLayerNormal,
    runLayer,
    
    -- * Activation functions
    softMaxNormal,
    softMax,
    logistic,
    
    -- * Network operations
    runNetNormal,
    runNetwork,
    
    -- * Loss functions
    crossEntropyNormal,
    crossEntropy,
    netErr
) where

import           Control.DeepSeq
import           Control.Monad
import           Data.Foldable
import           Data.Proxy
import           GHC.Generics                        (Generic)
import           GHC.TypeLits
import           Lens.Micro
import           Lens.Micro.TH
import           Numeric.Backprop
import           Numeric.Backprop.Class
-- import           Numeric.Backprop.Num
import           Numeric.LinearAlgebra.Static
import qualified Numeric.LinearAlgebra               as HM
import           Numeric.OneLiner
import qualified System.Random.MWC                   as MWC
import qualified System.Random.MWC.Distributions     as MWC

-- | A neural network layer with weights and biases
data Layer i o = Layer 
    { _lWeights :: !(L o i)
    , _lBiases  :: !(R o)
    } deriving (Show, Generic)

instance NFData (Layer i o)
makeLenses ''Layer

-- | A three-layer neural network
data Network i h1 h2 o = Net 
    { _nLayer1 :: !(Layer i  h1)
    , _nLayer2 :: !(Layer h1 h2)
    , _nLayer3 :: !(Layer h2 o)
    } deriving (Show, Generic)

instance NFData (Network i h1 h2 o)
makeLenses ''Network

instance (KnownNat i, KnownNat o) => Num (Layer i o) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

instance ( KnownNat i
         , KnownNat h1
         , KnownNat h2
         , KnownNat o
         ) => Num (Network i h1 h2 o) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

instance (KnownNat i, KnownNat o) => Fractional (Layer i o) where
    (/)          = gDivide
    recip        = gRecip
    fromRational = gFromRational

instance ( KnownNat i
         , KnownNat h1
         , KnownNat h2
         , KnownNat o
         ) => Fractional (Network i h1 h2 o) where
    (/)          = gDivide
    recip        = gRecip
    fromRational = gFromRational

-- | Backprop instances
instance (KnownNat i, KnownNat o) => Backprop (Layer i o)
instance (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o) => Backprop (Network i h1 h2 o)

-- | Backprop-aware matrix-vector multiplication
infixr 8 #>!
(#>!) :: (KnownNat m, KnownNat n, Reifies s W)
      => BVar s (L m n)
      -> BVar s (R n)
      -> BVar s (R m)
(#>!) = liftOp2 . op2 $ \m v ->
  ( m #> v, \g -> (g `outer` v, tr m #> g) )

-- | Backprop-aware dot product
infixr 8 <.>!
(<.>!) :: (KnownNat n, Reifies s W)
       => BVar s (R n)
       -> BVar s (R n)
       -> BVar s Double
(<.>!) = liftOp2 . op2 $ \x y ->
  ( x <.> y, \g -> (konst g * y, x * konst g) )

-- | Backprop-aware constant vector creation
konst' :: (KnownNat n, Reifies s W)
       => BVar s Double
       -> BVar s (R n)
konst' = liftOp1 . op1 $ \c -> (konst c, HM.sumElements . extract)

-- | Backprop-aware sum of elements
sumElements' :: (KnownNat n, Reifies s W)
             => BVar s (R n)
             -> BVar s Double
sumElements' = liftOp1 . op1 $ \x -> (HM.sumElements (extract x), konst)

-- | Run a layer without backpropagation
runLayerNormal :: (KnownNat i, KnownNat o)
               => Layer i o
               -> R i
               -> R o
runLayerNormal l x = (l ^. lWeights) #> x + (l ^. lBiases)
{-# INLINE runLayerNormal #-}

-- | Run a layer with backpropagation
runLayer :: (KnownNat i, KnownNat o, Reifies s W)
         => BVar s (Layer i o)
         -> BVar s (R i)
         -> BVar s (R o)
runLayer l x = (l ^^. lWeights) #>! x + (l ^^. lBiases)
{-# INLINE runLayer #-}

-- | Softmax activation without backpropagation
softMaxNormal :: KnownNat n => R n -> R n
softMaxNormal x = konst (1 / HM.sumElements (extract expx)) * expx
  where
    expx = exp x
{-# INLINE softMaxNormal #-}

-- | Softmax activation with backpropagation
softMax :: (KnownNat n, Reifies s W) => BVar s (R n) -> BVar s (R n)
softMax x = konst' (1 / sumElements' expx) * expx
  where
    expx = exp x
{-# INLINE softMax #-}

-- | Logistic activation function
logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))
{-# INLINE logistic #-}

-- | Run network without backpropagation
runNetNormal :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o)
             => Network i h1 h2 o
             -> R i
             -> R o
runNetNormal n = softMaxNormal
               . runLayerNormal (n ^. nLayer3)
               . logistic
               . runLayerNormal (n ^. nLayer2)
               . logistic
               . runLayerNormal (n ^. nLayer1)
{-# INLINE runNetNormal #-}

-- | Run network with backpropagation
runNetwork :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o, Reifies s W)
           => BVar s (Network i h1 h2 o)
           -> R i
           -> BVar s (R o)
runNetwork n = softMax
             . runLayer (n ^^. nLayer3)
             . logistic
             . runLayer (n ^^. nLayer2)
             . logistic
             . runLayer (n ^^. nLayer1)
             . constVar
{-# INLINE runNetwork #-}

-- | Cross-entropy loss without backpropagation
crossEntropyNormal :: KnownNat n => R n -> R n -> Double
crossEntropyNormal targ res = -(log res <.> targ)
{-# INLINE crossEntropyNormal #-}

-- | Cross-entropy loss with backpropagation
crossEntropy :: (KnownNat n, Reifies s W)
             => R n
             -> BVar s (R n)
             -> BVar s Double
crossEntropy targ res = -(log res <.>! constVar targ)
{-# INLINE crossEntropy #-}

-- | Network error function
netErr :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o, Reifies s W)
       => R i
       -> R o
       -> BVar s (Network i h1 h2 o)
       -> BVar s Double
netErr x targ n = crossEntropy targ (runNetwork n x)
{-# INLINE netErr #-}

-- | Backprop instances for static vectors and matrices
instance Backprop (R n) where
    zero = zeroNum
    add  = addNum
    one  = oneNum

instance (KnownNat n, KnownNat m) => Backprop (L m n) where
    zero = zeroNum
    add  = addNum
    one  = oneNum

-- | MWC Variate instances for random generation
instance KnownNat n => MWC.Variate (R n) where
    uniform g = randomVector <$> MWC.uniform g <*> pure Uniform
    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

instance (KnownNat m, KnownNat n) => MWC.Variate (L m n) where
    uniform g = uniformSample <$> MWC.uniform g <*> pure 0 <*> pure 1
    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

instance (KnownNat i, KnownNat o) => MWC.Variate (Layer i o) where
    uniform g = Layer <$> MWC.uniform g <*> MWC.uniform g
    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

instance ( KnownNat i, KnownNat h1, KnownNat h2, KnownNat o
         ) => MWC.Variate (Network i h1 h2 o) where
    uniform g = Net <$> MWC.uniform g <*> MWC.uniform g <*> MWC.uniform g
    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g