modules
=======

.. toctree::

.. automodule:: pydrobert.torch.modules

Attention
---------
.. autoclass:: GlobalSoftAttention
.. autoclass:: ConcatSoftAttention
.. autoclass:: DotProductSoftAttention
.. autoclass:: GeneralizedDotProductSoftAttention
.. autoclass:: MultiHeadedAttention

Decoding
--------
.. autoclass:: BeamSearch
.. autoclass:: CTCGreedySearch
.. autoclass:: CTCPrefixSearch
.. autoclass:: RandomWalk
.. autoclass:: SequenceLogProbabilities

Features
--------
.. autoclass:: ChunkBySlices
.. autoclass:: ChunkTokenSequencesBySlices
.. autoclass:: DenseImageWarp
.. autoclass:: FeatureDeltas
.. autoclass:: MeanVarianceNormalization
.. autoclass:: PadMaskedSequence
.. autoclass:: PadVariable
.. autoclass:: PolyharmonicSpline
.. autoclass:: RandomShift
.. autoclass:: SliceSpectData
.. autoclass:: SparseImageWarp
.. autoclass:: SpecAugment
.. autoclass:: Warp1DGrid

Language Models
---------------
.. autoclass:: ExtractableSequentialLanguageModel
.. autoclass:: MixableSequentialLanguageModel
.. autoclass:: SequentialLanguageModel
.. autoclass:: ExtractableShallowFusionLanguageModel
.. autoclass:: LookupLanguageModel
.. autoclass:: MixableShallowFusionLanguageModel
.. autoclass:: ShallowFusionLanguageModel

Reinforcement Learning
----------------------
.. autoclass:: GumbelOneHotCategoricalRebarControlVariate
.. autoclass:: LogisticBernoulliRebarControlVariate
.. autoclass:: TimeDistributedReturn

String Matching
---------------
.. autoclass:: EditDistance
.. autoclass:: ErrorRate
.. autoclass:: FillAfterEndOfSequence
.. autoclass:: HardOptimalCompletionDistillationLoss
.. autoclass:: MinimumErrorRateLoss
.. autoclass:: OptimalCompletion
.. autoclass:: PrefixEditDistances
.. autoclass:: PrefixErrorRates