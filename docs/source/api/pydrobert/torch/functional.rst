functional
==========

.. toctree::

.. automodule:: pydrobert.torch.functional

Combinatorics
-------------
.. autofunction:: binomial_coefficient
.. autofunction:: enumerate_vocab_sequences
.. autofunction:: enumerate_binary_sequences
.. autofunction:: enumerate_binary_sequences_with_cardinality
.. autofunction:: simple_random_sampling_without_replacement

Decoding
--------
.. autofunction:: beam_search_advance
.. autofunction:: ctc_greedy_search
.. autofunction:: ctc_prefix_search_advance
.. autofunction:: random_walk_advance
.. autofunction:: sequence_log_probs

Features
--------
.. autofunction:: chunk_by_slices
.. autofunction:: chunk_token_sequences_by_slices
.. autofunction:: dense_image_warp
.. autofunction:: feat_deltas
.. autofunction:: mean_var_norm
.. autofunction:: pad_masked_sequence
.. autofunction:: pad_variable
.. autofunction:: polyharmonic_spline
.. autofunction:: random_shift
.. autofunction:: slice_spect_data
.. autofunction:: sparse_image_warp
.. autofunction:: spec_augment
.. autofunction:: spec_augment_apply_parameters
.. autofunction:: spec_augment_draw_parameters
.. autofunction:: warp_1d_grid

Reinforcement Learning
----------------------
.. autofunction:: time_distributed_return

String Matching
---------------
.. autofunction:: edit_distance
.. autofunction:: error_rate
.. autofunction:: fill_after_eos
.. autofunction:: hard_optimal_completion_distillation_loss
.. autofunction:: minimum_error_rate_loss
.. autofunction:: optimal_completion
.. autofunction:: prefix_edit_distances
.. autofunction:: prefix_error_rates