Command-Line Interface
======================

chunk-torch-spect-data-dir
--------------------------

::

  usage: chunk-torch-spect-data-dir [-h] [--file-prefix FILE_PREFIX]
                                    [--file-suffix FILE_SUFFIX]
                                    [--feat-subdir FEAT_SUBDIR]
                                    [--ali-subdir ALI_SUBDIR]
                                    [--ref-subdir REF_SUBDIR]
                                    [--num-workers NUM_WORKERS]
                                    [--mp-chunk-size MP_CHUNK_SIZE]
                                    [--policy {fixed,ali,ref}]
                                    [--lobe-size LOBE_SIZE]
                                    [--window-type {symmetric,causal,future}]
                                    [--pad-mode {constant,reflect,replicate}]
                                    [--pad-constant PAD_CONSTANT]
                                    [--partial-tokens]
                                    [--retain-token-boundaries] [--quiet]
                                    [--format-utt FORMAT_UTT]
                                    in_dir out_dir
  
  Create a new SpectDataSet directory by chunking another
  
  This command breaks SpectDataSet sequences into sub-sequences (chunks), storing the
  results in a new directory. New utterances are named according to "--format-utt".
  
  Sequences are sliced according to one of three policies set by the "--policy" flag
  (default "fixed"). They are:
  
  - fixed: extract a fixed-sized window at fixed-length intervals along the feature
           sequence.
  - ali: use per-frame alignments to segment the feature sequence into intervals with
         matching labels. Requires per-frame alignments (data in the "ali/" subdirectory).
  - ref: use reference token sequence segments as slices. Requires reference sequences
         (data in the "ali/" subdirectory) and for them to contain segment boundary
         information.
  
  Overlapping chunks may be created by specifying "--lobe-size" (default "0") and
  "--window-type" (default "symmetric"). More details on the policies and windowing can
  be found in the Python module pydrobert.torch.modules.SliceSpectData.
  
  By default, only valid slices (i.e. those entirely within the boundaries of the input
  sequences) are counted. Specifying "--pad-mode" will include slices partially within
  boundaries as well as how to pad features and per-frame alignments to fill the
  remainder.
  
  See the command "get-torch-spect-data-dir-info" for more info SpectDataSet directories.
  
  positional arguments:
    in_dir                The torch data directory to chunk (input)
    out_dir               The torch data directory to store chunks (output)
  
  optional arguments:
    -h, --help            show this help message and exit
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --feat-subdir FEAT_SUBDIR
                          Subdirectory where features are stored.
    --ali-subdir ALI_SUBDIR
                          Subdirectory where per-frame alignments are stored.
    --ref-subdir REF_SUBDIR
                          Subdirectory where reference token sequences are
                          stored.
    --num-workers NUM_WORKERS
                          The number of workers to spawn to process the data. 0
                          is serial. Defaults to the CPU count
    --mp-chunk-size MP_CHUNK_SIZE
                          The number of utterances that a multiprocessing worker
                          will process at once. Impacts speed and memory
                          consumption.
    --policy {fixed,ali,ref}
                          The policy for determining slices from the data. See
                          SliceSpectData.
    --lobe-size LOBE_SIZE
                          Size of a side lobe of a slice. See SliceSpectData.
    --window-type {symmetric,causal,future}
                          Type of window used in slicing. See SliceSpectData.
    --pad-mode {constant,reflect,replicate}
                          If specified, determines how to chunks of features and
                          alignments exceeding the original sequence boundaries.
                          constant: pad with the value of '--pad-constant'.
                          reflect: padded values are the reflection around
                          sequence boundaries. replicate: padded values match
                          the first and final sequence values.
    --pad-constant PAD_CONSTANT
                          Constant used when padding with '--pad-mode=constant'
    --partial-tokens      If set, reference token sequences which only partly
                          overlap with a chunk will still be included with the
                          chunk.
    --retain-token-boundaries
                          If set, segment boundaries of reference token
                          sequences will keep their original values rather than
                          being made relative to the chunk.
    --quiet               Suppress any warnings.
    --format-utt FORMAT_UTT
                          Format string with which to format utterance ids of
                          chunks. Available keys are 'utt_id': the old utterance
                          id, 'start': the start frame of the chunk (inclusive),
                          'end': the end frame of the chunk (exclusive), and
                          'idx': the 0-index of the chunk within the utterance

compute-mvn-stats-for-torch-feat-data-dir
-----------------------------------------

::

  usage: compute-mvn-stats-for-torch-feat-data-dir [-h]
                                                   [--file-prefix FILE_PREFIX]
                                                   [--file-suffix FILE_SUFFIX]
                                                   [--num-workers NUM_WORKERS]
                                                   [--dim DIM] [--id2gid ID2GID]
                                                   [--bessel]
                                                   dir out
  
  Compute mean and standard deviation over a torch feature directory
  
  A feature directory is of the form
  
  dir/
      <file_prefix><id_1><file_suffix>
      <file_prefix><id_2><file_suffix>
      ...
  
  where each file contains a dynamically-sized tensor whose last dimension (by default) is
  a feature vector. Letting F be a feature vector, this command computes the mean and
  standard deviation of the features in the directory, storing them as a pickled
  dictionary of tensors (with keys 'mean' and 'std') to the file 'out'. Those statistics
  may be used with a pydrobert.torch.modules.MeanVarianceNormalization layer.
  
  positional arguments:
    dir                   The feature directory
    out                   Output path
  
  optional arguments:
    -h, --help            show this help message and exit
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --num-workers NUM_WORKERS
                          The number of workers to spawn to process the data. 0
                          is serial. Defaults to the CPU count
    --dim DIM             The dimension of the feature vector
    --id2gid ID2GID       Path to a file mapping feature tensors to groups. See
                          below for more info
    --bessel              Apply Bessel's correction
                          (https://en.wikipedia.org/wiki/Bessel's_correction) to
                          estimates.
  
  If --id2gid is specified, it points to a file which maps file ids to groups. Each group
  gets its own statistics which are estimated using only the feature vectors from the
  files assigned to them. With <id_1>, <id_2>, etc. part of the file names in the feature
  directory as above and <gid_1>, <gid_2>, etc. strings without spaces representing group
  ids, then the argument passed to --id2gid is a file with lines
  
      <id_x> <gid_y>
  
  defining a surjective mapping from file ids to group ids. 'out' will then store a
  pickled, nested dictionary
  
      {
          <gid_1>: {'mean': ..., 'var': ...},
          <gid_2>: {'mean': ..., 'var': ...},
          ...
      }
  
  of the statistics of all groups.

compute-torch-token-data-dir-error-rates
----------------------------------------

::

  usage: compute-torch-token-data-dir-error-rates [-h] [--id2token ID2TOKEN]
                                                  [--replace REPLACE]
                                                  [--ignore IGNORE]
                                                  [--file-prefix FILE_PREFIX]
                                                  [--file-suffix FILE_SUFFIX]
                                                  [--swap] [--warn-missing]
                                                  [--distances] [--per-utt]
                                                  [--batch-size BATCH_SIZE]
                                                  [--quiet]
                                                  [--costs INS DEL SUB | --nist-costs]
                                                  dir [hyp] [out]
  
  Compute error rates between reference and hypothesis token data dirs
  
  WARNING!!!!
  The error rates reported by this command have changed since version v0.3.0 of
  pydrobert-pytorch when the insertion, deletion, and substitution costs do not all equal
  1. Consult the documentation of "pydrobert.torch.functional.error_rate" for more
  information.
  
  This is a very simple script that computes and prints the error rates between the "ref/"
  (reference/gold standard) token sequences and "hyp/" (hypothesis/generated) token
  sequences in a SpectDataSet directory. Consult the Wikipedia article on the Levenshtein
  distance (https://en.wikipedia.org/wiki/Levenshtein_distance>) for more info on error
  rates. The error rate for the entire partition will be calculated as the total number of
  insertions, deletions, and substitutions made in all transcriptions divided by the sum
  of lengths of reference transcriptions.
  
  Error rates are printed as ratios, not by "percentage."
  
  While convenient and accurate, this script has very few features. Consider pairing the
  command "torch-token-data-dir-to-trn" with sclite
  (http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm) instead.
  
  Many tasks will ignore some tokens (e.g. silences) or collapse others (e.g. phones).
  Please consult a standard recipe (such as those in Kaldi http://kaldi-asr.org/) before
  performing these computations.
  
  positional arguments:
    dir                   If the 'hyp' argument is not specified, this is the
                          parent directory of two subdirectories, 'ref/' and
                          'hyp/', which contain the reference and hypothesis
                          transcripts, respectively. If the '--hyp' argument is
                          specified, this is the reference transcript directory
    hyp                   The hypothesis transcript directory
    out                   Where to print the error rate to. Defaults to stdout
  
  optional arguments:
    -h, --help            show this help message and exit
    --id2token ID2TOKEN   A file containing mappings from unique IDs to tokens
                          (e.g. words or phones). Each line has the format "<id>
                          <token>". The flag "--swap" can be used to swap the
                          expected ordering (i.e. to "<token> <id>")
    --replace REPLACE     A file containing pairs of elements per line. The
                          first is the element to replace, the second what to
                          replace it with. If '--id2token' is specified, the
                          file should contain tokens. If '--id2token' is not
                          specified, the file should contain IDs (integers).
                          This is processed before '--ignore'
    --ignore IGNORE       A file containing a whitespace-delimited list of
                          elements to ignore in both the reference and
                          hypothesis transcripts. If '--id2token' is specified,
                          the file should contain tokens. If '--id2token' is not
                          specified, the file should contain IDs (integers).
                          This is processed after '--replace'
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --swap                If set, swaps the order of the key and value in
                          token/id mapping
    --warn-missing        If set, warn and exclude any utterances that are
                          missing either a reference or hypothesis transcript.
                          The default is to error
    --distances           If set, return the average distance per utterance
                          instead of the total errors over the number of
                          reference tokens
    --per-utt             If set, return lines of ``<utt_id> <error_rate>``
                          denoting the per-utterance error rates instead of the
                          average
    --batch-size BATCH_SIZE
                          The number of error rates to compute at once. Reduce
                          if you run into memory errors
    --quiet               Suppress warnings which arise from edit distance
                          computations
    --costs INS DEL SUB   The costs of an insertion, deletion, and substitution,
                          respectively
    --nist-costs          Use NIST (sclite, score) default costs for insertions,
                          deletions, and substitutions (3/3/4)

ctm-to-torch-token-data-dir
---------------------------

::

  usage: ctm-to-torch-token-data-dir [-h] [--file-prefix FILE_PREFIX]
                                     [--file-suffix FILE_SUFFIX] [--swap]
                                     [--unk-symbol UNK_SYMBOL]
                                     [--num-workers NUM_WORKERS]
                                     [--mp-chunk-size MP_CHUNK_SIZE]
                                     [--skip-frame-times | --feat-sizing | --frame-shift-ms FRAME_SHIFT_MS]
                                     [--wc2utt WC2UTT | --utt2wc UTT2WC]
                                     ctm token2id dir
  
  Convert a NIST "ctm" file to a SpectDataSet token data dir
  
  A "ctm" file is a transcription file with token alignments (a.k.a. a time-marked
  conversation file) used in the sclite
  (http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>) toolkit. Here is the
  format
  
      utt_1 A 0.2 0.1 hi
      utt_1 A 0.3 1.0 there  ;; comment
      utt_2 A 0.0 1.0 next
      utt_3 A 0.1 0.4 utterance
  
  Where the first number specifies the token start time (in seconds) and the second the
  duration.
  
  This command reads in a "ctm" file and writes its contents as token sequences compatible
  with the "ref/" directory of a SpectDataSet. See the command
  "get-torch-spect-data-dir-info" for more info about a SpectDataSet directory.
  
  positional arguments:
    ctm                   The "ctm" file to read token segments from
    token2id              A file containing mappings from tokens (e.g. words or
                          phones) to unique IDs. Each line has the format
                          "<token> <id>". The flag "--swap" can be used to swap
                          the expected ordering (i.e. to "<id> <token>")
    dir                   The directory to store token sequences to. If the
                          directory does not exist, it will be created
  
  optional arguments:
    -h, --help            show this help message and exit
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --swap                If set, swaps the order of the key and value in
                          token/id mapping
    --unk-symbol UNK_SYMBOL
                          If set, will map out-of-vocabulary tokens to this
                          symbol
    --num-workers NUM_WORKERS
                          The number of workers to spawn to process the data. 0
                          is serial. Defaults to the CPU count
    --mp-chunk-size MP_CHUNK_SIZE
                          The number of utterances that a multiprocessing worker
                          will process at once. Impacts speed and memory
                          consumption.
    --skip-frame-times    If true, will store token tensors of shape (R,)
                          instead of (R, 3), foregoing segment start and end
                          times.
    --feat-sizing         If true, will store token tensors of shape (R, 1)
                          instead of (R, 3), foregoing segment start and end
                          times (which trn does not have). The extra dimension
                          will allow data in this directory to be loaded as
                          features in a SpectDataSet.
    --frame-shift-ms FRAME_SHIFT_MS
                          The number of milliseconds that have passed between
                          consecutive frames. Used to convert between time in
                          seconds and frame index. If your features are the raw
                          samples, set this to 1000 / sample_rate_hz
    --wc2utt WC2UTT       A file mapping wavefile name and channel combinations
                          (e.g. 'utt_1 A') to utterance IDs. Each line of the
                          file has the format '<wavefile_name> <channel>
                          <utt_id>'. If neither '--wc2utt' nor '--utt2wc' has
                          been specied, the wavefile name will be treated as the
                          utterance ID
    --utt2wc UTT2WC       A file mapping utterance IDs to wavefile name and
                          channel combinations (e.g. 'utt_1 A'). Each line of
                          the file has the format '<utt_id> <wavefile_name>
                          <channel>'. If neither '--wc2utt' nor '--utt2wc' has
                          been specied, the wavefile name will be treated as the
                          utterance ID

get-torch-spect-data-dir-info
-----------------------------

::

  usage: get-torch-spect-data-dir-info [-h] [--file-prefix FILE_PREFIX]
                                       [--file-suffix FILE_SUFFIX]
                                       [--feat-subdir FEAT_SUBDIR]
                                       [--ali-subdir ALI_SUBDIR]
                                       [--ref-subdir REF_SUBDIR]
                                       [--strict | --fix [N]]
                                       dir [out_file]
  
  Write info about the specified SpectDataSet data dir
  
  NOTE: additional keys (6, 8-10) have been added since pydrobert-pytorch v0.3.0. In
  addition, validation now allows for empty reference segments.
  
  A torch SpectDataSet data dir is of the form
  
      dir/
          feat/
              <file_prefix><utt1><file_suffix>
              <file_prefix><utt2><file_suffix>
              ...
          [ali/
              <file_prefix><utt1><file_suffix>
              <file_prefix><utt1><file_suffix>
              ...
          ]
          [ref/
              <file_prefix><utt1><file_suffix>
              <file_prefix><utt1><file_suffix>
              ...
          ]
  
  Where "feat/" contains float tensors of shape (T, F), where T is the number of frames
  (variable) and F is the number of filters (fixed). "ali/" if there, contains long
  tensors of shape (T,) indicating the appropriate per-frame class labels (likely pdf-ids
  for discriminative training in an DNN-HMM). "ref/", if there, contains long tensors of
  shape (R, 3) indicating a sequence of reference tokens where element indexed by "[i, 0]"
  is a token id, "[i, 1]" is the inclusive start frame of the token (or a negative value
  if unknown), and "[i, 2]" is the exclusive end frame of the token. Token sequences may
  instead be of shape (R,) if no segment times are available in the corpus.
  
  This command writes the following space-delimited key-value pairs to an output file in
  sorted order:
  
  1.  "max_ali_class", the maximum inclusive class id found over "ali/"
       (if available, -1 if not).
  2.  "max_ref_class", the maximum inclussive class id found over "ref/"
       (if available, -1 if not).
  3.  "num_utterances", the total number of listed utterances.
  4.  "num_filts", F.
  5.  "total_frames", the sum of T over the data dir.
  6.  "total_tokens", the sum of R over the data dir (if available, -1 if not).
  7.  "count_<i>", the number of instances of the class "<i>" that appear in "ali/"
      (if available).
  8.  "segs_<i>". The number of segments of the class "<i>" that appear in "ali/"
      (if available). A segment of "<i>" is a maximal run of instances of "<i>" which
      appear sequentially in an alignment. For example, the alignment "0 1 0 1 1 1" would
      have "count_0 = 2" and "count_1 = 4", but "segs_0 = segs_1 = 2".
  9.  "rcount_<i>", the total number of frames reference tokens with type index "<i>"
      occupy according to the segment boundaries listed in the sequences in "ref/" (if
      available). If any token sequence containing index "<i>" does not provide segment
      boundaries (or "<i>" never occurs), "rcount_<i>" is set to "-1".
  10. "rsegs_<i>", the total number of segments (i.e. tokens) with type index "<i>"
      that appear in "ref/" (if available).
  
  If "max_ali_class" was found (>= 0), all key/value pairs for "count_0-<max_ali_class>"
  and "segs_0-<max_ali_class>" will be specified in the file, even if they aren't found
  in the directory. Indices "<i>" will be left-padded with zeros so that keys are sorted
  in increasing index. The same holds for "max_ref_class", "rcount_<i>", and "rsegs_<i>".
  
  In an invalid data directory, the stored key/value pairs are not guaranteed to be
  correct. Passing the "--strict" flag will validate the directory first. Passing "--fix"
  instead will validate the directory and fix any small issues. See the function
  "validate_spect_data_set" in the pydrobert.torch.data Python module for more
  information on the validation process.
  
  Note that the output can be parsed as a Kaldi (http://kaldi-asr.org/) text table of
  integers.
  
  positional arguments:
    dir                   The torch data directory
    out_file              The file to write to. If unspecified, stdout
  
  optional arguments:
    -h, --help            show this help message and exit
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --feat-subdir FEAT_SUBDIR
                          Subdirectory where features are stored.
    --ali-subdir ALI_SUBDIR
                          Subdirectory where per-frame alignments are stored.
    --ref-subdir REF_SUBDIR
                          Subdirectory where reference token sequences are
                          stored.
    --strict              If set, validate the data directory before collecting
                          info. The process is described in
                          pydrobert.torch.data.validate_spect_data_set
    --fix [N]             If set, validate the data directory before collecting
                          info, potentially fixing small errors in the
                          directory. An optional integer argument controls the
                          cropping threshold for ali/ and ref/ (defaults to 1).
                          The process is described in
                          pydrobert.torch.validate_spect_data_set.

print-torch-ali-data-dir-length-moments
---------------------------------------

::

  usage: print-torch-ali-data-dir-length-moments [-h] [--precision PRECISION]
                                                 [--bessel] [--std]
                                                 [--exclude-ids EXCLUDE_IDS [EXCLUDE_IDS ...]]
                                                 [--file-prefix FILE_PREFIX]
                                                 [--file-suffix FILE_SUFFIX]
                                                 [--num-workers NUM_WORKERS]
                                                 [--mp-chunk-size MP_CHUNK_SIZE]
                                                 dir [out]
  
  Compute the mean and variance of segment lengths from an ali data dir
  
  A segment in an "ali/" directory tensor is a maximal sequence of frames with the same
  id. This command computes the mean and variance of segment lengths, printing them on one
  line as
  
      <mean> (<var>)
  
  The input to this command is the "ali/" subdirectory of the SpectDataSet, not its root.
  
  See the command "get-torch-spect-data-dir-info" for more info about a SpectDataSet
  directory.
  
  positional arguments:
    dir                   The ali/ dir (input)
    out                   Where to print statistics. Defaults to stdout
  
  optional arguments:
    -h, --help            show this help message and exit
    --precision PRECISION
                          Precision with which to print stats
    --bessel              Perform Bessel correction on the variance estimate
    --std                 Print standard deviation instead of variance
    --exclude-ids EXCLUDE_IDS [EXCLUDE_IDS ...]
                          If specified, segments with ali ids in this list will
                          be excluded fromcounts
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --num-workers NUM_WORKERS
                          The number of workers to spawn to process the data. 0
                          is serial. Defaults to the CPU count
    --mp-chunk-size MP_CHUNK_SIZE
                          The number of utterances that a multiprocessing worker
                          will process at once. Impacts speed and memory
                          consumption.

print-torch-ref-data-dir-length-moments
---------------------------------------

::

  usage: print-torch-ref-data-dir-length-moments [-h] [--strict | --quiet]
                                                 [--precision PRECISION]
                                                 [--bessel] [--std]
                                                 [--exclude-ids EXCLUDE_IDS [EXCLUDE_IDS ...]]
                                                 [--file-prefix FILE_PREFIX]
                                                 [--file-suffix FILE_SUFFIX]
                                                 [--num-workers NUM_WORKERS]
                                                 [--mp-chunk-size MP_CHUNK_SIZE]
                                                 dir [out]
  
  Compute the mean and variance of segment lengths from an ali data dir
  
  A segment in an "ali/" directory tensor is a maximal sequence of frames with the same
  id. This command computes the mean and variance of segment lengths, printing them on one
  line as
  
      <mean> (<var>)
  
  The input to this command is the "ali/" subdirectory of the SpectDataSet, not its root.
  
  See the command "get-torch-spect-data-dir-info" for more info about a SpectDataSet
  directory.
  
  positional arguments:
    dir                   The ref/ dir (input)
    out                   Where to print statistics. Defaults to stdout
  
  optional arguments:
    -h, --help            show this help message and exit
    --strict              Error when boundary info is not available
    --quiet               Suppress warnings about missing boundary info
    --precision PRECISION
                          Precision with which to print stats
    --bessel              Perform Bessel correction on the variance estimate
    --std                 Print standard deviation instead of variance
    --exclude-ids EXCLUDE_IDS [EXCLUDE_IDS ...]
                          If specified, segments with token ids in this list
                          will be excluded fromcounts
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --num-workers NUM_WORKERS
                          The number of workers to spawn to process the data. 0
                          is serial. Defaults to the CPU count
    --mp-chunk-size MP_CHUNK_SIZE
                          The number of utterances that a multiprocessing worker
                          will process at once. Impacts speed and memory
                          consumption.

subset-torch-spect-data-dir
---------------------------

::

  usage: subset-torch-spect-data-dir [-h] [--copy | --symlink]
                                     (--utt-list UTTID [UTTID ...] | --utt-list-file PATH | --first-n N | --first-ratio R | --last-n N | --last-ratio R | --shortest-n N | --shortest-ratio R | --longest-n N | --longest-ratio R | --rand-n N | --rand-ratio R)
                                     [--only] [--seed SEED]
                                     [--feat-subdir FEAT_SUBDIR]
                                     [--ali-subdir ALI_SUBDIR]
                                     [--ref-subdir REF_SUBDIR]
                                     [--file-prefix FILE_PREFIX]
                                     [--file-suffix FILE_SUFFIX]
                                     [--num-workers NUM_WORKERS]
                                     [--mp-chunk-size MP_CHUNK_SIZE]
                                     src dest
  
  Make a new SpectDataDir from a subset of utterances of another
  
  This command determines a set of utterances via a flag, then hard links all files in the
  "feat/", "ali/" and "ref/" subdirectories matching the utterance id to in the "src"
  directory to the "dest" directory.
  
  See the command "get-torch-spect-data-dir-info" for more info about a SpectDataSet
  directory.
  
  positional arguments:
    src                   The directory to extract from
    dest                  The directory to extract to
  
  optional arguments:
    -h, --help            show this help message and exit
    --copy                Copy extracted files (instead of hard link)
    --symlink             Symlink extracted files (instead of hard link).
                          Symlinks will be relative to the destination.
    --utt-list UTTID [UTTID ...]
                          Extract the utterances listed directly after this flag
    --utt-list-file PATH  Extract the utterances listed in the passed file, one-
                          per-line
    --first-n N           Extract this number of utterances listed first by id
    --first-ratio R       Extract this ratio of utterances (rounding down)
                          listed first by id
    --last-n N            Extract this number of utterances listed last by id
    --last-ratio R        Extract this ratio of utterances (rounding down)
                          listed last by id
    --shortest-n N        Extract this number of utterances listed first by
                          increasing length, then by id
    --shortest-ratio R    Extract this ratio of utterances listed first by
                          increasing length, then by id
    --longest-n N         Extract this number of utterances listed first by
                          decreasing length, then by id
    --longest-ratio R     Extract this ratio of utterances listed first by
                          decreasing length, then by id
    --rand-n N            Extract this number of utterances listed randomly
    --rand-ratio R        Extract this ratio of utterances listed randomly
    --only                If set, extract only the data directly stored in 'src'
    --seed SEED           Seed used in --rand-* flags for determinism. If
                          unspecified, non-deterministic
    --feat-subdir FEAT_SUBDIR
                          Subdirectory where features are stored.
    --ali-subdir ALI_SUBDIR
                          Subdirectory where per-frame alignments are stored.
    --ref-subdir REF_SUBDIR
                          Subdirectory where reference token sequences are
                          stored.
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --num-workers NUM_WORKERS
                          The number of workers to spawn to process the data. 0
                          is serial. Defaults to the CPU count
    --mp-chunk-size MP_CHUNK_SIZE
                          The number of utterances that a multiprocessing worker
                          will process at once. Impacts speed and memory
                          consumption.
  
  Available utterances to extract are determined by the contents of the "feat/"
  subdirectory, unless "--only" was specified. Any extra or missing utterances in "ali/"
  and "ref/" will be ignored.
  
  If "--utt-list" or "--utt-list-file" is chosen, this command ignores any missing
  utterances.
  
  When a criterion involves extracting some number of utterances which exceeds the total
  number of utterances, that total is extracted instead.
  
  Ratios are rounded down to the nearest utterance.
  
  Sorting by id is performed according to python's sort method, i.e. by locale.
  
  When "--only" is paired with "--shortest-*" or "--longest-*", "src" is assumed to also
  be the directory to extract lengths from. Otherwise it's "feat/".
  
  This command has a similar functionality to Kaldi's (https://github.com/kaldi-asr)
  subset_data_dir.sh script, but defaults to hard links for cross-compatibility.

textgrids-to-torch-token-data-dir
---------------------------------

::

  usage: textgrids-to-torch-token-data-dir [-h] [--file-prefix FILE_PREFIX]
                                           [--file-suffix FILE_SUFFIX] [--swap]
                                           [--unk-symbol UNK_SYMBOL]
                                           [--num-workers NUM_WORKERS]
                                           [--mp-chunk-size MP_CHUNK_SIZE]
                                           [--textgrid-suffix TEXTGRID_SUFFIX]
                                           [--fill-symbol FILL_SYMBOL]
                                           [--skip-frame-times | --feat-sizing | --frame-shift-ms FRAME_SHIFT_MS]
                                           [--tier-name TIER_ID | --tier-idx TIER_ID]
                                           tg_dir token2id dir
  
  Convert a directory of TextGrid files into a SpectDataSet ref/ dir
  
  A "TextGrid" file is a transcription file for a single utterance used by the Praat
  software (https://www.fon.hum.uva.nl/praat/).
  
  This command accepts a directory of TextGrid files
  
      tg_dir/
          <file-prefix>utt_1.<textgrid_suffix>
          <file-prefix>utt_2.<textgrid_suffix>
          ...
  
  and writes each file as a separate token sequence compatible with the "ref/" directory
  of a SpectDataSet. If the extracted tier is an IntervalTier, the start and end points
  will be saved with each token. If a TextTier (PointTier), the start and end points of
  each segment will be identified with the point.
  
  See the command "get-torch-spect-data-dir-info" for more info about a SpectDataSet
  directory.
  
  positional arguments:
    tg_dir                The directory containing the TextGrid files
    token2id              A file containing mappings from tokens (e.g. words or
                          phones) to unique IDs. Each line has the format
                          "<token> <id>". The flag "--swap" can be used to swap
                          the expected ordering (i.e. to "<id> <token>")
    dir                   The directory to store token sequences to. If the
                          directory does not exist, it will be created
  
  optional arguments:
    -h, --help            show this help message and exit
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --swap                If set, swaps the order of the key and value in
                          token/id mapping
    --unk-symbol UNK_SYMBOL
                          If set, will map out-of-vocabulary tokens to this
                          symbol
    --num-workers NUM_WORKERS
                          The number of workers to spawn to process the data. 0
                          is serial. Defaults to the CPU count
    --mp-chunk-size MP_CHUNK_SIZE
                          The number of utterances that a multiprocessing worker
                          will process at once. Impacts speed and memory
                          consumption.
    --textgrid-suffix TEXTGRID_SUFFIX
                          The file suffix in tg_dir indicating a TextGrid file.
    --fill-symbol FILL_SYMBOL
                          If set, unlabelled intervals in the TextGrid files
                          will be assigned this symbol. Relevant only if a point
                          grid.
    --skip-frame-times    If true, will store token tensors of shape (R,)
                          instead of (R, 3), foregoing segment start and end
                          times.
    --feat-sizing         If true, will store token tensors of shape (R, 1)
                          instead of (R, 3), foregoing segment start and end
                          times (which trn does not have). The extra dimension
                          will allow data in this directory to be loaded as
                          features in a SpectDataSet.
    --frame-shift-ms FRAME_SHIFT_MS
                          The number of milliseconds that have passed between
                          consecutive frames. Used to convert between time in
                          seconds and frame index. If your features are the raw
                          samples, set this to 1000 / sample_rate_hz
    --tier-name TIER_ID   The name of the tier to extract.
    --tier-idx TIER_ID    The index of the tier to extract.

torch-ali-data-dir-to-torch-token-data-dir
------------------------------------------

::

  usage: torch-ali-data-dir-to-torch-token-data-dir [-h]
                                                    [--file-prefix FILE_PREFIX]
                                                    [--file-suffix FILE_SUFFIX]
                                                    [--num-workers NUM_WORKERS]
                                                    [--mp-chunk-size MP_CHUNK_SIZE]
                                                    ali_dir ref_dir
  
  Convert an ali/ dir to a ref/ dir
  
  This command converts a "ali/" directory from a SpectDataSet to an "ref/" directory.
  The former contains frame-wise alignments; the latter contains token sequences. The
  frame-wise labels are set to the token ids.
  
  To construct the token sequence, the alignment sequence is partitioned into segments,
  each segment corresponding to the longest contiguous span of the same frame-wise label.
  
  See the command "get-torch-spect-data-dir-info" for more info SpectDataSet directories.
  
  positional arguments:
    ali_dir               The frame alignment data directory (input)
    ref_dir               The token sequence data directory (output)
  
  optional arguments:
    -h, --help            show this help message and exit
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --num-workers NUM_WORKERS
                          The number of workers to spawn to process the data. 0
                          is serial. Defaults to the CPU count
    --mp-chunk-size MP_CHUNK_SIZE
                          The number of utterances that a multiprocessing worker
                          will process at once. Impacts speed and memory
                          consumption.

torch-spect-data-dir-to-wds
---------------------------

::

  usage: torch-spect-data-dir-to-wds [-h] [--file-prefix FILE_PREFIX]
                                     [--file-suffix FILE_SUFFIX]
                                     [--feat-subdir FEAT_SUBDIR]
                                     [--ali-subdir ALI_SUBDIR]
                                     [--ref-subdir REF_SUBDIR] [--shard]
                                     [--max-samples-per-shard MAX_SAMPLES_PER_SHARD]
                                     [--max-size-per-shard MAX_SIZE_PER_SHARD]
                                     dir tar_path
  
  Convert a SpectDataSet to a WebDataset
      
  A torch SpectDataSet data dir is of the form
  
      dir/
          feat/
              <file_prefix><utt1><file_suffix>
              <file_prefix><utt2><file_suffix>
              ...
          [ali/
              <file_prefix><utt1><file_suffix>
              <file_prefix><utt1><file_suffix>
              ...
          ]
          [ref/
              <file_prefix><utt1><file_suffix>
              <file_prefix><utt1><file_suffix>
              ...
          ]
  
  Where "feat/" contains float tensors of shape (N, F), where N is the number of
  frames (variable) and F is the number of filters (fixed). "ali/" if there, contains
  long tensors of shape (N,) indicating the appropriate class labels (likely pdf-ids
  for discriminative training in an DNN-HMM). "ref/", if there, contains long tensors
  of shape (R, 3) indicating a sequence of reference tokens where element indexed by
  "[i, 0]" is a token id, "[i, 1]" is the inclusive start frame of the token (or a
  negative value if unknown), and "[i, 2]" is the exclusive end frame of the token.
  
  This command converts the data directory into a tar file to be used as a
  WebDataset (https://github.com/webdataset/webdataset), whose contents are files
  
      <utt1>.feat.pth
      [<utt1>.ali.pth]
      [<utt1>.ref.pth]
      <utt2>.feat.pth
      [<utt2>.ali.pth]
      [<utt2>.ref.pth]
      ...
  
  holding tensors with the same interpretation as above.
  
  This command does not require WebDataset to be installed.
  
  positional arguments:
    dir                   The torch data directory
    tar_path              The path to store files to
  
  optional arguments:
    -h, --help            show this help message and exit
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --feat-subdir FEAT_SUBDIR
                          Subdirectory where features are stored.
    --ali-subdir ALI_SUBDIR
                          Subdirectory where per-frame alignments are stored.
    --ref-subdir REF_SUBDIR
                          Subdirectory where reference token sequences are
                          stored.
    --shard               Split samples among multiple tar files. 'tar_path'
                          will be extended with a suffix '.x', where x is the
                          shard number.
    --max-samples-per-shard MAX_SAMPLES_PER_SHARD
                          If sharding ('--shard' is specified), dictates the
                          number of samples in each file.
    --max-size-per-shard MAX_SIZE_PER_SHARD
                          If sharding ('--shard' is specified), dictates the
                          maximum size in bytes of each file.

torch-token-data-dir-to-ctm
---------------------------

::

  usage: torch-token-data-dir-to-ctm [-h] [--file-prefix FILE_PREFIX]
                                     [--file-suffix FILE_SUFFIX] [--swap]
                                     [--frame-shift-ms FRAME_SHIFT_MS]
                                     [--wc2utt WC2UTT | --utt2wc UTT2WC | --channel CHANNEL]
                                     dir id2token ctm
  
  Convert a SpectDataSet token data directory to a NIST "ctm" file
  
  A "ctm" file is a transcription file with token alignments (a.k.a. a time-marked
  conversation file) used in the sclite
  (http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm) toolkit. Here is the
  format::
  
      utt_1 A 0.2 0.1 hi
      utt_1 A 0.3 1.0 there  ;; comment
      utt_2 A 0.0 1.0 next
      utt_3 A 0.1 0.4 utterance
  
  Where the first number specifies the token start time (in seconds) and the second the
  duration.
  
  This command scans the contents of a directory like "ref/" in a SpectDataSet and
  converts each such file into a transcription. Every token in a given transcription must
  have information about its duration. Each such transcription is then written to the
  "ctm" file. See the command "get-torch-spect-data-dir-info" for more info about a
  SpectDataSet directory.
  
  positional arguments:
    dir                   The directory to read token sequences from
    id2token              A file containing mappings from unique IDs to tokens
                          (e.g. words or phones). Each line has the format "<id>
                          <token>". The flag "--swap" can be used to swap the
                          expected ordering (i.e. to "<token> <id>")
    ctm                   The "ctm" file to write token segments to
  
  optional arguments:
    -h, --help            show this help message and exit
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --swap                If set, swaps the order of the key and value in
                          token/id mapping
    --frame-shift-ms FRAME_SHIFT_MS
                          The number of milliseconds that have passed between
                          consecutive frames. Used to convert between time in
                          seconds and frame index. If your features are the raw
                          samples, set this to 1000 / sample_rate_hz
    --wc2utt WC2UTT       A file mapping wavefile name and channel combinations
                          (e.g. 'utt_1 A') to utterance IDs. Each line of the
                          file has the format '<wavefile_name> <channel>
                          <utt_id>'.
    --utt2wc UTT2WC       A file mapping utterance IDs to wavefile name and
                          channel combinations (e.g. 'utt_1 A'). Each line of
                          the file has the format '<utt_id> <wavefile_name>
                          <channel>'.
    --channel CHANNEL     If neither "--wc2utt" nor "--utt2wc" is specified,
                          utterance IDs are treated as wavefile names and are
                          given the value of this flag as a channel

torch-token-data-dir-to-textgrids
---------------------------------

::

  usage: torch-token-data-dir-to-textgrids [-h] (--feat-dir FEAT_DIR | --infer)
                                           [--file-prefix FILE_PREFIX]
                                           [--file-suffix FILE_SUFFIX] [--swap]
                                           [--frame-shift-ms FRAME_SHIFT_MS]
                                           [--num-workers NUM_WORKERS]
                                           [--mp-chunk-size MP_CHUNK_SIZE]
                                           [--textgrid-suffix TEXTGRID_SUFFIX]
                                           [--tier-name TIER_NAME]
                                           [--precision PRECISION] [--quiet]
                                           [--force-method {1,2,3}]
                                           ref_dir id2token tg_dir
  
  Convert a SpectDataSet ref/ dir into a directory of TextGrid files
  
  A "TextGrid" file is a transcription file for a single utterance used by the Praat
  software (https://www.fon.hum.uva.nl/praat/).
  
  This command accepts a directory of token sequences compatible with the "ref/"
  directory of a SpectDataSet and outputs a directory of TextGrid files
  
      tg_dir/
          <file-prefix>utt_1.<textgrid_suffix>
          <file-prefix>utt_2.<textgrid_suffix>
          ...
  
  A token sequence ref is a tensor of shape either (R, 3) or just (R,). The latter has no
  segment information and is just the tokens. The former contains triples "tok, start,
  end", where "tok" is the token id, "start" is the starting frame inclusive, and "end" is
  the ending frame exclusive. A negative value for either boundary means the information
  is not available.
  
  By default, this command tries to save the sequence as a tier preserving as much
  information in the token sequence as possible in a consistent way. The following methods
  are attempted in order:
  
  1. If ref is of shape (R, 3), all segments boundaries are available, and all segments
     are of nonzero length, the sequence will be saved as an IntervalTier containing
     segment boundaries.
  2. If ref is of shape (R, 3) and either the start or end boundary is available for every
     token, the sequence will be saved as a TextTier (PointTier) with points set to the
     available boundary (with precedence going to the greater).
  3. Otherwise, the token sequence is written as an interval tier with a single segment
     spanning the recording and containing all tokens.
  
  In addition, the total length of the features in frames must be determined. Either the
  flag "--feat-dir" must be specified in order to get the length directly from the feature
  sequences, or "--infer" must be specified. The latter guesses the length to be the
  maximum end boundary of the token sequence available, or 0 (with a warning if "--quiet"
  unset) if none are.
  
  Note that Praat usually works either with point data or with intervals which
  collectively partition the audio. It can parse TextGrid files with non-contiguous
  intervals, but they are rendered strangely.
  
  See the command "get-torch-spect-data-dir-info" for more info about a SpectDataSet
  directory.
  
  positional arguments:
    ref_dir               The token sequence data directory (input)
    id2token              A file containing mappings from unique IDs to tokens
                          (e.g. words or phones). Each line has the format "<id>
                          <token>". The flag "--swap" can be used to swap the
                          expected ordering (i.e. to "<token> <id>")
    tg_dir                The TextGrid directory (output)
  
  optional arguments:
    -h, --help            show this help message and exit
    --feat-dir FEAT_DIR   Path to features
    --infer               Infer lengths based on maximum segment boundaries
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --swap                If set, swaps the order of the key and value in
                          token/id mapping
    --frame-shift-ms FRAME_SHIFT_MS
                          The number of milliseconds that have passed between
                          consecutive frames. Used to convert between time in
                          seconds and frame index. If your features are the raw
                          samples, set this to 1000 / sample_rate_hz
    --num-workers NUM_WORKERS
                          The number of workers to spawn to process the data. 0
                          is serial. Defaults to the CPU count
    --mp-chunk-size MP_CHUNK_SIZE
                          The number of utterances that a multiprocessing worker
                          will process at once. Impacts speed and memory
                          consumption.
    --textgrid-suffix TEXTGRID_SUFFIX
                          The file suffix in tg_dir indicating a TextGrid file.
    --tier-name TIER_NAME
                          The name to save the tier with
    --precision PRECISION
                          Precision with which to save floating point values in
                          TextGrid files
    --quiet               If set, suppresses warnings when lengths cannot be
                          determined
    --force-method {1,2,3}
                          Force a specific method of writing to TextGrid (1-3
                          above). Not enough information will lead to an error.

torch-token-data-dir-to-torch-ali-data-dir
------------------------------------------

::

  usage: torch-token-data-dir-to-torch-ali-data-dir [-h] [--feat-dir FEAT_DIR]
                                                    [--file-prefix FILE_PREFIX]
                                                    [--file-suffix FILE_SUFFIX]
                                                    [--num-workers NUM_WORKERS]
                                                    [--mp-chunk-size MP_CHUNK_SIZE]
                                                    ref_dir ali_dir
  
  Convert a ref/ dir to an ali/ dir
  
  This command converts a "ref/" directory from a SpectDataSet to an "ali/" directory. The
  former contains sequences of tokens; the latter contains frame-wise alignments. The
  token ids are set to the frame-wise labels.
  
  A reference token sequence "ref" partitions a frame sequence of length T if
  
  1. ref is of shape (R, 3), with R > 1 and all ref[r, 1:] >= 0 (it contains segment
     boundaries).
  2. ref[0, 1] = 0 (it starts at frame 0).
  3. for all 0 <= r < R - 1, ref[r, 2] = ref[r + 1, 1] (boundaries contiguous).
  4. ref[R - 1, 2] = T (it ends after T frames).
  
  When ref partitions the frame sequence, it can be converted into a per-frame alignment
  tensor "ali" of shape (T,), where ref[r, 1] <= t < ref[r, 2] implies ali[t] = ref[r, 0].
  
  WARNING! This operation is potentially destructive: a per-frame alignment cannot
  distinguish between two of the same token next to one another and one larger token.
  
  See the command "get-torch-spect-data-dir-info" for more info SpectDataSet directories.
  
  positional arguments:
    ref_dir               The token sequence data directory (input)
    ali_dir               The frame alignment data directory (output)
  
  optional arguments:
    -h, --help            show this help message and exit
    --feat-dir FEAT_DIR   The feature data directory. While not necessary for
                          the conversion, specifying this directory will allow
                          the total number of frames in each utterance to be
                          checked by loading the associated feature matrix.
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --num-workers NUM_WORKERS
                          The number of workers to spawn to process the data. 0
                          is serial. Defaults to the CPU count
    --mp-chunk-size MP_CHUNK_SIZE
                          The number of utterances that a multiprocessing worker
                          will process at once. Impacts speed and memory
                          consumption.

torch-token-data-dir-to-trn
---------------------------

::

  usage: torch-token-data-dir-to-trn [-h] [--file-prefix FILE_PREFIX]
                                     [--file-suffix FILE_SUFFIX] [--swap]
                                     [--num-workers NUM_WORKERS]
                                     dir id2token trn
  
  Convert a SpectDataSet token data dir to a NIST trn file
  
  A "trn" file is the standard transcription file without alignment information used
  in the sclite (http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm)
  toolkit. It has the format
  
      here is a transcription (utterance_a)
      here is another (utterance_b)
  
  This command scans the contents of a directory like "ref/" in a SpectDataSeet and
  converts each such file into a transcription. Each such transcription is then
  written to a "trn" file. See the command "get-torch-spect-data-dir-info" for more
  info about a SpectDataSet directory.
  
  positional arguments:
    dir                   The directory to read token sequences from
    id2token              A file containing mappings from unique IDs to tokens
                          (e.g. words or phones). Each line has the format "<id>
                          <token>". The flag "--swap" can be used to swap the
                          expected ordering (i.e. to "<token> <id>")
    trn                   The "trn" file to write transcriptions to
  
  optional arguments:
    -h, --help            show this help message and exit
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --swap                If set, swaps the order of the key and value in
                          token/id mapping
    --num-workers NUM_WORKERS
                          The number of workers to spawn to process the data. 0
                          is serial. Defaults to the CPU count

trn-to-torch-token-data-dir
---------------------------

::

  usage: trn-to-torch-token-data-dir [-h] [--alt-handler {error,first}]
                                     [--file-prefix FILE_PREFIX]
                                     [--file-suffix FILE_SUFFIX] [--swap]
                                     [--unk-symbol UNK_SYMBOL]
                                     [--num-workers NUM_WORKERS]
                                     [--mp-chunk-size MP_CHUNK_SIZE]
                                     [--skip-frame-times | --feat-sizing]
                                     trn token2id dir
  
  Convert a NIST "trn" file to the specified SpectDataSet data dir
  
  A "trn" file is the standard transcription file without alignment information used in
  the sclite (http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm) toolkit. It
  has the format
  
      here is a transcription (utterance_a)
      here is another (utterance_b)
  
  This command reads in a "trn" file and writes its contents as token sequences compatible
  with the "ref/" directory of a SpectDataSet. See the command
  "get-torch-spect-data-dir-info" for more info about a SpectDataSet directory.
  
  positional arguments:
    trn                   The input trn file
    token2id              A file containing mappings from tokens (e.g. words or
                          phones) to unique IDs. Each line has the format
                          "<token> <id>". The flag "--swap" can be used to swap
                          the expected ordering (i.e. to "<id> <token>")
    dir                   The directory to store token sequences to. If the
                          directory does not exist, it will be created
  
  optional arguments:
    -h, --help            show this help message and exit
    --alt-handler {error,first}
                          How to handle transcription alternates. If "error",
                          error if the "trn" file contains alternates. If
                          "first", always treat the alternate as canon
    --file-prefix FILE_PREFIX
                          The file prefix indicating a torch data file
    --file-suffix FILE_SUFFIX
                          The file suffix indicating a torch data file
    --swap                If set, swaps the order of the key and value in
                          token/id mapping
    --unk-symbol UNK_SYMBOL
                          If set, will map out-of-vocabulary tokens to this
                          symbol
    --num-workers NUM_WORKERS
                          The number of workers to spawn to process the data. 0
                          is serial. Defaults to the CPU count
    --mp-chunk-size MP_CHUNK_SIZE
                          The number of utterances that a multiprocessing worker
                          will process at once. Impacts speed and memory
                          consumption.
    --skip-frame-times    If true, will store token tensors of shape (R,)
                          instead of (R, 3), foregoing segment start and end
                          times.
    --feat-sizing         If true, will store token tensors of shape (R, 1)
                          instead of (R, 3), foregoing segment start and end
                          times (which trn does not have). The extra dimension
                          will allow data in this directory to be loaded as
                          features in a SpectDataSet.

