Command-Line Interface
======================

get-torch-spect-data-dir-info
-----------------------------

::

  usage: get-torch-spect-data-dir-info [-h] [--file-prefix FILE_PREFIX]
                                       [--file-suffix FILE_SUFFIX]
                                       [--feat-subdir FEAT_SUBDIR]
                                       [--ali-subdir ALI_SUBDIR]
                                       [--ref-subdir REF_SUBDIR]
                                       [--strict | --fix]
                                       dir [out_file]
  
  Write info about the specified SpectDataSet data dir
  
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
  
      This command writes the following space-delimited key-value pairs to an
      output file in sorted order:
  
      1. "max_ali_class", the maximum inclusive class id found over "ali/"
         (if available, -1 if not)
      2. "max_ref_class", the maximum inclussive class id found over "ref/"
         (if available, -1 if not)
      3. "num_utterances", the total number of listed utterances
      4. "num_filts", F
      5. "total_frames", the sum of N over the data dir
      6. "count_<i>", the number of instances of the class "<i>" that appear in "ali/"
         (if available). If "count_<i>" is a valid key, then so are "count_<0 to i>".
         "count_<i>" is left-padded with zeros to ensure that the keys remain in the same
         order in the table as the class indices.  The maximum i will be equal to
         the value of "max_ali_class"
  
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
                          Subdirectory where features are stored
    --ali-subdir ALI_SUBDIR
                          Subdirectory where alignments are stored
    --ref-subdir REF_SUBDIR
                          Subdirectory where reference token sequences are
                          stored
    --strict              If set, validate the data directory before collecting
                          info. The process is described in
                          pydrobert.torch.data.validate_spect_data_set
    --fix                 If set, validate the data directory before collecting
                          info, potentially fixing small errors in the
                          directory. The process is described in
                          pydrobert.torch.validate_spect_data_set

trn-to-torch-token-data-dir
---------------------------

::

  usage: trn-to-torch-token-data-dir [-h] [--alt-handler {error,first}]
                                     [--file-prefix FILE_PREFIX]
                                     [--file-suffix FILE_SUFFIX] [--swap]
                                     [--unk-symbol UNK_SYMBOL]
                                     [--num-workers NUM_WORKERS]
                                     [--chunk-size CHUNK_SIZE]
                                     [--skip-frame-times | --feat-sizing]
                                     trn token2id dir
  
  Convert a NIST "trn" file to the specified SpectDataSet data dir
  
      A "trn" file is the standard transcription file without alignment information used
      in the sclite (http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm)
      toolkit. It has the format
  
          here is a transcription (utterance_a)
          here is another (utterance_b)
  
      This command reads in a "trn" file and writes its contents as token sequences
      compatible with the "ref/" directory of a SpectDataSet. See the command
      "get-torch-spect-data-dir-info" for more info about a SpectDataSet directory
      
  
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
    --chunk-size CHUNK_SIZE
                          The number of lines that a worker will process at
                          once. Impacts speed and memory consumption.
    --skip-frame-times    If true, will store token tensors of shape (R,)
                          instead of (R, 3), foregoing segment start and end
                          times (which trn does not have).
    --feat-sizing         If true, will store token tensors of shape (R, 1)
                          instead of (R, 3), foregoing segment start and end
                          times (which trn does not have). The extra dimension
                          will allow data in this directory to be loaded as
                          features in a SpectDataSet.

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

ctm-to-torch-token-data-dir
---------------------------

::

  usage: ctm-to-torch-token-data-dir [-h] [--file-prefix FILE_PREFIX]
                                     [--file-suffix FILE_SUFFIX] [--swap]
                                     [--frame-shift-ms FRAME_SHIFT_MS]
                                     [--wc2utt WC2UTT | --utt2wc UTT2WC]
                                     [--unk-symbol UNK_SYMBOL]
                                     ctm token2id dir
  
  Convert a NIST "ctm" file to a SpectDataSet token data dir
  
      A "ctm" file is a transcription file with token alignments (a.k.a. a time-marked
      conversation file) used in the sclite
      (http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm>) toolkit. Here is
      the format
  
          utt_1 A 0.2 0.1 hi
          utt_1 A 0.3 1.0 there  ;; comment
          utt_2 A 0.0 1.0 next
          utt_3 A 0.1 0.4 utterance
  
      Where the first number specifies the token start time (in seconds) and the second
      the duration.
  
      This command reads in a "ctm" file and writes its contents as token sequences
      compatible with the "ref/" directory of a SpectDataSet. See the command
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
    --frame-shift-ms FRAME_SHIFT_MS
                          The number of milliseconds that have passed between
                          consecutive frames. Used to convert between time in
                          seconds and frame index. If your features are the raw
                          sample, set this to 1000 / sample_rate_hz
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
    --unk-symbol UNK_SYMBOL
                          If set, will map out-of-vocabulary tokens to this
                          symbol

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
  
      Where the first number specifies the token start time (in seconds) and the second
      the duration.
  
      This command scans the contents of a directory like "ref/" in a SpectDataSete and
      converts each such file into a transcription. Every token in a given transcription
      must have information about its duration. Each such transcription is then written to
      the "ctm" file. See the command "get-torch-spect-data-dir-info" for more info about
      a SpectDataSet directory.
      
  
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
      pydrobert-pytorch when the insertion, deletion, and substitution costs do not all
      equal 1. Consult the documentation of "pydrobert.torch.functional.error_rate" for
      more information.
  
      This is a very simple script that computes and prints the error rates between the
      "ref/" (reference/gold standard) token sequences and "hyp/" (hypothesis/generated)
      token sequences in a SpectDataSet directory. Consult the Wikipedia article on the
      Levenshtein distance (https://en.wikipedia.org/wiki/Levenshtein_distance>) for more
      info on error rates. The error rate for the entire partition will be calculated as
      the total number of insertions, deletions, and substitutions made in all
      transcriptions divided by the sum of lengths of reference transcriptions.
  
      Error rates are printed as ratios, not by "percentage."
  
      While convenient and accurate, this script has very few features. Consider pairing
      the command "torch-token-data-dir-to-trn" with sclite
      (http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm) instead.
  
      Many tasks will ignore some tokens (e.g. silences) or collapse others (e.g. phones).
      Please consult a standard recipe (such as those in Kaldi http://kaldi-asr.org/)
      before performing these computations.
      
  
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

