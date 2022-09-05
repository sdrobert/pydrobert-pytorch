# Copyright 2022 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings

from typing import Container, Optional, Set, Tuple, Union

import torch
import param

import pydrobert.torch.config as config

from ._feats import MeanVarianceNormalization, FeatureDeltas


class SpectDataParams(param.Parameterized):
    """Parameters for SpectDataSet"""

    subset_ids = param.List(
        [],
        class_=str,
        bounds=None,
        doc="A list of utterance ids. If non-empty, the data set will be "
        "restricted to these utterances",
    )
    sos = param.Integer(
        None,
        doc="A special symbol used to indicate the start of a sequence "
        "in reference and hypothesis transcriptions. If set, `sos` will be "
        "prepended to every reference transcription on read",
    )
    eos = param.Integer(
        None,
        doc="A special symbol used to indicate the end of a sequence in "
        "reference and hypothesis transcriptions. If set, `eos` will be "
        "appended to every reference transcription on read",
    )
    delta_order = param.Integer(
        0,
        bounds=(0, None),
        softbounds=(0, 2),
        doc="Order of delta coefficients to apply to the spectral features. 0 means "
        "none",
    )
    do_mvn = param.Boolean(
        False,
        doc="Whether to perform mean-variance normalization on the spectral features. "
        "If mean and variance statistics are not passed to the dataset/dataloader as "
        "an argument, feature coefficients will be unit normalized per utterance",
    )

    @classmethod
    def get_tunable(cls) -> Set[str]:
        return {"delta_order", "do_mvn"}

    @classmethod
    def suggest_params(
        cls, trial, base=None, only: Container[str] = None, prefix: str = ""
    ):
        """Populate a parameterized instance with values from trial"""
        params = cls() if base is None else base
        if only is None:
            only = cls.get_tunable()
        if "delta_order" in only:
            bounds = params.param.params()["delta_order"].get_soft_bounds()
            params.delta_order = trial.suggest_int(prefix + "delta_order", *bounds)
        if "do_mvn" in only:
            params.do_mvn = trial.suggest_categorical(prefix + "do_mvn", [True, False])

        return params


class SpectDataSet(torch.utils.data.Dataset):
    """Accesses spectrographic filter data stored in a data directory

    Parameters
    ----------
    data_dir
        A path to the data directory
    file_prefix
        The prefix that indicates that the file counts toward the data set
    file_suffix
        The suffix that indicates that the file counts toward the data set
    warn_on_missing
        If ``ali/`` or ``ref/`` exist, there's a mismatch between the
        utterances in the directories, and `warn_on_missing` is :obj:`True`, a
        warning will be issued (via ``warnings``) regarding each such mismatch
    subset_ids
        Deprecated. Use params.subset_ids.
    sos
        Deprecated. Use params.sos.
    eos
        Deprecated. Use params.eos.
    feat_subdir, ali_subdir, ref_subdir
        Change the names of the subdirectories under which feats, alignments, and
        references are stored. If `ali_subdir` or `ref_subdir` is :obj:`None`, they will
        not be searched for
    params
        Populates the parameters of this class with the instance. If unset, a new
        `SpectDataParams` instance is initialized.
    feat_mean
        If specified and ``params.do_mvn`` is :obj:`True`, this tensor will be used
        as the mean in mean-variance normalization.
    feat_std
        If specified and ``params.do_mvn`` is :obj:`True`, this tensor will be used
        as the standard deviation in mean-variance normalization.
    suppress_alis : bool
        If :obj:`True`, `ali` will not be yielded, nor will alignment information
        be counted towards the list of utterance ids if available.
    suppress_uttids : bool
        If :obj:`True`, `uttid` will not be yielded.
    tokens_only : bool
        If :obj:`True`, `ref` will drop the segment information if present, always
        yielding tuples of shape ``(R,)``.
    
    Yields
    ------
    tup
        For a given utterance, a tuple:

        1. `feat`, the filter bank data.
        2. `ali` (if `suppress_ali` is :obj:`False`), frame-level alignments or
           :obj:`None` if not available.
        3. `ref`, a sequence of reference tokens or :obj:`None` if not available.
        4. `uttid` (if `suppress_uttid` is :obj:`False`), the string representing the
           utterance id.

    Examples
    --------
    Creating a spectral data directory with random data

    >>> import os
    >>> data_dir = 'data'
    >>> os.makedirs(data_dir + '/feat', exist_ok=True)
    >>> os.makedirs(data_dir + '/ali', exist_ok=True)
    >>> os.makedirs(data_dir + '/ref', exist_ok=True)
    >>> num_filts, min_frames, max_frames, min_ref, max_ref = 40, 10, 20, 3, 10
    >>> num_ali_classes, num_ref_classes = 100, 2000
    >>> for utt_idx in range(30):
    >>>     num_frames = torch.randint(
    ...         min_frames, max_frames + 1, (1,)).long().item()
    >>>     num_tokens = torch.randint(
    ...         min_ref, max_ref + 1, (1,)).long().item()
    >>>     feats = torch.randn(num_frames, num_filts)
    >>>     torch.save(feats, data_dir + '/feat/{:02d}.pt'.format(utt_idx))
    >>>     ali = torch.randint(num_ali_classes, (num_frames,)).long()
    >>>     torch.save(ali, data_dir + '/ali/{:02d}.pt'.format(utt_idx))
    >>>     # usually these would be sorted by order in utterance. Negative
    >>>     # values represent "unknown" for start end end frames
    >>>     ref_tokens = torch.randint(num_tokens, (num_tokens,))
    >>>     ref_starts = torch.randint(1, num_frames // 2, (num_tokens,))
    >>>     ref_ends = 2 * ref_starts
    >>>     ref = torch.stack([ref_tokens, ref_starts, ref_ends], -1).long()
    >>>     torch.save(ref, data_dir + '/ref/{:02d}.pt'.format(utt_idx))

    Accessing individual elements in a spectral data directory

    >>> data = SpectDataSet('data')
    >>> data[0]  # random access feat, ali, ref
    >>> for feat, ali, ref in data:  # iterator
    >>>     pass

    Writing evaluation data back to the directory

    >>> data = SpectDataSet('data')
    >>> num_ali_classes, num_ref_classes, min_ref, max_ref = 100, 2000, 3, 10
    >>> num_frames = data[3][0].shape[0]
    >>> # pdfs (or more accurately, pms) are likelihoods of classes over data
    >>> # per frame, used in hybrid models. Usually logits
    >>> pdf = torch.randn(num_frames, num_ali_classes)
    >>> data.write_pdf(3, pdf)  # will share name with data.utt_ids[3]
    >>> # both refs and hyps are sequences of tokens, such as words or phones,
    >>> # with optional frame alignments
    >>> num_tokens = torch.randint(min_ref, max_ref, (1,)).long().item()
    >>> hyp = torch.full((num_tokens, 3), INDEX_PAD_VALUE).long()
    >>> hyp[..., 0] = torch.randint(num_ref_classes, (num_tokens,))
    >>> data.write_hyp('special', hyp)  # custom name
    """

    data_dir: str
    file_prefix: str
    file_suffix: str
    feat_subdir: str
    ali_subdir: Optional[str]
    ref_subdir: Optional[str]
    params: SpectDataParams
    has_ali: bool
    has_ref: bool
    suppress_alis: bool
    suppress_uttids: bool
    tokens_only: bool
    utt_ids: Tuple[str, ...]
    transform: Optional[torch.nn.Module]

    def __init__(
        self,
        data_dir: str,
        file_prefix: str = "",
        file_suffix: str = ".pt",
        warn_on_missing: bool = True,
        subset_ids: Optional[Set[str]] = None,
        sos: Optional[int] = None,
        eos: Optional[int] = None,
        feat_subdir: str = "feat",
        ali_subdir: Optional[str] = "ali",
        ref_subdir: Optional[str] = "ref",
        params: Optional[SpectDataParams] = None,
        feat_mean: Optional[torch.Tensor] = None,
        feat_std: Optional[torch.Tensor] = None,
        suppress_alis: bool = None,
        suppress_uttids: bool = True,
        tokens_only: bool = None,
    ):
        super(SpectDataSet, self).__init__()
        if suppress_alis is None:
            warnings.warn(
                "A future version of pydrobert-pytorch will set suppress_alis=True by "
                "default. To ensure future compatibility, set suppress_alis=False",
                DeprecationWarning,
                stacklevel=2,
            )
            suppress_alis = False
        if tokens_only is None:
            warnings.warn(
                "A future version of pydrobert-pytorch will set tokens_only=True by "
                "default. To ensure future compatibility, set tokens_only=False",
                DeprecationWarning,
                stacklevel=2,
            )
            tokens_only = False
        self.data_dir = data_dir
        self.feat_subdir = feat_subdir
        self.ali_subdir = ali_subdir
        self.ref_subdir = ref_subdir
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.suppress_alis = suppress_alis
        self.suppress_uttids = suppress_uttids
        self.tokens_only = tokens_only
        if params is None:
            params = SpectDataParams()
        self.params = params
        self.sos = params.sos
        self.eos = params.eos
        if sos is not None:
            warnings.warn(
                "Specifying sos by keyword argument is deprecated. Use params instead",
                DeprecationWarning,
            )
            self.sos = sos
        if eos is not None:
            warnings.warn(
                "Specifying eos by keyword argument is deprecated. Use params instead",
                DeprecationWarning,
            )
            self.eos = eos
        if ali_subdir and not suppress_alis:
            self.has_ali = os.path.isdir(os.path.join(data_dir, ali_subdir))
        else:
            self.has_ali = False
        if ref_subdir:
            self.has_ref = os.path.isdir(os.path.join(data_dir, ref_subdir))
        else:
            self.has_ref = False
        if self.has_ali:
            self.has_ali = any(
                x.startswith(file_prefix) and x.endswith(file_suffix)
                for x in os.listdir(os.path.join(data_dir, ali_subdir))
            )
        if self.has_ref:
            self.has_ref = any(
                x.startswith(file_prefix) and x.endswith(file_suffix)
                for x in os.listdir(os.path.join(data_dir, ref_subdir))
            )
        if subset_ids is None:
            subset_ids = set(params.subset_ids)
        else:
            warnings.warn(
                "passing subset_ids to the dataset directly is deprecated. Set "
                "params.subset_ids instead.",
                DeprecationWarning,
                2,
            )
        self.utt_ids = tuple(
            sorted(self.find_utt_ids(warn_on_missing, subset_ids=subset_ids))
        )
        transforms = []
        if params.do_mvn:
            transforms.append(MeanVarianceNormalization(mean=feat_mean, std=feat_std))
        if params.delta_order:
            transforms.append(FeatureDeltas(order=params.delta_order))
        if len(transforms):
            transform = torch.nn.Sequential(*transforms)
            if config.USE_JIT:
                transform = torch.jit.script(transform)
            self.transform = transform
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.utt_ids)

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, str, None], ...]:
        return self.get_utterance_tuple(idx)

    def find_utt_ids(
        self, warn_on_missing: bool, subset_ids: Set[str] = set()
    ) -> Set[str]:
        """Returns a set of all utterance ids from data_dir"""
        neg_fsl = -len(self.file_suffix)
        if not neg_fsl:
            neg_fsl = None
        fpl = len(self.file_prefix)
        utt_ids = set(
            x[fpl:neg_fsl]
            for x in os.listdir(os.path.join(self.data_dir, self.feat_subdir))
            if x.startswith(self.file_prefix) and x.endswith(self.file_suffix)
        )
        if subset_ids:
            utt_ids &= subset_ids
        if self.has_ali:
            ali_utt_ids = set(
                x[fpl:neg_fsl]
                for x in os.listdir(os.path.join(self.data_dir, self.ali_subdir))
                if x.startswith(self.file_prefix) and x.endswith(self.file_suffix)
            )
            if subset_ids:
                ali_utt_ids &= subset_ids
            if warn_on_missing:
                for utt_id in utt_ids.difference(ali_utt_ids):
                    warnings.warn("Missing ali for uttid: '{}'".format(utt_id))
                for utt_id in ali_utt_ids.difference(utt_ids):
                    warnings.warn("Missing feat for uttid: '{}'".format(utt_id))
        if self.has_ref:
            ref_utt_ids = set(
                x[fpl:neg_fsl]
                for x in os.listdir(os.path.join(self.data_dir, self.ref_subdir))
                if x.startswith(self.file_prefix) and x.endswith(self.file_suffix)
            )
            if subset_ids:
                ref_utt_ids &= subset_ids
            if warn_on_missing:
                for utt_id in utt_ids.difference(ref_utt_ids):
                    warnings.warn("Missing ref for uttid: '{}'".format(utt_id))
                for utt_id in ref_utt_ids.difference(utt_ids):
                    warnings.warn("Missing feat for uttid: '{}'".format(utt_id))
        if self.has_ali:
            utt_ids &= ali_utt_ids
        if self.has_ref:
            utt_ids &= ref_utt_ids
        return utt_ids

    def get_utterance_tuple(
        self, idx: int
    ) -> Tuple[Union[torch.Tensor, str, None], ...]:
        utt_id = self.utt_ids[idx]
        feat = torch.load(
            os.path.join(
                self.data_dir,
                self.feat_subdir,
                self.file_prefix + utt_id + self.file_suffix,
            )
        )
        if self.transform is not None:
            feat = self.transform(feat)
        if self.has_ali:
            ali = torch.load(
                os.path.join(
                    self.data_dir,
                    self.ali_subdir,
                    self.file_prefix + utt_id + self.file_suffix,
                )
            )
        else:
            ali = None
        if self.has_ref:
            ref = torch.load(
                os.path.join(
                    self.data_dir,
                    self.ref_subdir,
                    self.file_prefix + utt_id + self.file_suffix,
                )
            )
            D = ref.ndim
            if self.tokens_only and D == 2:
                ref, D = ref[..., 0], 1
            if self.sos is not None:
                if D == 2:
                    sos_sym = torch.full_like(ref[0], -1)
                    sos_sym[0] = self.sos
                    ref = torch.cat([sos_sym.unsqueeze(0), ref], 0)
                else:
                    ref = torch.cat([torch.full_like(ref[:1], self.sos), ref], 0)
            if self.eos is not None:
                if D == 2:
                    eos_sym = torch.full_like(ref[0], -1)
                    eos_sym[0] = self.eos
                    ref = torch.cat([ref, eos_sym.unsqueeze(0)], 0)
                else:
                    ref = torch.cat([ref, torch.full_like(ref[:1], self.eos)], 0)

        else:
            ref = None
        if self.suppress_alis:
            if self.suppress_uttids:
                return feat, ref
            else:
                return feat, ref, utt_id
        elif self.suppress_uttids:
            return feat, ali, ref
        else:
            return feat, ali, ref, utt_id

    def write_pdf(
        self, utt: Union[str, int], pdf: torch.Tensor, pdfs_dir: Optional[str] = None
    ) -> None:
        """Write a pdf tensor to the data directory

        This method writes a pdf matrix to the directory `pdfs_dir` with the name
        ``<file_prefix><utt><file_suffix>``

        Parameters
        ----------
        utt
            The name of the utterance to write. If an integer is specified,
            `utt` is assumed to index an utterance id specified in
            ``self.utt_ids``
        pdf
            The tensor to write. It will be converted to a CPU float tensor using the
            command ``pdf.cpu().float()``
        pdfs_dir
            The directory pdfs are written to. If :obj:`None`, it will be set to
            ``self.data_dir + '/pdfs'``
        """
        if isinstance(utt, int):
            utt = self.utt_ids[utt]
        if pdfs_dir is None:
            pdfs_dir = os.path.join(self.data_dir, "pdfs")
        os.makedirs(pdfs_dir, exist_ok=True)
        torch.save(
            pdf.cpu().float(),
            os.path.join(pdfs_dir, self.file_prefix + utt + self.file_suffix),
        )

    def write_hyp(
        self, utt: Union[str, int], hyp: torch.Tensor, hyp_dir: Optional[str] = None
    ) -> None:
        """Write hypothesis tensor to the data directory

        This method writes a sequence of hypothesis tokens to the directory `hyp_dir`
        with the name ``<file_prefix><utt><file_suffix>``

        If the ``sos`` attribute of this instance is not :obj:`None`, any tokens in
        `hyp` matching it will be considered the start of the sequence, so every symbol
        including and before the last instance will be removed from the utterance before
        saving

        If the ``eos`` attribute of this instance is not :obj:`None`, any tokens in
        `hyp` matching it will be considered the end of the sequence, so every symbol
        including and after the first instance will be removed from the utterance before
        saving

        Parameters
        ----------
        utt
            The name of the utterance to write. If an integer is specified,
            `utt` is assumed to index an utterance id specified in
            ``self.utt_ids``
        hyp
            The tensor to write. Either of shape ``(R,)`` or ``(R, 3)``. It will be
            converted to a long tensor using the command ``hyp.cpu().long()``
        hyp_dir
            The directory pdfs are written to. If :obj:`None`, it will be set to
            ``self.data_dir + '/hyp'``
        """
        if isinstance(utt, int):
            utt = self.utt_ids[utt]
        if hyp_dir is None:
            hyp_dir = os.path.join(self.data_dir, "hyp")
        if not os.path.isdir(hyp_dir):
            os.makedirs(hyp_dir)
        hyp = hyp.cpu().long()
        if self.sos is not None:
            if hyp.dim() == 1:
                sos_idxs = torch.nonzero(hyp.eq(self.sos), as_tuple=False)
            else:
                sos_idxs = torch.nonzero(hyp[:, 0].eq(self.sos), as_tuple=False)
            if sos_idxs.numel():
                sos_idx = sos_idxs[-1].item()
                hyp = hyp[sos_idx + 1 :]
        if self.eos is not None:
            if hyp.dim() == 1:
                eos_idxs = torch.nonzero(hyp.eq(self.eos), as_tuple=False)
            else:
                eos_idxs = torch.nonzero(hyp[:, 0].eq(self.eos), as_tuple=False)
            if eos_idxs.numel():
                eos_idx = eos_idxs[0].item()
                hyp = hyp[:eos_idx]
        torch.save(
            hyp.cpu().long(),
            os.path.join(hyp_dir, self.file_prefix + utt + self.file_suffix),
        )


def validate_spect_data_set(data_set: SpectDataSet, fix: bool = False) -> None:
    """Validate SpectDataSet data directory

    The data directory is valid if the following conditions are observed

    1. All tensors are on the CPU
    2. All features are tensor instances of the same dtype
    3. All features have two dimensions
    4. All features have the same size second dimension
    5. If alignments are present
       1. All alignments are long tensor instances
       2. All alignments have one dimension
       3. Features and alignments have the same size first axes for a given utterance id
    6. If reference sequences are present

       1. All references are long tensor instances
       2. All alignments have the same number of dimensions: either 1 or 2
       3. If 2-dimensional

          1. The second dimension has length 3
          2. For the start and end points of a reference token, ``r[i, 1:]``, either
             both of them are negative (indicating no alignment), or ``0 <= r[i, 1] <
             r[i, 2] <= T``, where ``T`` is the number of frames in the utterance. We do
             not enforce that tokens be non-overlapping

    Raises a :class:`ValueError` if a condition is violated.

    If `fix` is :obj:`True`, the following changes to the data will be permitted instead
    of raising an error. Any of these changes will be warned of using :mod:`warnings`
    and then written back to disk.

    1. Any CUDA tensors will be converted into CPU tensors
    2. A reference or alignment of bytes or 32-bit integers can be upcast to long
       tensors.
    3. A reference token with only a start or end bound (but not both) will have the
       existing one removed.
    4. A reference token with an exclusive boundary exceeding the number of features by
       one will be decreased by one. This is only possible if the exclusive end remains
       above the inclusive start.
    """
    num_filts = None
    ref_is_2d = None
    feat_dtype = None
    for idx in range(len(data_set.utt_ids)):
        fn = data_set.utt_ids[idx] + data_set.file_suffix
        feat, ali, ref = data_set.get_utterance_tuple(idx)
        write_back = False
        prefix = "'{}' (index {})".format(fn, idx)
        dir_ = os.path.join(data_set.data_dir, data_set.feat_subdir)
        prefix_ = "{} in '{}'".format(prefix, dir_)
        if not isinstance(feat, torch.Tensor) or feat_dtype not in {None, feat.dtype}:
            raise ValueError(
                "{} is not a tensor or not the same tensor type as previous features"
                "".format(prefix_)
            )
        if feat.device.type == "cuda":
            msg = "{} is a cuda tensor".format(prefix_)
            if fix:
                warnings.warn(msg)
                feat = feat.cpu()
                write_back = True
            else:
                raise ValueError(msg)
        feat_dtype = feat.dtype
        if feat.dim() != 2:
            raise ValueError("{} does not have two dimensions".format(prefix_))
        if num_filts is None:
            num_filts = feat.size(1)
        elif feat.size(1) != num_filts:
            raise ValueError(
                "{} has second dimension of size {}, which "
                "does not match prior utterance ('{}') size of {}".format(
                    prefix_,
                    feat.size(1),
                    data_set.utt_ids[idx - 1] + data_set.file_suffix,
                    num_filts,
                )
            )
        if write_back:
            torch.save(feat, os.path.join(dir_, fn))
            write_back = False
        if ali is not None:
            dir_ = os.path.join(data_set.data_dir, data_set.ali_subdir)
            prefix_ = "{} in '{}'".format(prefix, dir_)
            if isinstance(ali, torch.Tensor) and ali.device.type == "cuda":
                msg = "{} is a cuda tensor".format(prefix_)
                if fix:
                    warnings.warn(msg + ". Converting")
                    ali = ali.cpu()
                    write_back = True
                else:
                    raise ValueError(msg)
            if not isinstance(ali, torch.LongTensor):
                msg = "{} is not a long tensor".format(prefix_)
                if fix and isinstance(
                    ali,
                    (
                        torch.ByteTensor,
                        torch.CharTensor,
                        torch.ShortTensor,
                        torch.IntTensor,
                    ),
                ):
                    warnings.warn(msg + ". Converting")
                    ali = ali.long()
                    write_back = True
                else:
                    raise ValueError(msg)
            if len(ali.shape) != 1:
                raise ValueError("{} does not have one dimension".format(prefix_))
            if ali.shape[0] != feat.shape[0]:
                raise ValueError(
                    "{} does not have the same first dimension of"
                    " size ({}) as its companion in '{}' ({})".format(
                        prefix_,
                        feat.shape[0],
                        os.path.join(data_set.data_dir, data_set.ali_subdir),
                        ali.shape[0],
                    )
                )
            if write_back:
                torch.save(ali, os.path.join(dir_, fn))
                write_back = False
        if ref is not None:
            dir_ = os.path.join(data_set.data_dir, data_set.ref_subdir)
            prefix_ = "{} in '{}'".format(prefix, dir_)
            if isinstance(ref, torch.Tensor) and ref.device.type == "cuda":
                msg = "{} is a cuda tensor".format(prefix_)
                if fix:
                    warnings.warn(msg + ". Converting")
                    ref = ref.cpu()
                    write_back = True
                else:
                    raise ValueError(msg)
            if not isinstance(ref, torch.LongTensor):
                msg = "{} is not a long tensor".format(prefix_)
                if fix and isinstance(
                    ref,
                    (
                        torch.ByteTensor,
                        torch.CharTensor,
                        torch.ShortTensor,
                        torch.IntTensor,
                    ),
                ):
                    warnings.warn(msg + ". Converting")
                    ref = ref.long()
                    write_back = True
                else:
                    raise ValueError(msg)
            if ref.dim() == 2:
                if ref_is_2d is False:
                    raise ValueError(
                        "{} is 2D. Previous transcriptions were 1D".format(prefix_)
                    )
                ref_is_2d = True
                if ref.size(1) != 3:
                    raise ValueError("{} does not have shape (D, 3)".format(prefix_))
                for idx2, r in enumerate(ref):
                    msg = (
                        "{} has a reference token (index {}) with invalid boundaries"
                        "".format(prefix_, idx2)
                    )
                    if not (r[1] < 0 and r[2] < 0):
                        if r[1] < 0 or r[2] < 0:
                            if fix:
                                warnings.warn(msg + ". Removing unpaired boundary")
                                r[1:] = -1
                                write_back = True
                            else:
                                raise ValueError(msg)
                        elif r[2] > feat.size(0):
                            if fix and r[2] - 1 == feat.size(0) and r[1] < r[2] - 1:
                                warnings.warn(msg + ". Reducing upper bound by 1")
                                r[2] -= 1
                                write_back = True
                            else:
                                raise ValueError(msg)
                        elif r[1] >= r[2]:
                            raise ValueError(msg)

            elif ref.dim() == 1:
                if ref_is_2d is True:
                    raise ValueError(
                        "{} is 1D. Previous transcriptions were 2D".format(prefix_)
                    )
                ref_is_2d = False
            else:
                raise ValueError("{} is not 1D nor 2D".format(prefix_))
            if write_back:
                torch.save(ref, os.path.join(dir_, fn))


def extract_window(
    feat: torch.Tensor, frame_idx: int, left: int, right: int, reverse: bool = False
) -> torch.Tensor:
    """Slice the feature matrix to extract a context window

    Parameters
    ----------
    feat
        Of shape ``(T, F)``, where ``T`` is the time/frame axis, and ``F``
        is the frequency axis
    frame_idx
        The "center frame" ``0 <= frame_idx < T``
    left
        The number of frames in the window to the left (before) the center
        frame. Any frames below zero are edge-padded
    right
        The number of frames in the window to the right (after) the center
        frame. Any frames above ``T`` are edge-padded
    reverse
        If :obj:`True`, flip the window along the time/frame axis

    Returns
    -------
    window : torch.Tensor
        Of shape ``(1 + left + right, F)``
    """
    T, F = feat.shape
    if frame_idx - left < 0 or frame_idx + right + 1 > T:
        win_size = 1 + left + right
        window = feat.new(win_size, F)
        left_pad = max(left - frame_idx, 0)
        right_pad = max(frame_idx + right + 1 - T, 0)
        window[left_pad : win_size - right_pad] = feat[
            max(0, frame_idx - left) : frame_idx + right + 1
        ]
        if left_pad:
            window[:left_pad] = feat[0]
        if right_pad:
            window[-right_pad:] = feat[-1]
    else:
        window = feat[frame_idx - left : frame_idx + right + 1]
    if reverse:
        window = torch.flip(window, [0])
    return window


class ContextWindowDataParams(SpectDataParams):
    """Parameters for ContextWindowDataSet

    This implements the :class:`pydrobert.param.optuna.TunableParameterized`
    interface
    """

    # context windows are more model parameters than data parameters, but
    # we're going to extract them as part of the data loading process, which
    # is easily parallelized by the DataLoader
    context_left = param.Integer(
        4,
        bounds=(0, None),
        softbounds=(3, 8),
        doc="How many frames to the left of (before) the current frame are "
        "included when determining the class of the current frame",
    )
    context_right = param.Integer(
        4,
        bounds=(0, None),
        softbounds=(3, 8),
        doc="How many frames to the right of (after) the current frame are "
        "included when determining the class of the current frame",
    )
    reverse = param.Boolean(
        False,
        doc="Whether to reverse each context window along the time/frame dimension",
    )

    @classmethod
    def get_tunable(cls):
        """Returns a set of tunable parameters"""
        return super().get_tunable() | {"context_left", "context_right", "reverse"}

    @classmethod
    def suggest_params(cls, trial, base=None, only=None, prefix=""):
        """Populate a parameterized instance with values from trial"""
        params = cls() if base is None else base
        super().suggest_params(trial, params, only, prefix)
        if only is None:
            only = cls._tunable
        pdict = params.param.params()
        for name in ("context_left", "context_right"):
            if name in only:
                softbounds = pdict[name].get_soft_bounds()
                setattr(params, name, trial.suggest_int(prefix + name, *softbounds))
        if "reverse" in only:
            params.reverse = trial.suggest_categorical(
                prefix + "reverse", [True, False]
            )
        return params


class ContextWindowDataSet(SpectDataSet):
    """SpectDataSet, extracting fixed-width windows over the utterance

    Like a :class:`SpectDataSet`, but replaces the `feat` tensor with `window`, which
    runs a sliding window over the frame dimension of `feat`.

    Parameters
    ----------
    data_dir
    left
        Deprecated
    right
        Deprecated
    file_prefix
    file_suffix
    warn_on_missing
    subset_ids
        Deprecated
    feat_subdir, ali_subdir
    reverse
        Deprecated
    params
    suppress_uttids
        If :obj:`True`, `uttid` will not be yielded.

    Yields
    ------
    tup
        For a given utterance, a tuple:

        1. `window`, windowed spectral features of shape ``(T, 1 + left + right, F)``,
           where the ``T`` axis indexes the so-called center frame and the ``1 + left +
           right`` axis contains frame vectors (size ``F``) including the center frame
           and the those to the `left` and `right`.
        2. `ali`, window-level alignments, or :obj:`None` if not available.
        3. `uttid` (if `suppress_uttid` is :obj:`False`), the string representing the
           utterance id.
        
    Examples
    --------
    >>> # see 'SpectDataSet' to set up data directory
    >>> data = ContextWindowDataSet('data')
    >>> data[0]  # random access returns (window, ali) pairs
    >>> for window, ali in data:
    >>>     pass  # so does the iterator
    >>> data.get_utterance_tuple(3)  # gets the original (feat, ali) pair
    """

    params: ContextWindowDataParams

    def __init__(
        self,
        data_dir: str,
        left: Optional[int] = None,
        right: Optional[int] = None,
        file_prefix: str = "",
        file_suffix: str = ".pt",
        warn_on_missing: bool = True,
        subset_ids: Optional[Set[str]] = None,
        feat_subdir: str = "feat",
        ali_subdir: Optional[str] = "ali",
        reverse: Optional[bool] = None,
        params: Optional[ContextWindowDataParams] = None,
        feat_mean: Optional[torch.Tensor] = None,
        feat_std: Optional[torch.Tensor] = None,
        suppress_uttids: bool = True,
    ):
        if params is None:
            params = ContextWindowDataParams()
        if left is not None:
            warnings.warn(
                "Specifying left by argument is deprecated. Please use "
                "params.context_left",
                DeprecationWarning,
            )
            self.left = left
        else:
            self.left = params.context_left
        if right is not None:
            warnings.warn(
                "Specifying right by argument is deprecated. Please use "
                "params.context_right",
                DeprecationWarning,
            )
            self.right = right
        else:
            self.right = params.context_right
        if reverse is not None:
            warnings.warn(
                "Specifying reverse by argument is deprecated. Please use "
                "params.reverse"
            )
            self.reverse = reverse
        else:
            self.reverse = params.reverse
        super().__init__(
            data_dir,
            file_prefix,
            file_suffix,
            warn_on_missing,
            subset_ids,
            None,
            None,
            feat_subdir,
            ali_subdir,
            "ref",
            params,
            feat_mean,
            feat_std,
            False,
            suppress_uttids,
            False,
        )

    def get_utterance_tuple(self, idx) -> Tuple[Union[torch.Tensor, str, None], ...]:
        tup = super().get_utterance_tuple(idx)
        if self.suppress_uttids:
            return tup[:2]
        else:
            return tup[:2] + tup[-1:]

    def get_windowed_utterance(
        self, idx: int
    ) -> Tuple[Union[torch.Tensor, str, None], ...]:
        feat, ali = super().get_utterance_tuple(idx)[:2]
        num_frames, num_filts = feat.shape
        window = torch.empty(num_frames, 1 + self.left + self.right, num_filts)
        for center_frame in range(num_frames):
            window[center_frame] = extract_window(
                feat, center_frame, self.left, self.right, reverse=self.reverse
            )
        if self.suppress_uttids:
            return window, ali
        else:
            return window, ali, self.utt_ids[idx]

    def __getitem__(self, idx: int):
        return self.get_windowed_utterance(idx)

