# Copyright 2022 Sean Robertson
#
# Code for polyharmonic_spline is converted from tensorflow code
# https://github.com/tensorflow/addons/blob/v0.11.2/tensorflow_addons/image/interpolate_spline.py
# code for sparse_image_warp is derived from tensorflow code, though it's not identical
# https://github.com/tensorflow/addons/blob/v0.11.2/tensorflow_addons/image/sparse_image_warp.py
#
# Which are also Apache 2.0 Licensed:
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

from typing import Any, Optional, Tuple, TYPE_CHECKING, Union

import torch

from ._compat import meshgrid, script, linalg_solve


@script
def _get_tensor_eps(
    x: torch.Tensor,
    eps16: float = torch.finfo(torch.float16).eps,
    eps32: float = torch.finfo(torch.float32).eps,
    eps64: float = torch.finfo(torch.float64).eps,
) -> float:
    if x.dtype == torch.float16:
        return eps16
    elif x.dtype == torch.float32:
        return eps32
    elif x.dtype == torch.float64:
        return eps64
    else:
        raise RuntimeError(f"Expected x to be floating-point, got {x.dtype}")


@script
def _phi(r: torch.Tensor, k: int) -> torch.Tensor:
    if k % 2:
        return r ** k
    else:
        return (r ** k) * (torch.clamp(r, min=_get_tensor_eps(r))).log()


@script
def _apply_interpolation(
    w: torch.Tensor, v: torch.Tensor, c: torch.Tensor, x: torch.Tensor, k: int
) -> torch.Tensor:
    r = torch.cdist(x, c)  # (N, Q, T)
    phi_r = _phi(r, k)  # (N, Q, T)
    phi_r_w = torch.bmm(phi_r, w)  # (N, Q, O)
    x1 = torch.cat([x, torch.ones_like(x[..., :1])], 2)  # (N, Q, I+1)
    x1_v = torch.bmm(x1, v)  # (N, Q, O)
    return phi_r_w + x1_v


@script
def _solve_interpolation(
    c: torch.Tensor, f: torch.Tensor, k: int, reg: float, full: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    # based on
    # https://mathematica.stackexchange.com/questions/65763/understanding-polyharmonic-splines
    # Symbol map (theirs => ours)
    # x,y => c  (N, T, I)
    # A => A    (N, T, T)
    # W => B    (N, T, I+1)
    # v => w    (N, T, O)
    # bb => v   (N, I+1, O)
    # wa => f   (N, T, O)
    r_cc = torch.cdist(c, c)  # (N, T, T)
    A = _phi(r_cc, k)  # (N, T, T)
    if reg > 0.0:
        A = A + torch.eye(A.shape[1], dtype=A.dtype, device=A.device).unsqueeze(0) * reg
    B = torch.cat([c, torch.ones_like(c[..., :1])], 2)  # (N, T, I+1)

    if full:
        # full matrix method (TF)
        ABt = torch.cat([A, B.transpose(1, 2)], 1)  # (N, T+I+1, T)
        zeros = torch.zeros(
            (B.shape[0], B.shape[2], B.shape[2]), device=B.device, dtype=B.dtype
        )
        B0 = torch.cat([B, zeros], 1,)  # (N, T+I+1, I+1)
        ABtB0 = torch.cat([ABt, B0], 2)  # (N, T+I+1, T+I+1)
        zeros = torch.zeros(
            (B.shape[0], B.shape[2], f.shape[2]), device=f.device, dtype=f.dtype
        )
        f0 = torch.cat([f, zeros], 1,)  # (N, T+I+1, O)
        wv = linalg_solve(ABtB0, f0)
        w, v = wv[:, : B.shape[1]], wv[:, B.shape[1] :]
    else:
        # block decomposition
        Ainv = torch.inverse(A)  # (N, T, T)
        Ainv_f = torch.bmm(Ainv, f)  # (N, T, O)
        Ainv_B = torch.bmm(Ainv, B)  # (N, T, I+1)
        Bt = B.transpose(1, 2)  # (N, I+1, T)
        Bt_Ainv_B = torch.bmm(Bt, Ainv_B)  # (N, I+1, I+1)
        Bt_Ainv_f = torch.bmm(Bt, Ainv_f)  # (N, I+1, O)
        v = linalg_solve(Bt_Ainv_B, Bt_Ainv_f)
        Ainv_B_v = torch.bmm(Ainv_B, v)  # (N, T, O)
        w = Ainv_f - Ainv_B_v  # (N, T, O)

    # orthagonality constraints
    # assert torch.allclose(w.sum(1), torch.tensor(0.0, device=w.device)), w.sum()
    # assert torch.allclose(
    #     torch.bmm(w.transpose(1, 2), c), torch.tensor(0.0, device=w.device)
    # ), torch.bmm(w.transpose(1, 2), c).sum()

    return w, v


@script
def polyharmonic_spline(
    train_points: torch.Tensor,
    train_values: torch.Tensor,
    query_points: torch.Tensor,
    order: int,
    regularization_weight: float = 0.0,
    full_matrix: bool = True,
) -> torch.Tensor:
    """Functional version of PolyharmonicSpline
    
    See Also
    --------
    pydrobert.torch.layers.PolyharmonicSpline
        For a description of this function and its parameters
    """
    train_points = train_points.float()
    query_points = query_points.float()

    w, v = _solve_interpolation(
        train_points, train_values, order, regularization_weight, full_matrix
    )

    query_values = _apply_interpolation(w, v, train_points, query_points, order)
    return query_values


class PolyharmonicSpline(torch.nn.Module):
    """Guess values at query points using a learned polyharmonic spline

    A spline estimates a function ``f : points -> values`` from a fixed number of
    training points/knots and the values of ``f`` at those points. It does that by
    solving a series of piecewise linear equations between knots such that the values at
    the knots match the given values (and some additional constraints depending on the
    spline).

    This module is based on the `interpolate_spline
    <https://www.tensorflow.org/addons/api_docs/python/tfa/image/interpolate_spline>`__
    function from Tensorflow, which implements a `Polyharmonic Spline
    <https://en.wikipedia.org/wiki/Polyharmonic_spline>`__. For technical details,
    consult the TF documentation.

    The call signature of this module, once instantiated, is::

        query_values = polyharmonic_spline(
            train_points, train_values, query_points, query_values
        )
    
    `train_points` is tensor of shape ``(N, T, I)`` representing the training
    points/knots for ``N`` different functions. ``N`` is the batch dimension, ``T`` is
    the number of training points, and ``I`` is the size of the vector input to ``f``.
    `train_values` is a float tensor of shape ``(N, T, O)`` of ``f`` evaluated on
    `train_points`. ``O`` is the size of the output vector of ``f``. `query_points` is
    a tensor of shape ``(N, Q, I)`` representing the points you wish to have
    estimates for. ``Q`` is the number of such points. `query_values` is a tensor of
    shape ``(N, Q, O)`` consisting of the values estimated by the spline

    Parameters
    ----------
    order : int
        Order of the spline (> 0). 1 = linear. 2 = thin plate spline.
    regularization_weight : float, optional
        Weight placed on the regularization term. See TF for more info.
    full_matrix : bool, optional
        Whether to solve linear equations via a full concatenated matrix or a block
        decomposition. Setting to :obj:`True` better matches TF and appears to slightly
        improve numerical accuracy at the cost of twice the run time and more memory
        usage.

    Throws
    ------
    RuntimeError
        This module can return a :class`RuntimeError` when no unique spline can be
        estimated. In general, the spline will require at least ``I+1`` non-degenerate
        points (linearly independent). See the Wikipedia entry on splnes for more info.
    """

    __constants__ = ["order", "regularization_weight", "full_matrix"]

    order: int
    regularization_weight: float
    full_matrix: bool

    def __init__(
        self, order: int, regularization_weight: float = 0.0, full_matrix: bool = True
    ):
        super().__init__()
        if order <= 0:
            raise ValueError(f"order must be positive, got {order}")
        self.order = order
        self.regularization_weight = regularization_weight
        self.full_matrix = full_matrix

    def forward(
        self,
        train_points: torch.Tensor,
        train_values: torch.Tensor,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        return polyharmonic_spline(
            train_points,
            train_values,
            query_points,
            self.order,
            self.regularization_weight,
            self.full_matrix,
        )


@script
def _deterimine_pinned_points(k: int, sizes: torch.Tensor) -> torch.Tensor:

    w_max = (sizes[:, :1] - 1).expand(-1, k + 1)  # (N, k+1)
    h_max = (sizes[:, 1:] - 1).expand(-1, k + 1)  # (N, k+1)
    range_ = torch.linspace(
        0.0, 1.0, k + 1, dtype=sizes.dtype, device=sizes.device
    )  # (k+1,)
    w_range = w_max * range_  # (N, k+1)
    h_range = h_max * range_  # (N, k+1)
    zeros = torch.zeros_like(w_range)  # (N, k+1)

    # (0, 0) -> (W - 1, 0) inclusive
    bottom_edge = torch.stack([w_range, zeros], 2)  # (N, k+1, 2)
    # (0, 0) -> (0, H - 1) exclusive
    left_edge = torch.stack([zeros[:, 1:-1], h_range[:, 1:-1]], 2)  # (N, k-1, 2)
    # (0, H - 1) -> (W - 1, H - 1) inclusive
    top_edge = torch.stack([w_range, h_max], 2)  # (N, k+1, 2)
    # (W - 1, 0) -> (W - 1, H - 1) exclusive
    right_edge = torch.stack([w_max[:, 1:-1], h_range[:, 1:-1]], 2)  # (N, k-1, 2)

    return torch.cat([bottom_edge, left_edge, top_edge, right_edge], 1)  # (N, 4k, 2)


@script
def warp_1d_grid(
    src: torch.Tensor,
    flow: torch.Tensor,
    lengths: torch.Tensor,
    max_length: Optional[int] = None,
    interpolation_order: int = 1,
) -> torch.Tensor:
    """Functional version of Warp1DGrid

    See Also
    --------
    pydrobert.torch.layers.Warp1DGrid
        For a description of this function and its parameters
    """
    device = src.device
    N = src.shape[0]
    if max_length is None:
        T = int(math.ceil(lengths.max().item())) if lengths.numel() else 0
    else:
        T = max_length
    src, flow, lengths = src.float(), flow.float(), lengths.float()
    eps = _get_tensor_eps(src)  # the epsilon avoids singular matrices
    src = torch.min(src, lengths - 1).clamp_min(0)
    dst = torch.min(src + flow, lengths - 1).clamp_min(0)
    src = (2.0 * src + 1.0) / T - 1.0
    dst = (2.0 * dst + 1.0) / T - 1.0
    lowers = torch.full((N,), 1 / T - 1 - eps, dtype=torch.float, device=device)
    uppers = (2 * lengths - 1) / T - 1.0 + eps
    src = torch.stack([lowers, src, uppers], 1)  # (N, 3)
    dst = torch.stack([lowers, dst, uppers], 1)  # (N, 3)
    # sparse_grid = (2.0 * src + 1.0) / T - 1.0  # (N,3)
    t = (2.0 * torch.arange(T, device=device) + 1.0) / T - 1.0
    grid = polyharmonic_spline(
        dst.unsqueeze(-1),  # dst (N, 3, 1)
        src.unsqueeze(-1),  # (N, 3, 1)
        t.unsqueeze(0).expand(N, T).unsqueeze(-1),  # (N, T, 1)
        interpolation_order,
    ).squeeze(
        -1
    )  # (N, T)
    return grid


class Warp1DGrid(torch.nn.Module):
    """Interpolate grid values for a dimension of a grid_sample

    This module determines a grid along a single dimension of a signal,
    image, volume, whatever. 

    When instantiated, this method has the signature::

        grid = warp_1d_grid(src, flow, lengths)
    
    `src` is a tensor of shape ``(N,)`` containing source points. `flow` is
    a tensor of shape ``(N,)`` containing corresponding flow fields for `src`.
    `lengths` is a long tensor of shape ``(N,)`` specifying the number of
    valid indices along the dimension in question. The return value is a tensor
    `grid` of shape ``(N, max_length)`` which provides coodinates for one
    dimension of the grid passed to :func:`torch.nn.functional.grid_sample`.
    See the example below.

    Parameters
    ----------
    max_length : int or `None`, optional
        A maximum length to which the grid will be padded. If unspecified, it will be
        taken to be ``lengths.max().ceil()``. If `grid` is being plugged in to
        :func:`grid_sample`, ensure `max_length` matches the size of the dimension of
        the image being warped.
    interpolation_order : int, optional
        The degree of the spline used ot interpolate the grid.

    Notes
    -----
    The return value `grid` assumes `align_corners` has been set to :obj:`False` in
    :func:`grid_sample`.

    The values in `grid` depend on the value of `max_length`. `grid` does not contain
    absolute pixel indices, instead mapping the range ``[0, max_length - 1]`` to the
    real values in ``[-1, 1]``. Therefore, unless `max_length` is set to some fixed
    value, the ``n``-th batch element ``grid[n]`` can differ according to the remaining
    values in `length`. However, the ``n``-th batched image passed to
    :func:`grid_sample` should still be warped in a way (roughly) agnostic to
    surrounding batched images.
    """

    __constants__ = ["max_length", "interpolation_order"]

    interpolation_order: int
    max_length: Optional[int]

    def __init__(self, max_length: Optional[int] = None, interpolation_order: int = 1):
        super().__init__()
        if max_length is not None and max_length < 0:
            raise ValueError("max_length must be non-negative")
        self.max_length = max_length
        self.interpolation_order = interpolation_order

    def extra_repr(self) -> str:
        s = f"interpolation_order={self.interpolation_order}"
        if self.max_length is not None:
            s = f"max_length={self.max_length}, " + s
        return s

    def forward(
        self, src: torch.Tensor, flow: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return warp_1d_grid(
            src, flow, lengths, self.max_length, self.interpolation_order
        )


@script
def dense_image_warp(
    image: torch.Tensor,
    flow: torch.Tensor,
    indexing: str = "hw",
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """Functional version of DenseImageWarp
    
    See Also
    --------
    pydrobert.torch.layers.DenseImageWarp
        For a description of this function and its parameters
    """
    # from tfa.image.dense_image_warp
    # output[n, c, h, w] = image[n, c, h - flow[n, h, w, 0], w - flow[n, h, w, 1]]
    # outside of image uses border

    # from torch.nn.functional.grid_sample
    # output[n, c, h, w] = image[n, c, h, f(grid[n, h, w, 1], H),
    # f(grid[n, h, w, 0], W)]
    # where
    # f(x, X) = ((x + 1) * X - 1) / 2
    # therefore
    # output[n, c, h, w] = image[n, c, ((grid[n, h, w, 1] + 1) * H - 1) / 2,
    #                                  ((grid[n, h, w, 0] + 1) * W - 1) / 2]
    #
    # ((grid[n, h, w, 1] + 1) * H - 1) / 2 = h - flow[n, h, w, 0]
    # grid[n, h, w, 1] = (2 * h - 2 * flow[n, h, w, 0] + 1) / H - 1
    # likewise
    # grid[n, h, w, 0] = (2 * w - 2 * flow[n, h, w, 1] + 1) / W - 1

    flow = flow.float()

    N, C, H, W = image.shape
    h = torch.arange(H, dtype=image.dtype, device=image.device)  # (H,)
    w = torch.arange(W, dtype=image.dtype, device=image.device)  # (W,)
    h, w = meshgrid(h, w)  # (H, W), (H, W)
    if indexing == "hw":
        # grid_sample uses wh sampling, so we flip both the flow and hw along final axis
        hw = torch.stack((w, h), 2).unsqueeze(0)  # (1, H, W, 2)
        flow = flow.flip(-1)
    elif indexing == "wh":
        hw = torch.stack((w, h), 2).unsqueeze(0)  # (1, H, W, 2)
    else:
        raise ValueError("Invalid indexing! must be one of 'wh' or 'hw'")
    HW = torch.tensor([[[[W, H]]]], dtype=image.dtype, device=image.device)  # (1,1,1,2)
    grid = (2 * hw - 2 * flow + 1.0) / HW - 1.0

    return torch.nn.functional.grid_sample(
        image, grid, mode=mode, padding_mode=padding_mode, align_corners=False
    )


class DenseImageWarp(torch.nn.Module):
    """Warp an input image with per-pixel flow vectors

    Once initialized, this module is called with the signature::

        warped = dense_image_warp(image, flow)

    `image` is a float tensor of shape ``(N, C, H, W)``, where ``N`` is the batch
    dimension, ``C`` is the channel dimension, ``H`` is the height dimension, and ``W``
    is the width dimension. `flow` is a float tensor of shape ``(N, H, W, 2)``.
    It returns a new image `warped` of shape ``(N, C, H, W)`` such that

    ::
        warped[n, c, h, w] = image[n, c, h - flow[n, h, w, 0], w - flow[n, h, w, 1]]

    If the reference indices ``h - ...`` and ``w - ...`` are not integers, the value is
    interpolated from the neighboring pixel values.

    This reproduces the functionality of Tensorflow's `dense_image_warp
    <https://www.tensorflow.org/addons/api_docs/python/tfa/image/dense_image_warp>`__,
    except `image` is in ``NCHW`` order instead of ``NHWC`` order. It wraps
    `torch.nn.functional.grid_sample`.

    Warning
    -------
    `flow` is not an optical flow. Please consult the TF documentation for more details.

    Parameters
    ----------
    indexing : {'hw', 'wh'}, optional
        If `indexing` is ``"hw"``, ``flow[..., 0] = h``, the height index, and
        ``flow[..., 1] = w`` is the width index. If ``"wh"``, ``flow[..., 0] = w``
        and ``flow[..., 1] = h``. The default in TF is ``"hw"``, whereas torch's
        `grid_sample` is ``"wh"``
    mode : {'bilinear', 'nearest'}, optional
        The method of interpolation. Either use bilinear interpolation or the nearest
        pixel value. The TF default is ``"bilinear"``
    padding_mode : {"border", "zeros", "reflection"}
        Controls how points outside of the image boundaries are interpreted.
        ``"border"``: copy points at around the border of the image. ``"zero"``:
        use zero-valued pixels. ``"reflection"``: reflect pixels into the image starting
        from the boundaries.
    """

    __constants__ = ["indexing", "mode", "padding_mode"]

    indexing: str
    mode: str
    padding_mode: str

    def __init__(
        self,
        indexing: str = "hw",
        mode: str = "bilinear",
        padding_mode: str = "border",
    ):
        super().__init__()
        if indexing not in {"hw", "wh"}:
            raise ValueError(f"indexing must be either 'hw' or 'wh', got '{indexing}'")
        if mode not in {"bilinear", "nearest"}:
            raise ValueError(
                f"mode must be either 'bilinear' or 'nearest', got '{mode}'"
            )
        if padding_mode not in {"border", "zeros", "reflection"}:
            raise ValueError(
                "padding_mode must be one of 'border', 'zeros', or 'relection', got "
                f"'{padding_mode}'"
            )
        self.indexing = indexing
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return dense_image_warp(
            image, flow, self.indexing, self.mode, self.padding_mode
        )


# N.B. We do this ugly thing so that a trace can be aware of the returned type
# (rather than just "Any")
@script
def _sparse_image_warp_flow(
    image,
    source_points,
    dest_points,
    indexing: str,
    field_interpolation_order: int,
    field_regularization_weight: float,
    field_full_matrix: bool,
    pinned_boundary_points: int,
    dense_interpolation_mode: str,
    dense_padding_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if indexing == "hw":
        source_points = source_points.flip(-1)
        dest_points = dest_points.flip(-1)

    source_points = source_points.float()
    dest_points = dest_points.float()

    N, C, H, W = image.shape
    WH = torch.tensor([[W, H]] * N, dtype=image.dtype, device=image.device)

    M = source_points.shape[1]
    if not M:
        return (
            image,
            torch.zeros((N, H, W, 2), dtype=torch.float, device=image.device),
        )

    if pinned_boundary_points > 0:
        pinned_points = _deterimine_pinned_points(pinned_boundary_points, WH)
        source_points = torch.cat([source_points, pinned_points], 1)  # (N,M',2)
        dest_points = torch.cat([dest_points, pinned_points], 1)  # (N,M+4k=M',2)
        # now just pretend M' was M all along

    H_range = torch.arange(H, dtype=image.dtype, device=image.device)  # (H,)
    W_range = torch.arange(W, dtype=image.dtype, device=image.device)  # (W,)
    h, w = meshgrid(H_range, W_range)  # (H, W), (H, W)
    query_points = torch.stack([w.flatten(), h.flatten()], 1)  # (H * W, 2)

    train_points = dest_points
    train_values = dest_points - source_points
    flow = polyharmonic_spline(
        train_points,
        train_values,
        query_points.unsqueeze(0).expand(N, H * W, 2),
        field_interpolation_order,
        regularization_weight=field_regularization_weight,
        full_matrix=field_full_matrix,
    )

    flow = flow.view(N, H, W, 2)

    warped = dense_image_warp(
        image,
        flow,
        indexing="wh",
        mode=dense_interpolation_mode,
        padding_mode=dense_padding_mode,
    )

    if indexing == "hw":
        flow = flow.flip(-1)

    return warped, flow


@script
def _sparse_image_warp_noflow(
    image,
    source_points,
    dest_points,
    indexing: str,
    field_interpolation_order: int,
    field_regularization_weight: float,
    field_full_matrix: bool,
    pinned_boundary_points: int,
    dense_interpolation_mode: str,
    dense_padding_mode: str,
) -> torch.Tensor:
    # all our computations assume "wh" ordering, so we flip it here if necessary.
    # Though unintuitive, we need this for our call to grid_sample
    if indexing == "hw":
        source_points = source_points.flip(-1)
        dest_points = dest_points.flip(-1)

    source_points = source_points.float()
    dest_points = dest_points.float()

    N, C, H, W = image.shape
    WH = torch.tensor([[W, H]] * N, dtype=image.dtype, device=image.device)

    M = source_points.shape[1]
    if not M:
        return image

    if pinned_boundary_points > 0:
        pinned_points = _deterimine_pinned_points(pinned_boundary_points, WH)
        source_points = torch.cat([source_points, pinned_points], 1)  # (N,M',2)
        dest_points = torch.cat([dest_points, pinned_points], 1)  # (N,M+4k=M',2)
        # now just pretend M' was M all along

    H_range = torch.arange(H, dtype=image.dtype, device=image.device)  # (H,)
    W_range = torch.arange(W, dtype=image.dtype, device=image.device)  # (W,)
    h, w = meshgrid(H_range, W_range)  # (H, W), (H, W)
    query_points = torch.stack([w.flatten(), h.flatten()], 1)  # (H * W, 2)

    # If we can return just the warped image, we can bypass our call to dense_image_warp
    # by interpolating the 'grid' parameter of 'grid_sample' instead of the 'flow'
    # parameter of 'dense_image_warp'
    # coord = ((grid + 1) * size - 1) / 2
    # grid = (2 coord + 1) / size - 1
    train_points = dest_points  # (N, M, 2)
    train_values = (2.0 * source_points + 1.0) / WH.unsqueeze(1) - 1.0  # (N, M, 2)

    grid = polyharmonic_spline(
        train_points,
        train_values,
        query_points.unsqueeze(0).expand(N, H * W, 2),
        field_interpolation_order,
        regularization_weight=field_regularization_weight,
        full_matrix=field_full_matrix,
    )

    grid = grid.view(N, H, W, 2)

    warped = torch.nn.functional.grid_sample(
        image,
        grid,
        mode=dense_interpolation_mode,
        padding_mode=dense_padding_mode,
        align_corners=False,
    )

    return warped


if TYPE_CHECKING:

    def sparse_image_warp(
        image: torch.Tensor,
        source_points: torch.Tensor,
        dest_points: torch.Tensor,
        indexing: str = "hw",
        field_interpolation_order: int = 2,
        field_regularization_weight: float = 0.0,
        field_full_matrix: bool = True,
        pinned_boundary_points: int = 0,
        dense_interpolation_mode: str = "bilinear",
        dense_padding_mode: str = "border",
        include_flow: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Functional version of SparseImageWarp
        
        See Also
        --------
        pydrobert.torch.layers.SparseImageWarp
            For a description of this function and its parameters
        """
        pass


else:

    def sparse_image_warp(
        image: torch.Tensor,
        source_points: torch.Tensor,
        dest_points: torch.Tensor,
        indexing: str = "hw",
        field_interpolation_order: int = 2,
        field_regularization_weight: float = 0.0,
        field_full_matrix: bool = True,
        pinned_boundary_points: int = 0,
        dense_interpolation_mode: str = "bilinear",
        dense_padding_mode: str = "border",
        include_flow: bool = True,
    ) -> Any:
        if include_flow:
            return _sparse_image_warp_flow(
                image,
                source_points,
                dest_points,
                indexing,
                field_interpolation_order,
                field_regularization_weight,
                field_full_matrix,
                pinned_boundary_points,
                dense_interpolation_mode,
                dense_padding_mode,
            )
        else:
            return _sparse_image_warp_noflow(
                image,
                source_points,
                dest_points,
                indexing,
                field_interpolation_order,
                field_regularization_weight,
                field_full_matrix,
                pinned_boundary_points,
                dense_interpolation_mode,
                dense_padding_mode,
            )


class SparseImageWarp(torch.nn.Module):
    r"""Warp an image by specifying mappings between few control points

    This module, when instantiated, has the signature::

        warped[, flow] = sparse_image_warp(image, source_points, dest_points)

    `image` is a source image of shape `(N, C, H, W)``, where ``N`` is the batch
    dimension, ``C`` the channel dimension, ``H`` the image height, and ``W`` the image
    width. `source_points` and `dest_points` are tensors of shape ``(N, M, 2)``, where
    ``M`` is the number of control points. `warped` is a float tensor of shape ``(N, C,
    H, W)`` containing the warped images. The point ``source_points[n, m, :]`` in
    `image` will be mapped to ``dest_points[n, m, :]`` in `warped`. If `include_flow` is
    :obj:`True`, `flow`, a float tensor of shape ``(N, H, W, 2)``. ``flow[n, h, w, :]``
    is the flow for coordinates ``h, w`` in whatever order was specified by `indexing`.
    See :class:`DenseImageWarp` for more details about `flow`.

    This module mirrors the behaviour of Tensorflow's `sparse_image_warp
    <https://www.tensorflow.org/addons/api_docs/python/tfa/image/sparse_image_warp>`__,
    except `image` is in ``NCHW`` order instead of ``NHWC`` order. For more details,
    please consult their documentation.

    Parameters
    ----------
    indexing : {'hw', 'wh'}, optional
        If `indexing` is ``"hw"``, ``source_points[n, m, 0]`` and
        ``dest_points[n, m, 0]`` index the height dimension in `image` and `warped`,
        respectively, and ``source_points[n, m, 1]`` and ``dest_points[n, m, 1]`` the
        width dimension. If `indexing` is ``"wh"``, the width dimension is the 0-index
        and height the 1.
    field_interpolation_order : int, optional
        The order of the polyharmonic spline used to interpolate the rest of the points
        from the control. See :func:`polyharmonic_spline` for more info.
    field_regularization_weight : int, optional
        The regularization weight of the polyharmonic spline used to interpolate the
        rest of the points from the control. See :func:`polyharmonic_spline` for more
        info.
    field_full_matrix : bool, optional
        Determines the method of calculating the polyharmonic spline used to interpolate
        the rest of the points from the control. See :func:`polyharmonic_spline` for
        more info.
    pinned_boundary_points : int, optional
        Dictates whether and how many points along the boundary of `image` are mapped
        identically to points in `warped`. This keeps the boundary of the `image` from
        being pulled into the interior of `warped`. When :obj:`0`, no points are added.
        When :obj:`1`, four points are added, one in each corner of the image. When
        ``k > 2``, one point in each corner of the image is added, then ``k - 1``
        equidistant points along each of the four edges, totaling ``4 * k`` points.
    dense_interpolation_mode : {'bilinear', 'nearest'}, optional
        The method with which partial indices in the derived mapping are interpolated.
        See :func:`dense_image_warp` for more info.
    dense_padding_mode : {'border', 'zero', 'reflection'}, optional
        What to do when points in the derived mapping fall outside of the boundaries.
        See :func:`dense_image_warp` for more info.
    include_flow : bool, optional
        If :obj:`True`, include the flow field `flow` interpolated from the control
        points in the return value.
    
    Warnings
    --------
    When this module is scripted, its return type will be :class:`typing.Any`. This
    reflects the fact that either `warn` is returned on its own (a tensor) or both
    `warn` and `flow` (a tuple). Use :func:`torch.jit.isinstance` for type refinement in
    subsequent scripting. Tracing will infer the correct type.
    """

    __constants__ = [
        "indexing",
        "field_interpolation_order",
        "field_regularization_weight",
        "field_full_matrix",
        "pinned_boundary_points",
        "dense_interpolation_mode",
        "dense_padding_mode",
        "include_flow",
    ]

    field_interpolation_order: int
    field_regularization_weight: float
    field_full_matrix: bool
    pinned_boundary_points: int
    dense_interpolation_mode: str
    dense_padding_mode: str
    include_flow: bool

    def __init__(
        self,
        indexing: str = "hw",
        field_interpolation_order: int = 2,
        field_regularization_weight: float = 0.0,
        field_full_matrix: bool = True,
        pinned_boundary_points: int = 0,
        dense_interpolation_mode: str = "bilinear",
        dense_padding_mode: str = "border",
        include_flow: bool = True,
    ):
        super().__init__()
        if field_interpolation_order <= 0:
            raise ValueError(
                "field_interpolation_order must be positive, got "
                f"{field_interpolation_order}"
            )
        if pinned_boundary_points < 0:
            raise ValueError(
                "pinned_boundary_points must be non-negative, got "
                f"{pinned_boundary_points}"
            )
        if indexing not in {"hw", "wh"}:
            raise ValueError(f"indexing must be either 'hw' or 'wh', got '{indexing}'")
        if dense_interpolation_mode not in {"bilinear", "nearest"}:
            raise ValueError(
                "dense_interpolation_mode must be either 'bilinear' or 'nearest', got "
                f"'{dense_interpolation_mode}'"
            )
        if dense_padding_mode not in {"border", "zeros", "reflection"}:
            raise ValueError(
                "dense_padding_mode must be one of 'border', 'zeros', or 'relection', "
                f"got '{dense_padding_mode}'"
            )
        self.indexing = indexing
        self.field_interpolation_order = field_interpolation_order
        self.field_regularization_weight = field_regularization_weight
        self.field_full_matrix = field_full_matrix
        self.pinned_boundary_points = pinned_boundary_points
        self.dense_interpolation_mode = dense_interpolation_mode
        self.dense_padding_mode = dense_padding_mode
        self.include_flow = include_flow

    def extra_repr(self) -> str:
        return ", ".join(f"{x}={getattr(self, x)}" for x in self.__constants__)

    if TYPE_CHECKING:

        def forward(
            self,
            image: torch.Tensor,
            source_points: torch.Tensor,
            dest_points: torch.Tensor,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            pass

    else:

        def forward(
            self,
            image: torch.Tensor,
            source_points: torch.Tensor,
            dest_points: torch.Tensor,
        ) -> Any:
            return sparse_image_warp(
                image,
                source_points,
                dest_points,
                self.indexing,
                self.field_interpolation_order,
                self.field_regularization_weight,
                self.field_full_matrix,
                self.pinned_boundary_points,
                self.dense_interpolation_mode,
                self.dense_padding_mode,
                self.include_flow,
            )


@script
def pad_variable(
    x: torch.Tensor,
    lens: torch.Tensor,
    pad: torch.Tensor,
    mode: str = "constant",
    value: float = 0.0,
) -> torch.Tensor:
    """Functional version of PadVariable

    See Also
    --------
    pydrobert.torch.layers.PadVariable
        For a description of this function and its parameters
    """
    old_shape = x.shape
    ndim = len(old_shape)
    if ndim < 2:
        raise ValueError("Expected x to be at least two dimensional")
    N, T = old_shape[:2]
    if lens.shape != (N,):
        raise ValueError(
            f"For x of shape {old_shape}, lens should have shape ({N},) but got"
            f"{lens.shape}"
        )
    if pad.shape != (2, N):
        raise ValueError(
            f"For x of shape {old_shape}, pad should have shape (2, {N}), but got "
            f"{pad.shape}"
        )
    x = x.reshape(N, T, -1)
    F = x.size(2)
    new_lens = lens + pad.sum(0)
    Tp = int(new_lens.max().clamp_(min=T).item())
    arange_ = torch.arange(Tp, device=x.device)
    left_mask = (pad[0].unsqueeze(1) > arange_).unsqueeze(2).expand(N, Tp, F)
    if mode == "constant":
        buff = torch.tensor(value, device=x.device).to(x.dtype).view(1)
        left_pad = buff.expand(pad[0].sum() * F)
        right_pad = buff.expand(pad[1].sum() * F)
    elif mode == "reflect":
        if (pad >= lens.unsqueeze(0)).any():
            raise NotImplementedError(
                "For reflect padding, all padding lengths must be less than the "
                "sequence length"
            )
        max_, _ = pad.max(1)
        left_max, right_max = max_[0], max_[1]
        left_idxs = (
            (pad[0].unsqueeze(1) - arange_[:left_max])
            .clamp_(min=0)
            .unsqueeze(2)
            .expand(N, left_max, F)
        )
        left_pad = x.gather(1, left_idxs).masked_select(left_mask[:, :left_max])
        right_idxs = (
            (lens.unsqueeze(1) - arange_[:right_max] - 2)
            .clamp_(min=0)
            .unsqueeze(2)
            .expand(N, right_max, F)
        )
        right_mask_ = (
            (pad[1].unsqueeze(1) > arange_[:right_max])
            .unsqueeze(2)
            .expand(N, right_max, F)
        )
        right_pad = x.gather(1, right_idxs).masked_select(right_mask_)
    elif mode == "replicate":
        if (lens < 1).any():
            raise RuntimeError(f"For replicate padding, all lens must be > 0")
        max_, _ = pad.max(1)
        left_max, right_max = max_[0], max_[1]
        left_pad = (
            x[:, :1].expand(N, left_max, F).masked_select(left_mask[:, :left_max])
        )
        right_mask_ = (
            (pad[1].unsqueeze(1) > arange_[:right_max])
            .unsqueeze(2)
            .expand(N, right_max, F)
        )
        right_pad = (
            x.gather(1, (lens - 1).view(N, 1, 1).expand(N, right_max, F))
            .expand(N, right_max, F)
            .masked_select(right_mask_[:, :right_max])
        )
    else:
        raise ValueError(
            f"mode must be one of 'constant', 'reflect', 'replicate', got '{mode}'"
        )
    mid_mask = ((pad[0] + lens).unsqueeze(1) > arange_).unsqueeze(2).expand(N, Tp, F)
    len_mask = (lens.unsqueeze(1) > arange_[:T]).unsqueeze(2).expand(N, T, F)
    padded = torch.empty((N, Tp, F), device=x.device, dtype=x.dtype)
    padded = padded.masked_scatter(left_mask, left_pad)
    x = x.masked_select(len_mask)
    padded = padded.masked_scatter(mid_mask & ~left_mask, x)
    right_mask = (new_lens.unsqueeze(1) > arange_).unsqueeze(2).expand(N, Tp, F)
    padded = padded.masked_scatter(right_mask & ~mid_mask, right_pad)
    old_shape = list(old_shape)
    old_shape[1] = Tp
    return padded.view(old_shape)


class PadVariable(torch.nn.Module):
    """Pad variable-length input by a variable amount on each side

    This module attempts to replicate the behaviour of :func:`torch.nn.functional.pad`
    on a tensor containing variable sequence lengths with variable amounts of
    padding.

    When instantiated, this module has the signature::

        padded = pad_variable(x, lens, pad)

    `x` is a tensor of shape ``(N, T, *)`` where ``N`` is the batch index and ``T`` is
    the sequence index. `lens` is a long tensor of shape ``(N,)`` specifying the
    sequence lengths: only the values in the range ``x[n, :lens[n]]`` are considered
    part of the sequence of batch element ``n``. `pad` is a tensor of shape ``(2, N)``
    specifying how many elements at the start (``pad[0]``) and end (``pad[1]``) of each
    sequence. The return tensor `padded` will have shape ``(N, T', *)`` such that, for a
    given batch index ``n``::

        padded[n, :pad[0, n]] = left padding
        padded[n, pad[0,n]:pad[0,n] + lens[n]] = x[n, :lens[n]]
        padded[n, pad[0,n] + lens[n]:pad[0,n] + lens[n] + pad[1, n]] = right padding

    Parameters
    ----------
    mode : {'constant', 'reflect', 'replicate'}, optional
        How to pad the sequences. :obj:`'constant'`: fill the padding region with the
        value specified by `value`. :obj:`'reflect'`: padded values are reflections
        around the endpoints. For example, the first right-padded value of the ``n``-th
        sequence would be ``x[n, lens[n] - 2``, the third ``x[n, lens[n] - 3]``, and
        so on. :obj:`replicate`: padding duplicates the endpoints of each sequence.
        For example, the left-padded values of the ``n``-th sequence would all be
        ``x[n, 0]``; the right-padded values would be ``x[n, lens[n] - 1]``.
    value : scalar, optional
        The value to pad with when ``mode == 'constant'``.

    Raises
    ------
    NotImplementedError
        If any value in ``pad[:, n]`` equals or exceeds ``lens[n]`` when
        ``mode == 'reflect'``
    RuntimeError
        If any element in `lens` is less than 1 when ``mode == 'replicate'``

    Examples
    --------

    >>> x = torch.arange(10).view(2, 5)
    >>> x
    tensor([[0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]])
    >>> lens = torch.tensor([3, 4])
    >>> pad = torch.arange(4).view(2, 2)
    >>> pad.t()  # [[0_left, 0_right], [1_left, 1_right]]
    tensor([[0, 2],
            [1, 3]])
    >>> y = pad_variable(x, lens, pad)  # constant w/ value 0
    >>> y[0, :3 + 0 + 2]
    tensor([0, 1, 2, 0, 0])
    >>> y[1, :4 + 1 + 3]
    tensor([0, 5, 6, 7, 8, 0, 0, 0])
    >>> y = pad_variable(x, lens, pad, 'reflect')
    >>> y[0, :3 + 0 + 2]
    tensor([0, 1, 2, 1, 0])
    >>> y[1, :4 + 1 + 3]
    tensor([6, 5, 6, 7, 8, 7, 6, 5])
    >>> y = pad_variable(x, lens, pad, 'replicate')
    >>> y[0, :3 + 0 + 2]
    tensor([0, 1, 2, 2, 2])
    >>> y[1, :4 + 1 + 3]
    tensor([5, 5, 6, 7, 8, 8, 8, 8])
    """

    __constants__ = ["mode", "value"]

    mode: str
    value: float

    def __init__(self, mode: str = "constant", value: float = 0.0):
        super().__init__()
        if mode not in {"constant", "reflect", "replicate"}:
            raise ValueError(
                "mode should be one of 'constant', 'reflect', or 'replicate', got "
                f"'{mode}'"
            )
        self.mode = mode
        self.value = value

    def extra_repr(self) -> str:
        s = f"mode={self.mode}"
        if self.mode == "constant":
            s += f", value={self.value}"
        return s

    def forward(
        self, x: torch.Tensor, lens: torch.Tensor, pad: torch.Tensor
    ) -> torch.Tensor:
        return pad_variable(x, lens, pad, self.mode, self.value)


@script
def random_shift(
    in_: torch.Tensor,
    in_lens: torch.Tensor,
    prop: Tuple[float, float],
    mode: str,
    value: float,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional version of RandomShift

    See Also
    --------
    RandomShift
        For definitions of arguments and a description of this function
    """
    if in_.dim() < 2:
        raise RuntimeError(f"in_ must be at least 2 dimensional")
    if in_lens.dim() != 1 or in_lens.size(0) != in_.size(0):
        raise RuntimeError(
            f"For in_ of shape {in_.shape}, expected in_lens to be of shape "
            f"({in_.size(0)}), got {in_lens.shape}"
        )
    if training:
        in_lens_ = in_lens.float()
        pad = torch.stack([prop[0] * in_lens_, prop[1] * in_lens_])
        pad *= torch.rand_like(pad)
        pad = pad.long()
        out_lens = in_lens + pad.sum(0)
        return pad_variable(in_, in_lens, pad, mode, value), out_lens
    else:
        return in_, in_lens


class RandomShift(torch.nn.Module):
    """Pad to the left and right of each sequence by a random amount

    This layer is intended for training models which are robust to small shifts in some
    variable-length sequence dimension (e.g. speech recognition). It pads each input
    sequence with some number of elements at its beginning and end. The number of
    elements is randomly chosen but bounded above by some proportion of the input length
    specified by the user. Its call signature is

        out, out_lens = layer(in_, in_lens)

    Where: `in_` is a tensor of shape ``(N, T, *)`` where ``N`` is the batch dimension
    and ``T`` is the sequence dimension; `in_lens` is a long tensor of shape ``(N,)``;
    `out` is a tensor of the same type as `in_` of shape ``(N, T', *)``; and `out_lens`
    is of shape ``(N,)``. The ``n``-th input sequence is stored in the range
    ``in_[n, :in_lens[n]]``. The padded ``n``-th sequence is stored in the range
    ``out[n, :out_lens[n]]``. Values outside of these ranges are undefined.

    The amount of padding is dictated by the parameter `prop` this layer is initialized
    with. A proportion is a non-negative float dictating the maximum ratio of the
    original sequence length which may be padded, exclusive. `prop` can be a pair
    ``left, right`` for separate ratios of padding at the beginning and end of a
    sequence, or just one float if the proportions are the same. For example,
    ``prop=0.5`` of a sequence of length ``10`` could result in a sequence of length
    between ``10`` and ``18`` inclusive since each side of the sequence could be padded
    with ``0-4`` elements (``0.5 * 10 = 5`` is an exclusive bound).

    Padding is only applied if this layer is in training mode. If testing,
    ``out, out_lens = in_, in_lens``.

    Parameters
    ----------
    prop : float or tuple
    mode : {'reflect', 'constant', 'replicate'}, optional
        The method with which to pad the input sequence.
    value : float, optional
        The constant with which to pad the sequence if `mode` is set to
        :obj:`'constant'`.

    Raises
    ------
    NotImplementedError
        On initialization if `mode` is :obj:`'reflect'` and a value in `prop` exceeds
        ``1.0``. Reflection currently requires the amount of padding does not exceed
        the original sequence length.

    See Also
    --------
    pydrobert.torch.util.pad_variable
        For more details on the different types of padding. Note the default `mode` is
        different between this and the function.
    """

    __constants__ = ["prop", "mode", "value"]

    prop: Tuple[float, float]
    mode: str
    value: float

    def __init__(
        self,
        prop: Union[float, Tuple[float, float]],
        mode: str = "reflect",
        value: float = 0.0,
    ):
        super().__init__()
        try:
            prop = (float(prop), float(prop))
        except TypeError:
            prop = tuple(prop)
        if len(prop) != 2:
            raise ValueError(
                f"prop must be a single or pair of floating points, got '{prop}'"
            )
        if prop[0] < 0.0 or prop[1] < 0.0:
            raise ValueError("prop values must be non-negative")
        if mode == "reflect":
            if prop[0] > 1.0 or prop[1] > 1.0:
                raise NotImplementedError(
                    "if 'mode' is 'reflect', values in 'prop' must be <= 1"
                )
        elif mode not in {"constant", "replicate"}:
            raise ValueError(
                "'mode' must be one of 'reflect', 'constant', or 'replicate', got "
                f"'{mode}'"
            )
        self.mode = mode
        self.prop = prop
        self.value = value

    def extra_repr(self) -> str:
        return f"prop={self.prop}, mode={self.mode}, value={self.value}"

    def reset_parameters(self) -> None:
        pass

    def forward(
        self, in_: torch.Tensor, in_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return random_shift(
            in_, in_lens, self.prop, self.mode, self.value, self.training
        )


@script
def _spec_augment_check_input(
    feats: torch.Tensor, lengths: Optional[torch.Tensor] = None
):
    if feats.dim() != 3:
        raise RuntimeError(
            f"Expected feats to have three dimensions, got {feats.dim()}"
        )
    N, T, _ = feats.shape
    if lengths is not None:
        if lengths.dim() != 1:
            raise RuntimeError(
                f"Expected lengths to be one dimensional, got {lengths.dim()}"
            )
        if lengths.size(0) != N:
            raise RuntimeError(
                f"Batch dimension of feats ({N}) and lengths ({lengths.size(0)}) "
                "do not match"
            )
        if not torch.all((lengths <= T) & (lengths > 0)):
            raise RuntimeError(f"values of lengths must be between (1, {T})")


SpecAugmentParams = Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]


@script
def spec_augment_draw_parameters(
    feats: torch.Tensor,
    max_time_warp: float,
    max_freq_warp: float,
    max_time_mask: int,
    max_freq_mask: int,
    max_time_mask_proportion: float,
    num_time_mask: int,
    num_time_mask_proportion: float,
    num_freq_mask: int,
    lengths: Optional[torch.Tensor] = None,
) -> SpecAugmentParams:
    """Functional version of SpecAugment.draw_parameters

    See Also
    --------
    SpecAugment
        For definitions of arguments and a description of this function.
    """
    _spec_augment_check_input(feats, lengths)
    N, T, F = feats.shape
    device = feats.device
    eps = _get_tensor_eps(feats)
    omeps = 1 - eps
    if lengths is None:
        lengths = torch.full((N,), T, dtype=torch.float, device=device)
    else:
        lengths = lengths.to(device).float()
    # note that order matters slightly in whether we draw widths or positions first.
    # The paper specifies that position is drawn first for warps, whereas widths
    # are drawn first for masks
    if max_time_warp:
        # we want the range (W, length - W) exclusive to be where w_0 can come
        # from. If W >= length / 2, this is impossible. Rather than giving up,
        # we limit the maximum length to W < length / 2
        # N.B. We don't worry about going outside the valid range by a bit b/c
        # warp_1d_grid clamps.
        W = (lengths / 2 - eps).clamp(0, max_time_warp)
        w_0 = torch.rand((N,), device=device) * (lengths - 2 * W) + W
        w = torch.rand([N], device=device) * (2 * W) - W
    else:
        w_0 = w = torch.empty(0)
    if max_freq_warp:
        V = min(max(F / 2 - eps, 0), max_freq_warp)
        v_0 = torch.rand([N], device=device) * (F - 2 * V) + V
        v = torch.rand([N], device=device) * (2 * V) - V
    else:
        v_0 = v = torch.empty(0)
    if (
        max_time_mask
        and max_time_mask_proportion
        and num_time_mask
        and num_time_mask_proportion
    ):
        max_ = (
            torch.clamp(lengths * max_time_mask_proportion, max=max_time_mask,)
            .floor()
            .to(device)
        )
        nums_ = (
            torch.clamp(lengths * num_time_mask_proportion, max=num_time_mask,)
            .floor()
            .to(device)
        )
        t = (
            (
                torch.rand([N, num_time_mask], device=device)
                * (max_ + omeps).unsqueeze(1)
            )
            .long()
            .masked_fill(
                nums_.unsqueeze(1)
                <= torch.arange(num_time_mask, dtype=lengths.dtype, device=device),
                0,
            )
        )
        t_0 = (
            torch.rand([N, num_time_mask], device=device)
            * (lengths.unsqueeze(1) - t + omeps)
        ).long()
    else:
        t = t_0 = torch.empty(0)
    if max_freq_mask and num_freq_mask:
        max_ = min(max_freq_mask, F)
        f = (torch.rand([N, num_freq_mask], device=device) * (max_ + omeps)).long()
        f_0 = (torch.rand([N, num_freq_mask], device=device) * (F - f + omeps)).long()
    else:
        f = f_0 = torch.empty(0)
    return w_0, w, v_0, v, t_0, t, f_0, f


@script
def spec_augment_apply_parameters(
    feats: torch.Tensor,
    params: SpecAugmentParams,
    interpolation_order: int,
    lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Functional version of SpecAugment.apply_parameters

    See Also
    --------
    SpecAugment
        For definitions of arguments and a description of this function.
    """
    _spec_augment_check_input(feats, lengths)
    N, T, F = feats.shape
    device = feats.device
    if lengths is None:
        lengths = torch.full((N,), T, dtype=torch.long, device=device)
    lengths = lengths.to(feats.dtype)
    w_0, w, v_0, v, t_0, t, f_0, f = params
    new_feats = feats
    time_grid: Optional[torch.Tensor] = None
    freq_grid: Optional[torch.Tensor] = None
    do_warp = False
    if w_0 is not None and w_0.numel() and w is not None and w.numel():
        time_grid = warp_1d_grid(w_0, w, lengths, T, interpolation_order)
        do_warp = True
    if v_0 is not None and v_0.numel() and v is not None and v.numel():
        freq_grid = warp_1d_grid(
            v_0,
            v,
            torch.full((N,), F, dtype=torch.long, device=device),
            F,
            interpolation_order,
        )
        do_warp = True
    if do_warp:
        if time_grid is None:
            time_grid = torch.arange(T, device=device, dtype=torch.float)
            time_grid = (2 * time_grid + 1) / T - 1
            time_grid = time_grid.unsqueeze(0).expand(N, T)
        if freq_grid is None:
            freq_grid = torch.arange(F, device=device, dtype=torch.float)
            freq_grid = (2 * freq_grid + 1) / F - 1
            freq_grid = freq_grid.unsqueeze(0).expand(N, F)
        time_grid = time_grid.unsqueeze(2).expand(N, T, F)
        freq_grid = freq_grid.unsqueeze(1).expand(N, T, F)
        # note: grid coordinate are (freq, time) rather than (time, freq)
        grid = torch.stack([freq_grid, time_grid], 3)  # (N, T, F, 2)
        new_feats = torch.nn.functional.grid_sample(
            new_feats.unsqueeze(1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(1)
    tmask: Optional[torch.Tensor] = None
    fmask: Optional[torch.Tensor] = None
    if t_0 is not None and t_0.numel() and t is not None and t.numel():
        tmask = torch.arange(T, device=device).unsqueeze(0).unsqueeze(2)  # (1, T,1)
        t_1 = t_0 + t  # (N, MT)
        tmask = (tmask >= t_0.unsqueeze(1)) & (tmask < t_1.unsqueeze(1))  # (N,T,MT)
        tmask = tmask.any(2, keepdim=True)  # (N, T, 1)
    if f_0 is not None and f_0.numel() and f is not None and f.numel():
        fmask = torch.arange(F, device=device).unsqueeze(0).unsqueeze(2)  # (1, F,1)
        f_1 = f_0 + f  # (N, MF)
        fmask = (fmask >= f_0.unsqueeze(1)) & (fmask < f_1.unsqueeze(1))  # (N,F,MF)
        fmask = fmask.any(2).unsqueeze(1)  # (N, 1, F)
    if tmask is not None:
        if fmask is not None:
            tmask = tmask | fmask
        new_feats = new_feats.masked_fill(tmask, 0.0)
    elif fmask is not None:
        new_feats = new_feats.masked_fill(fmask, 0.0)
    return new_feats


@script
def spec_augment(
    feats: torch.Tensor,
    max_time_warp: float,
    max_freq_warp: float,
    max_time_mask: int,
    max_freq_mask: int,
    max_time_mask_proportion: float,
    num_time_mask: int,
    num_time_mask_proportion: float,
    num_freq_mask: int,
    interpolation_order: int,
    lengths: Optional[torch.Tensor] = None,
    training: bool = True,
) -> torch.Tensor:
    """Functional version of SpecAugment

    See Also
    --------
    SpecAugment
        For definitions of arguments and a description of this function.
    """
    _spec_augment_check_input(feats, lengths)
    if not training:
        return feats
    params = spec_augment_draw_parameters(
        feats,
        max_time_warp,
        max_freq_warp,
        max_time_mask,
        max_freq_mask,
        max_time_mask_proportion,
        num_time_mask,
        num_time_mask_proportion,
        num_freq_mask,
        lengths,
    )
    return spec_augment_apply_parameters(feats, params, interpolation_order, lengths)


class SpecAugment(torch.nn.Module):
    r"""Perform warping/masking of time/frequency dimensions of filter bank features

    SpecAugment [park2019]_ (and later [park2020]_) is a series of data transformations
    for training data augmentation of time-frequency features such as Mel-scaled
    triangular filter bank coefficients.

    An instance `spec_augment` of `SpecAugment` is called as

        new_feats = spec_augment(feats[, lengths])

    `feats` is a float tensor of shape ``(N, T, F)`` where ``N`` is the batch dimension,
    ``T`` is the time (frames) dimension, and ``F`` is the frequency (coefficients per
    frame) dimension. `lengths` is an optional long tensor of shape ``(N,)`` specifying
    the actual number of frames before right-padding per batch element. That is,
    for batch index ``n``, only ``feats[n, :lengths[n]]`` are valid. `new_feats` is
    of the same size as `feats` with some or all of the following operations performed
    in order independently per batch index:

    1. Choose a random frame along the time dimension. Warp `feats` such that ``feats[n,
       0]`` and feats[n, lengths[n] - 1]`` are fixed, but that random frame gets mapped
       to a random new location a few frames to the left or right.
    2. Do the same for the frequency dimension.
    3. Mask out (zero) one or more random-width ranges of frames in a random location
       along the time dimension.
    4. Do the same for the frequency dimension.

    The original SpecAugment implementation only performs steps 1, 3, and 4; step 2 is a
    trivial extension.

    Default parameter values are from [park2020]_.

    The `spec_augment` instance must be in training mode in order to apply any
    transformations; `spec_augment` always returns `feats` as-is in evaluation mode.

    Parameters
    ----------
    max_time_warp : float, optional
        A non-negative float specifying the maximum number of frames the chosen
        random frame can be shifted left or right by in step 1. Setting to :obj:`0`
        disables step 1.
    max_freq_warp : float, optional
        A non-negative float specifying the maximum number of coefficients the chosen
        random frequency coefficient index will be shifted up or down by in step 2.
        Setting to :obj:`0` disables step 2.
    max_time_mask : int, optional
        A non-negative integer specifying an absolute upper bound on the number of
        sequential frames in time that can be masked out by a single mask. The minimum
        of this upper bound and that from `max_time_mask_proportion` specifies the
        actual maximum. Setting this, `max_time_mask_proportion`, `num_time_mask`,
        or `num_time_mask_proportion` to :obj:`0` disables step 3.
    max_freq_mask : int, optional
        A non-negative integer specifying the maximum number of sequential coefficients
        in frequency that can be masked out by a single mask. Setting this or
        `num_freq_mask` to :obj:`0` disables step 4.
    max_time_mask_proportion : float, optional
        A value in the range :math:`[0, 1]` specifying a relative upper bound on the
        number of squential frames in time that can be masked out by a single mask. For
        batch element ``n``, the upper bound is ``int(max_time_mask_poportion *
        length[n])``. The minimum of this upper bound and that from `max_time_mask`
        specifies the actual maximum. Setting this, `max_time_mask`, `num_time_mask`,
        or `num_time_mask_proportion` to :obj:`0` disables step 4.
    num_time_mask : int, optional
        A non-negative integer specifying an absolute upper bound number of random masks
        in time per batch element to create. Setting this, `num_time_mask_proportion`,
        `max_time_mask`, or `max_time_mask_proportion` to :obj:`0` disables step 3.
        Drawn i.i.d. and may overlap.
    num_time_mask_proportion : float, optional
        A value in the range :math:`[0, 1]` specifying a relative upper bound on the
        number of time masks per element in the batch to create. For batch element
        ``n``, the upper bound is ``int(num_time_mask_proportion * length[n])``. The
        minimum of this upper bound and that from `num_time_mask` specifies the
        actual maximum. Setting this, `num_time_mask`, `max_time_mask`, or
        `max_time_mask_proportion` to :obj:`0` disables step 3. Drawn i.i.d. and may
        overlap.
    num_freq_mask : int, optional
        The total number of random masks in frequency per batch element to create.
        Setting this or `max_freq_mask` to :obj:`0` disables step 4. Drawn i.i.d. and
        may overlap.
    interpolation_order : int, optional
        Controls order of interpolation of warping. 1 = linear (default for
        [park2020]_). 2 = thin plate (default for [park2019]_). Higher orders are
        possible at increased computational cost.

    Notes
    -----
    There are a few differences between this implementation of warping and those you
    might find online or described in the source paper [park2019]_. These require some
    knowledge of what's happening under the hood and are unlikely to change the way you
    use this function. We assume we're warping in time, though the following applies to
    frequency warping as well.

    First, the warp parameters are real- rather than integer-valued. You can set
    `max_time_warp` or `max_freq_warp` to 0.5 if you'd like. The shift value drawn
    between ``[0, max_time_warp]`` is also real-valued. Since the underlying warp
    relies on interpolation between partial indices anyways (the vast majority of tensor
    values will be the result of interpolation), there is no preference for
    integer-valued parameters from a computational standpoint. Further, real-valued warp
    parameters allow for a virtually infinite number of warps instead of just a few.

    Finally, time warping is implemented by determining the transformation in one
    dimension (time) and broadcasting it across the other (frequency), rather than
    performing a two-dimensional warp. This is not in line with [park2019]_, but is
    with [park2020]_. I have confirmed with the first author that the slight warping
    of frequency that occurred due to the 2D warp was unintentional.
    """

    __constants__ = [
        "max_time_warp",
        "max_freq_warp",
        "max_time_mask",
        "max_freq_mask",
        "max_time_mask_proportion",
        "num_time_mask",
        "num_time_mask_proportion",
        "num_freq_mask",
        "interpolation_order",
    ]

    max_time_warp: float
    max_freq_warp: float
    max_time_mask: int
    max_freq_mask: int
    max_time_mask_proportion: float
    num_time_mask: int
    num_time_mask_proportion: float
    num_freq_mask: int
    interpolation_order: int

    def __init__(
        self,
        max_time_warp: float = 80.0,
        max_freq_warp: float = 0.0,
        max_time_mask: int = 100,
        max_freq_mask: int = 27,
        max_time_mask_proportion: float = 0.04,
        num_time_mask: int = 20,
        num_time_mask_proportion: float = 0.04,
        num_freq_mask: int = 2,
        interpolation_order: int = 1,
    ):
        super().__init__()
        self.max_time_warp = float(max_time_warp)
        self.max_freq_warp = float(max_freq_warp)
        self.max_time_mask = max_time_mask
        self.max_freq_mask = max_freq_mask
        self.max_time_mask_proportion = max_time_mask_proportion
        self.num_time_mask = num_time_mask
        self.num_time_mask_proportion = num_time_mask_proportion
        self.num_freq_mask = num_freq_mask
        self.interpolation_order = interpolation_order

    def extra_repr(self) -> str:
        s = "warp_t={},max_f={},num_f={},max_t={},max_t_p={:.2f},num_t={}".format(
            self.max_time_warp,
            self.max_freq_mask,
            self.num_freq_mask,
            self.max_time_mask,
            self.max_time_mask_proportion,
            self.num_time_mask,
        )
        if self.max_freq_warp:
            s += ",warp_f={}".format(self.max_freq_warp)
        return s

    def draw_parameters(
        self, feats: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> SpecAugmentParams:
        """Randomly draw parameterizations of augmentations

        Called as part of this layer's :func:`__call__` method.

        Parameters
        ----------
        feats : torch.Tensor
            Time-frequency features of shape ``(N, T, F)``.
        lengths : torch.Tensor or None, optional
            Long tensor of shape ``(N,)`` containing the number of frames before
            padding.

        Returns
        -------
        w_0 : torch.Tensor
            If step 1 is enabled, of shape ``(N,)`` containing the source points in the
            time warp (floatint-point). Otherwise, is empty.
        w : torch.Tensor
            If step 1 is enabled, of shape ``(N,)`` containing the number of frames to
            shift the source point by (positive or negative) in the destination in time.
            Positive values indicate a right shift. Otherwise is empty.
        v_0 : torch.Tensor
            If step 2 is enabled, of shape ``(N,)`` containing the source points in the
            frequency warp (floating point). Otherwise is empty.
        v : torch.Tensor
            If step 2 is enabled, of shape ``(N,)`` containing the number of
            coefficients to shift the source point by (positive or negative) in the
            destination in time. Positive values indicate a right shift. Otherwise is
            empty.
        t_0 : torch.Tensor
            If step 3 is enabled, of shape ``(N, M_T)`` where ``M_T`` is the number of
            time masks specifying the lower index (inclusive) of the time masks.
            Otherwise is empty.
        t : torch.Tensor
            If step 3 is enabled, of shape ``(N, M_T)`` specifying the number of frames
            per time mask. Otherise is empty.
        f_0 : torch.Tensor
            If step 4 is enabled, of shape ``(N, M_F)`` where ``M_F`` is the number of
            frequency masks specifying the lower index (inclusive) of the frequency
            masks. Otherwise is empty.
        f : torch.Tensor
            If step 4 is enabled, of shape ``(N, M_F)`` specifying the number of
            frequency coefficients per frequency mask. Otherwise is empty.
        """
        return spec_augment_draw_parameters(
            feats,
            self.max_time_warp,
            self.max_freq_warp,
            self.max_time_mask,
            self.max_freq_mask,
            self.max_time_mask_proportion,
            self.num_time_mask,
            self.num_time_mask_proportion,
            self.num_freq_mask,
            lengths,
        )

    def apply_parameters(
        self,
        feats: torch.Tensor,
        params: SpecAugmentParams,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Use drawn parameters to apply augmentations

        Called as part of this layer's :func:`__call__` method.

        Parameters
        ----------
        feats : torch.Tensor
            Time-frequency features of shape ``(N, T, F)``.
        params : sequence of torch.Tensor
            All parameter tensors returned by :func:`draw_parameters`.
        lengths : torch.Tensor, optional
            Tensor of shape ``(N,)`` containing the number of frames before padding.

        Returns
        -------
        new_feats : torch.Tensor
            Augmented time-frequency features of same shape as `feats`.
        """
        return spec_augment_apply_parameters(
            feats, params, self.interpolation_order, lengths
        )

    def reset_parameters(self) -> None:
        pass

    def forward(
        self, feats: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if lengths is None:
            # _spec_augment_check_input(feats)
            lengths = torch.full(
                (feats.size(0),), feats.size(1), dtype=torch.long, device=feats.device
            )
        if not self.training:
            return feats
        params = self.draw_parameters(feats, lengths)
        return self.apply_parameters(feats, params, lengths)
