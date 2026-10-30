"""Validated field sampling for joint extraction, over caller-supplied callbacks.

``sample_fields(points) -> (S x M, S x M x 3)`` returns every surface's raw
field and actual gradients; ``field_query([(points, surfaces, kind)])`` answers
targeted requests. Values are float64 arrays of the extraction namespace ``xp``
(NumPy, or Torch on the extraction device). Callbacks receive host NumPy
points unless they set ``device_native = True``; they may return either.
"""

import numpy as np

from .joint_xp import NUMPY, host


def _unique_rows(points, xp=NUMPY):
    """Sorted unique (K, 3) rows and each row's index into them; -0.0 equals 0.0, as in tuples.

    Rows sort lexicographically (x, then y, then z), as ``np.lexsort(points.T[::-1])``.
    """
    points = xp.asarray(points, dtype=np.float64).reshape(-1, 3) + 0.
    order = xp.argsort(points[:, 2])
    for axis in (1, 0):
        order = order[xp.argsort(points[order, axis])]
    ordered = points[order]
    start = xp.ones(len(ordered), dtype=bool)
    start[1:] = (ordered[1:] != ordered[:-1]).any(1)
    inverse = xp.empty(len(points), dtype=np.int64)
    inverse[order] = xp.cumsum(xp.astype(start, np.int64)) - 1
    return ordered[start], inverse


def unique_rows_per_request(point_sets, xp=NUMPY):
    """``_unique_rows`` of each (K_r, 3) array in ``point_sets``, from one sort of them all.

    Returns per array its sorted unique rows and each row's index into them.
    """
    sizes = [len(points) for points in point_sets]
    points = xp.concatenate([xp.asarray(p, dtype=np.float64).reshape(-1, 3) for p in point_sets]) + 0.
    ends = np.cumsum(sizes)
    request = xp.searchsorted(xp.asarray(ends), xp.arange(len(points)) + 1) if len(points) else xp.arange(0)
    request = xp.minimum(request, len(sizes) - 1)
    # Lexicographic (request, x, y, z): stable sorts from the least significant key.
    order = xp.argsort(points[:, 2])
    for axis in (1, 0):
        order = order[xp.argsort(points[order, axis])]
    order = order[xp.argsort(request[order])]
    ordered, ordered_request = points[order], request[order]
    start = xp.ones(len(ordered), dtype=bool)
    start[1:] = (ordered[1:] != ordered[:-1]).any(1) | (ordered_request[1:] != ordered_request[:-1])
    inverse = xp.empty(len(points), dtype=np.int64)
    inverse[order] = xp.cumsum(xp.astype(start, np.int64)) - 1
    unique, unique_request = ordered[start], ordered_request[start]
    # Unique rows are ordered by request: request r's end is the count of rows of requests <= r.
    unique_ends = (host(xp.searchsorted(unique_request, xp.arange(len(sizes)) + 1)) if len(points)
                   else np.zeros(len(sizes), dtype=np.int64))
    point_starts, unique_starts = np.concatenate(([0], ends[:-1])), np.concatenate(([0], unique_ends[:-1]))
    return [(unique[u0:u1], inverse[p0:p1] - int(u0))
            for p0, p1, u0, u1 in zip(point_starts.tolist(), ends.tolist(), unique_starts.tolist(), unique_ends.tolist())]


def _callback_points(callback, points):
    return points if getattr(callback, 'device_native', False) else host(points)


def _finite(xp, *arrays):
    """One host check (one sync on a device) for all arrays."""
    finite = xp.isfinite(arrays[0]).all()
    for a in arrays[1:]:
        finite = finite & xp.isfinite(a).all()
    return bool(finite)


class ExactPointCache:
    """All-surface samples keyed by exact coordinates; each point is sampled once.

    ``validate(points, raw, gradients)`` may reject freshly sampled blocks.
    New points are sampled in sorted order.
    """

    def __init__(self, sample_fields, n_surfaces, validate=None, xp=NUMPY):
        self._sample, self._n, self._validate, self._xp = sample_fields, n_surfaces, validate, xp
        self._points = xp.empty((0, 3))
        self._raw, self._gradients = xp.empty((n_surfaces, 0)), xp.empty((n_surfaces, 0, 3))

    def __len__(self):
        return len(self._points)

    def __call__(self, points):
        xp, n, stored = self._xp, self._n, len(self._points)
        merged, inverse = _unique_rows(xp.concatenate((self._points, xp.asarray(points, dtype=np.float64).reshape(-1, 3))), xp)
        requested = inverse[stored:]
        if len(merged) > stored:
            # Stored points are sorted and unique, so they keep their relative order.
            new = xp.ones(len(merged), dtype=bool)
            new[inverse[:stored]] = False
            missing = merged[new]
            raw, gradients = self._sample(_callback_points(self._sample, missing))
            raw, gradients = xp.copy(xp.asarray(raw, dtype=np.float64)), xp.copy(xp.asarray(gradients, dtype=np.float64))
            if tuple(raw.shape) != (n, len(missing)) or tuple(gradients.shape) != (n, len(missing), 3) or not _finite(
                    xp, raw, gradients):
                raise ValueError('invalid_callback_samples: finite all-surface scalar/gradient arrays required')
            if self._validate is not None:
                self._validate(missing, raw, gradients)
            all_raw, all_gradients = xp.empty((n, len(merged))), xp.empty((n, len(merged), 3))
            all_raw[:, ~new], all_gradients[:, ~new] = self._raw, self._gradients
            all_raw[:, new], all_gradients[:, new] = raw, gradients
            self._points, self._raw, self._gradients = merged, all_raw, all_gradients
        return self._raw[:, requested], self._gradients[:, requested]


class TargetedFieldBatch:
    """Validated batched targeted queries; empty requests are answered locally."""

    def __init__(self, field_query, xp=NUMPY):
        self._query, self._xp, self.sampled = field_query, xp, 0

    def __call__(self, requests):
        xp = self._xp
        requests = [(xp.asarray(points, dtype=np.float64).reshape(-1, 3), list(surfaces), kind)
                    for points, surfaces, kind in requests]
        live = [i for i, (points, _, _) in enumerate(requests) if len(points)]
        answers = self._query([(_callback_points(self._query, requests[i][0]),) + requests[i][1:]
                               for i in live]) if live else []
        if len(answers) != len(live):
            raise ValueError('invalid_callback_samples: one result per request required')
        results = [xp.empty((len(surfaces), len(points)) + ((3,) if kind == 'gradient' else ()))
                   for points, surfaces, kind in requests]
        for i, answer in zip(live, answers):
            points, surfaces, kind = requests[i]
            answer = xp.copy(xp.asarray(answer, dtype=np.float64))
            if tuple(answer.shape) != tuple(results[i].shape):
                raise ValueError(f'invalid_callback_samples: finite requested-surface {kind}s required')
            results[i] = answer
            self.sampled += len(points)
        # Finiteness of every answer in one host check; the first failing request names the error.
        if live and not _finite(xp, *(results[i] for i in live)):
            kind = next(requests[i][2] for i in live if not _finite(xp, results[i]))
            raise ValueError(f'invalid_callback_samples: finite requested-surface {kind}s required')
        return results
