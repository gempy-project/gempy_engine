"""Validated field sampling for joint extraction, over caller-supplied callbacks.

``sample_fields(points) -> (S x M, S x M x 3)`` returns every surface's raw
field and actual gradients; ``field_query([(points, surfaces, kind)])`` answers
targeted requests. Values are host NumPy float64.
"""

import numpy as np


def _host(values):
    return values.detach().cpu().numpy() if hasattr(values, 'detach') else np.asarray(values)


def _unique_rows(points):
    """Sorted unique (K, 3) rows and each row's index into them; -0.0 equals 0.0, as in tuples."""
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3) + 0.
    order = np.lexsort(points.T[::-1])
    ordered = points[order]
    start = np.ones(len(ordered), dtype=bool)
    start[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    inverse = np.empty(len(points), dtype=np.int64)
    inverse[order] = np.cumsum(start)-1
    return ordered[start], inverse


class ExactPointCache:
    """All-surface samples keyed by exact coordinates; each point is sampled once.

    ``validate(points, raw, gradients)`` may reject freshly sampled blocks.
    New points are sampled in sorted order.
    """

    def __init__(self, sample_fields, n_surfaces, validate=None):
        self._sample, self._n, self._validate = sample_fields, n_surfaces, validate
        self._points = np.empty((0, 3))
        self._raw, self._gradients = np.empty((n_surfaces, 0)), np.empty((n_surfaces, 0, 3))

    def __len__(self):
        return len(self._points)

    def __call__(self, points):
        n, stored = self._n, len(self._points)
        merged, inverse = _unique_rows(np.concatenate((self._points, np.asarray(points, dtype=np.float64).reshape(-1, 3))))
        requested = inverse[stored:]
        if len(merged) > stored:
            # Stored points are sorted and unique, so they keep their relative order.
            new = np.ones(len(merged), dtype=bool)
            new[inverse[:stored]] = False
            missing = merged[new]
            raw, gradients = self._sample(missing)
            raw, gradients = np.array(_host(raw), copy=True), np.array(_host(gradients), copy=True)
            if raw.shape != (n, len(missing)) or gradients.shape != (n, len(missing), 3) or not (
                    np.isfinite(raw).all() and np.isfinite(gradients).all()):
                raise ValueError('invalid_callback_samples: finite all-surface scalar/gradient arrays required')
            if self._validate is not None:
                self._validate(missing, raw, gradients)
            all_raw, all_gradients = np.empty((n, len(merged))), np.empty((n, len(merged), 3))
            all_raw[:, ~new], all_gradients[:, ~new] = self._raw, self._gradients
            all_raw[:, new], all_gradients[:, new] = raw, gradients
            self._points, self._raw, self._gradients = merged, all_raw, all_gradients
        return self._raw[:, requested], self._gradients[:, requested]


class TargetedFieldBatch:
    """Validated batched targeted queries; empty requests are answered locally."""

    def __init__(self, field_query):
        self._query, self.sampled = field_query, 0

    def __call__(self, requests):
        requests = [(np.asarray(points, dtype=float).reshape(-1, 3), list(surfaces), kind)
                    for points, surfaces, kind in requests]
        live = [i for i, (points, _, _) in enumerate(requests) if len(points)]
        answers = self._query([requests[i] for i in live]) if live else []
        if len(answers) != len(live):
            raise ValueError('invalid_callback_samples: one result per request required')
        results = [np.empty((len(surfaces), len(points)) + ((3,) if kind == 'gradient' else ()))
                   for points, surfaces, kind in requests]
        for i, answer in zip(live, answers):
            points, surfaces, kind = requests[i]
            answer = np.array(_host(answer), dtype=float, copy=True)
            if answer.shape != results[i].shape or not np.isfinite(answer).all():
                raise ValueError(f'invalid_callback_samples: finite requested-surface {kind}s required')
            results[i] = answer
            self.sampled += len(points)
        return results
