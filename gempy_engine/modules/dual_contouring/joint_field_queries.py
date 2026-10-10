"""Validated field sampling for joint extraction, over caller-supplied callbacks.

``sample_fields(points) -> (S x M, S x M x 3)`` returns every surface's raw
field and actual gradients; ``field_query([(points, surfaces, kind)])`` answers
targeted requests. Values are host NumPy float64.
"""

import numpy as np


def _host(values):
    return values.detach().cpu().numpy() if hasattr(values, 'detach') else np.asarray(values)


class ExactPointCache:
    """All-surface samples keyed by exact coordinates; each point is sampled once.

    ``validate(points, raw, gradients)`` may reject freshly sampled blocks.
    """

    def __init__(self, sample_fields, n_surfaces, validate=None):
        self._sample, self._n, self._validate = sample_fields, n_surfaces, validate
        self._index, self._raw, self._gradients = {}, [], []

    def __len__(self):
        return len(self._index)

    def __call__(self, points):
        n = self._n
        points = np.asarray(points).reshape(-1, 3)
        keys = list(map(tuple, points))
        missing = sorted(set(keys)-self._index.keys())
        if missing:
            raw, gradients = self._sample(np.asarray(missing))
            raw, gradients = np.array(_host(raw), copy=True), np.array(_host(gradients), copy=True)
            if raw.shape != (n, len(missing)) or gradients.shape != (n, len(missing), 3) or not (
                    np.isfinite(raw).all() and np.isfinite(gradients).all()):
                raise ValueError('invalid_callback_samples: finite all-surface scalar/gradient arrays required')
            if self._validate is not None:
                self._validate(np.asarray(missing), raw, gradients)
            self._index.update(zip(missing, range(len(self._index), len(self._index)+len(missing))))
            self._raw.append(raw)
            self._gradients.append(gradients)
        columns = np.fromiter((self._index[k] for k in keys), dtype=np.int64, count=len(keys))
        if len(self._raw) > 1:
            self._raw[:] = [np.concatenate(self._raw, axis=1)]
            self._gradients[:] = [np.concatenate(self._gradients, axis=1)]
        if not self._raw:
            return np.empty((n, 0)), np.empty((n, 0, 3))
        return self._raw[0][:, columns], self._gradients[0][:, columns]


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
