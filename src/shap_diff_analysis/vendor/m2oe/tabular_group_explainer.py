"""Minimal NumPy port of the M2OE shared-choice tabular group explainer.

Provenance
----------
Upstream: https://github.com/AIDALab-DIMES/M2OE (MIT license, preserved in
``LICENSE`` next to this file; copyright (c) 2025 AIDALab-DIMES).
Commit: 925c96ec0abca1e4cfd934d486f1217623184c7f (accessed 2026-08-20).
Paper: Angiulli, Fassetti, Nisticò, Palopoli. "Explaining Outliers and
Anomalous Groups via Subspace Density Contrastive Loss." Machine Learning
113:7565-7589, 2024. DOI 10.1007/s10994-024-06618-8.

What was ported
---------------
Only the tabular *group* explanation path needed by this project:

* the ``TabularMM_SC`` shared-choice masking-model architecture (upstream
  ``M2OE/models/AETabularMM_SC.py``): a ``CHOOSE`` network producing one
  batch-shared per-feature weight vector in ``[0, 1]`` from a constant input,
  a ``MASK`` network mapping ``[outlier, reference]`` pairs to per-feature
  patch magnitudes, and the mask applier ``O' = O + mask * choice``;
* the subspace density contrastive loss (upstream ``compute_loss``), term by
  term including the ``1e-4`` epsilon and the ``sqrt(D)`` normalisation;
* reference-set selection via scikit-learn ``NearestNeighbors`` and the
  outlier/reference pairing of upstream ``Explainer._rs_selection`` /
  ``_combine_data`` / the group ``mm_data`` construction;
* Adam (upstream optimiser) with mini-batch training, and the upstream
  restart behaviour (``num_tries`` retrainings while the binarised choice is
  empty, then an explicit exception).

What was excluded / adapted (deviations from the upstream code)
---------------------------------------------------------------
1. NumPy reimplementation instead of TensorFlow/Keras. The project locks a
   TF-free dependency set; the upstream ``M2OE`` PyPI package pulls in
   ``tensorflow>=2.3`` and ``mlxtend==0.22.0``. Architectural widths,
   activations, loss, optimiser and defaults follow upstream exactly.
2. Group-level training. Upstream ``TabularGroupExplainer.compute_explanation``
   first trains one model per outlier *row* and then merges groups with an
   O(M^2)-training hierarchical tree to *discover* which outliers belong
   together. Our protocol pre-assigns groups (one abnormal equipment vs the
   normal pool), so we train exactly one shared-choice model per group
   (``num_groups=1`` semantics) and skip discovery entirely - infeasible at
   M~100-200 rows per equipment otherwise.
3. Deterministic execution: seeded weight initialisation (Glorot-uniform,
   the Keras default), seeded batch shuffling and seeded row subsampling.
   The upstream code does not seed TensorFlow.
4. The per-feature normal-dispersion statistic ``normal_dist`` is computed in
   closed form as ``2 * Var(normal_data)`` (unbiased), which is numerically
   identical to the upstream ``O(N^2 D)`` double-sum formula but avoids
   materialising an ``N x N x D`` tensor.
5. ``max_group_rows`` caps the anomalous group with a seeded deterministic
   subsample (scalability adaptation for the experiment matrix).
6. Upstream ``AETabularMM_SC.call`` returns the Keras ``mask`` keyword
   argument instead of the computed masks (an upstream bug); this port
   returns the computed mask tensor.
7. The paper's explanation is a binary subspace (``choice > threshold``).
   This port additionally exposes the continuous soft choice weights so a
   total order over features (a ranking) can be derived.

The published method is unsupervised: only raw process features of the
anomalous group and the normal reference pool are used; no KPI target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.neighbors import NearestNeighbors

_EPSILON = 1e-4  # upstream constant in the sample-distance denominator
_SQRT_GUARD = 1e-12  # guards sqrt() gradients at exactly-zero arguments


@dataclass(frozen=True)
class GroupExplanation:
    """Result of explaining one anomalous group against a normal pool.

    choice: Soft per-feature choice weights in ``[0, 1]``; the shared
        subspace weight vector learned by the masking model.
    choice_binary: ``choice`` binarised at the explainer threshold; the
        paper's notion of the explaining subspace.
    mean_abs_mask: Mean absolute learned patch magnitude per feature over
        the training pairs (counterfactual displacement scale).
    final_loss: Contrastive loss on the full pair set after training.
    attempts: Number of training restarts used (1-based).
    """

    choice: np.ndarray
    choice_binary: np.ndarray
    mean_abs_mask: np.ndarray
    final_loss: float
    attempts: int


class _Dense:
    """A dense layer with hand-written forward/backward passes."""

    def __init__(
        self,
        rng: np.random.Generator,
        n_in: int,
        n_out: int,
        activation: Literal["relu", "linear", "sigmoid"],
    ) -> None:
        limit = np.sqrt(6.0 / (n_in + n_out))  # Glorot uniform, the Keras default
        self.weights = rng.uniform(-limit, limit, size=(n_in, n_out))
        self.bias = np.zeros(n_out)
        self.activation = activation
        self._input: np.ndarray | None = None
        self._pre_activation: np.ndarray | None = None
        self._output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        z = x @ self.weights + self.bias
        self._pre_activation = z
        if self.activation == "relu":
            output = np.maximum(z, 0.0)
        elif self.activation == "sigmoid":
            output = 1.0 / (1.0 + np.exp(-z))
        else:
            output = z
        self._output = output
        return output

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(grad_input, grad_weights, grad_bias)`` for ``grad_output``."""
        if self._input is None or self._output is None or self._pre_activation is None:
            raise RuntimeError("backward called before forward")
        if self.activation == "relu":
            grad_output = grad_output * (self._pre_activation > 0.0)
        elif self.activation == "sigmoid":
            grad_output = grad_output * self._output * (1.0 - self._output)
        grad_weights = self._input.T @ grad_output
        grad_bias = grad_output.sum(axis=0)
        grad_input = grad_output @ self.weights.T
        return grad_input, grad_weights, grad_bias

    def parameters(self) -> list[np.ndarray]:
        return [self.weights, self.bias]


class _SharedChoiceMaskingModel:
    """NumPy port of the upstream ``TabularMM_SC`` masking model.

    ``CHOOSE``: constant ones input -> Dense(4D, relu) -> Dense(4D, relu) ->
    Dense(D, sigmoid), producing one shared weight vector ``c`` broadcast to
    every pair. ``MASK``: ``[O, R]`` concatenation -> Dense(4D) -> Dense(4D)
    -> Dense(D), all linear (matching upstream), producing per-pair patch
    magnitudes ``m``. The patched outlier is ``P = O + m * c``.
    """

    def __init__(
        self,
        n_features: int,
        normal_dist: np.ndarray,
        loss_weights: tuple[float, float, float],
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        width = 4 * n_features
        self.n_features = n_features
        self.normal_dist = normal_dist
        self.loss_weights = loss_weights
        self._choose_layers = [
            _Dense(rng, n_features, width, "relu"),
            _Dense(rng, width, width, "relu"),
            _Dense(rng, width, n_features, "sigmoid"),
        ]
        self._mask_layers = [
            _Dense(rng, 2 * n_features, width, "linear"),
            _Dense(rng, width, width, "linear"),
            _Dense(rng, width, n_features, "linear"),
        ]

    def parameters(self) -> list[np.ndarray]:
        layers = [*self._choose_layers, *self._mask_layers]
        return [parameter for layer in layers for parameter in layer.parameters()]

    def forward_choose(self) -> np.ndarray:
        """Return the shared soft choice weight vector of shape ``(D,)``."""
        hidden = np.ones((1, self.n_features))
        for layer in self._choose_layers:
            hidden = layer.forward(hidden)
        return hidden[0]

    def forward_mask(self, outliers: np.ndarray, references: np.ndarray) -> np.ndarray:
        """Return per-pair patch magnitudes of shape ``(B, D)``."""
        hidden = np.concatenate([outliers, references], axis=1)
        for layer in self._mask_layers:
            hidden = layer.forward(hidden)
        return hidden

    def loss_and_grads(self, outliers: np.ndarray, references: np.ndarray) -> tuple[float, list[np.ndarray]]:
        """Return the contrastive loss and gradients for all parameters."""
        alpha1, alpha2, alpha3 = self.loss_weights
        n_pairs, n_features = outliers.shape
        choice = self.forward_choose()
        mask = self.forward_mask(outliers, references)
        patches = outliers + mask * choice[np.newaxis, :]

        residual = patches - references
        squared_residual = residual**2
        squared_difference = (references - outliers) ** 2

        margin_squared = np.sum(squared_residual * choice[np.newaxis, :], axis=1)
        margin_n = np.sqrt(margin_squared) / np.sqrt(n_features)
        difference_reduced = np.sum(squared_difference * choice[np.newaxis, :] ** 2, axis=1)
        dispersion = np.sqrt(np.sum(self.normal_dist * choice))
        sample_distance = dispersion / (difference_reduced + _EPSILON)
        ndim = np.sqrt(np.sum(choice**2))
        loss = float(np.mean(alpha1 * margin_n + alpha2 * sample_distance) + alpha3 * ndim)

        grad_margin = alpha1 / n_pairs
        grad_distance = alpha2 / n_pairs
        grad_ndim = alpha3

        # margin_n = sqrt(sum(q * c)) / sqrt(D); d/dq and direct d/dc parts.
        sqrt_margin = np.sqrt(np.maximum(margin_squared, _SQRT_GUARD))
        grad_margin_squared = grad_margin / (2.0 * np.sqrt(n_features) * sqrt_margin)
        grad_choice = (grad_margin_squared[:, np.newaxis] * squared_residual).sum(axis=0)
        grad_residual = 2.0 * grad_margin_squared[:, np.newaxis] * choice[np.newaxis, :] * residual

        # sample_distance = dispersion / (difference_reduced + epsilon).
        grad_difference_reduced = -grad_distance * dispersion / (difference_reduced + _EPSILON) ** 2
        grad_dispersion = np.sum(grad_distance / (difference_reduced + _EPSILON))
        dispersion_guarded = np.sqrt(np.maximum(np.sum(self.normal_dist * choice), _SQRT_GUARD))
        grad_choice = grad_choice + (grad_dispersion / (2.0 * dispersion_guarded)) * self.normal_dist
        grad_choice = grad_choice + (
            2.0 * choice[np.newaxis, :] * squared_difference * grad_difference_reduced[:, np.newaxis]
        ).sum(axis=0)

        # ndim = sqrt(sum(c^2)).
        grad_choice = grad_choice + grad_ndim * choice / np.sqrt(np.maximum(np.sum(choice**2), _SQRT_GUARD))

        # patches = outliers + mask * choice ties both branches together.
        grad_mask = grad_residual * choice[np.newaxis, :]
        grad_choice = grad_choice + (grad_residual * mask).sum(axis=0)

        grads_by_param: dict[int, np.ndarray] = {}
        grad_input = grad_choice[np.newaxis, :]
        for layer in reversed(self._choose_layers):
            grad_input, grad_weights, grad_bias = layer.backward(grad_input)
            grads_by_param[id(layer.weights)] = grad_weights
            grads_by_param[id(layer.bias)] = grad_bias
        grad_input = grad_mask
        for layer in reversed(self._mask_layers):
            grad_input, grad_weights, grad_bias = layer.backward(grad_input)
            grads_by_param[id(layer.weights)] = grad_weights
            grads_by_param[id(layer.bias)] = grad_bias

        parameters = self.parameters()
        return loss, [grads_by_param[id(parameter)] for parameter in parameters]


class _Adam:
    """Standard Adam optimiser over a fixed list of parameter arrays."""

    def __init__(self, parameters: list[np.ndarray], learning_rate: float) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-7
        self._m = [np.zeros_like(parameter) for parameter in parameters]
        self._v = [np.zeros_like(parameter) for parameter in parameters]
        self._t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        if len(grads) != len(self.parameters):
            raise ValueError("gradient list length does not match parameters")
        self._t += 1
        for index, (parameter, grad) in enumerate(zip(self.parameters, grads, strict=True)):
            self._m[index] = self.beta1 * self._m[index] + (1.0 - self.beta1) * grad
            self._v[index] = self.beta2 * self._v[index] + (1.0 - self.beta2) * grad**2
            m_hat = self._m[index] / (1.0 - self.beta1**self._t)
            v_hat = self._v[index] / (1.0 - self.beta2**self._t)
            parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


def _normal_dispersion(normal_data: np.ndarray) -> np.ndarray:
    """Per-feature normal dispersion used by the contrastive term.

    Equivalent to the upstream pairwise double sum
    ``mean_i sum_j (N[i]-N[j])^2 / (N-1)`` (which simplifies to ``2 s^2``
    with the unbiased sample variance ``s^2``) without the ``O(N^2 D)`` cost.
    """
    dispersion = 2.0 * normal_data.var(axis=0, ddof=1)
    if not np.isfinite(dispersion).all() or (dispersion < 0).any():
        raise ValueError("normal reference data must produce finite non-negative dispersion statistics")
    return dispersion


def _validate_arrays(outliers: np.ndarray, normal_data: np.ndarray) -> None:
    if outliers.ndim != 2 or normal_data.ndim != 2:
        raise ValueError("outliers and normal_data must be 2-D arrays")
    if outliers.shape[1] != normal_data.shape[1]:
        raise ValueError("outliers and normal_data must share the feature dimension")
    if outliers.shape[0] == 0 or normal_data.shape[0] < 2:
        raise ValueError("outliers must be non-empty and normal_data needs at least 2 rows")
    if not np.isfinite(outliers).all() or not np.isfinite(normal_data).all():
        raise ValueError("outliers and normal_data must contain finite values only")


class TabularGroupExplainer:
    """Explain one anomalous group against a normal pool (shared subspace).

    One shared-choice masking model is trained on outlier/reference pairs
    built from per-outlier nearest-neighbour reference sets, using the
    subspace density contrastive loss of Angiulli et al. (2024). The learned
    soft choice weights provide a deterministic per-feature ranking score.

    loss_weights: ``[alpha1, alpha2, alpha3]`` weights for patch proximity,
        subspace contrast and choice sparsity (upstream default).
    lr: Adam learning rate (upstream default).
    epochs: Training epochs per (re)start.
    batch_size: Mini-batch size (upstream default).
    n_neighbors: Reference-set size per outlier row (upstream default).
    threshold: Binarisation threshold applied to the soft choice.
    num_tries: Maximum training restarts when the binarised choice is empty;
        an error is raised when all restarts fail.
    seed: Seed for initialisation, shuffling and subsampling.
    max_group_rows: Cap on anomalous-group rows (seeded subsample); ``None``
        keeps the full group.
    """

    def __init__(
        self,
        loss_weights: tuple[float, float, float] = (1.0, 1.0, 0.5),
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 16,
        n_neighbors: int = 30,
        threshold: float = 0.5,
        num_tries: int = 3,
        seed: int = 42,
        max_group_rows: int | None = 128,
    ) -> None:
        """Validate and store the explainer configuration (see class docstring)."""
        if len(loss_weights) != 3 or min(loss_weights) < 0:
            raise ValueError("loss_weights must contain three non-negative weights")
        if epochs < 1 or batch_size < 1 or n_neighbors < 1 or num_tries < 1:
            raise ValueError("epochs, batch_size, n_neighbors and num_tries must be positive")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie strictly between 0 and 1")
        if max_group_rows is not None and max_group_rows < 1:
            raise ValueError("max_group_rows must be positive when provided")
        self.loss_weights = tuple(float(weight) for weight in loss_weights)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.n_neighbors = int(n_neighbors)
        self.threshold = float(threshold)
        self.num_tries = int(num_tries)
        self.seed = int(seed)
        self.max_group_rows = max_group_rows

    def _training_pairs(self, outliers: np.ndarray, normal_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selector = NearestNeighbors(n_neighbors=self.n_neighbors)
        selector.fit(normal_data)
        _, neighbour_index = selector.kneighbors(outliers)
        outlier_pairs = np.repeat(outliers, self.n_neighbors, axis=0)
        reference_pairs = normal_data[neighbour_index.reshape(-1)]
        return outlier_pairs, reference_pairs

    def _train_once(
        self,
        outlier_pairs: np.ndarray,
        reference_pairs: np.ndarray,
        normal_dist: np.ndarray,
        seed: int,
    ) -> tuple[_SharedChoiceMaskingModel, float]:
        model = _SharedChoiceMaskingModel(
            n_features=outlier_pairs.shape[1],
            normal_dist=normal_dist,
            loss_weights=self.loss_weights,  # type: ignore[arg-type]
            seed=seed,
        )
        optimizer = _Adam(model.parameters(), learning_rate=self.lr)
        rng = np.random.default_rng(seed)
        n_pairs = len(outlier_pairs)
        for _epoch in range(self.epochs):
            order = rng.permutation(n_pairs)
            for start in range(0, n_pairs, self.batch_size):
                batch = order[start : start + self.batch_size]
                _loss, grads = model.loss_and_grads(outlier_pairs[batch], reference_pairs[batch])
                optimizer.step(grads)
        final_loss, _grads = model.loss_and_grads(outlier_pairs, reference_pairs)
        return model, final_loss

    def explain_group(self, outliers: np.ndarray, normal_data: np.ndarray) -> GroupExplanation:
        """Learn the shared subspace distinguishing ``outliers`` from ``normal_data``."""
        _validate_arrays(outliers, normal_data)
        if self.n_neighbors > normal_data.shape[0]:
            raise ValueError(f"n_neighbors={self.n_neighbors} exceeds the normal pool size {normal_data.shape[0]}")
        rng = np.random.default_rng(self.seed)
        if self.max_group_rows is not None and len(outliers) > self.max_group_rows:
            keep = np.sort(rng.choice(len(outliers), size=self.max_group_rows, replace=False))
            group = outliers[keep]
        else:
            group = outliers
        normal_dist = _normal_dispersion(normal_data)
        outlier_pairs, reference_pairs = self._training_pairs(group, normal_data)

        for attempt in range(1, self.num_tries + 1):
            model, final_loss = self._train_once(outlier_pairs, reference_pairs, normal_dist, self.seed + attempt)
            choice = model.forward_choose()
            if (choice > self.threshold).any():
                return GroupExplanation(
                    choice=choice,
                    choice_binary=(choice > self.threshold).astype(float),
                    mean_abs_mask=np.abs(model.forward_mask(outlier_pairs, reference_pairs)).mean(axis=0),
                    final_loss=final_loss,
                    attempts=attempt,
                )
        raise RuntimeError(
            "M2OE explanation process failed: empty feature choice after all restarts; "
            "try a lower sparsity weight (alpha3)"
        )
