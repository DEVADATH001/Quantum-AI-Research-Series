"""Author: DEVADATH H K

Quantum AI Research Series

Project 03: Quantum Kernel SVM MNIST
Task: Quantum Kernel Learning (QKL) — train feature-map parameters by
      maximising Centered Kernel-Target Alignment (cKTA) via the parameter-shift rule.

Mathematical foundation
-----------------------
Let φ(x; θ) be the ZZ feature map with trainable parameter vector θ ∈ ℝ^p.
The centered kernel at θ is:

    K_θ[i,j] = |⟨φ(xᵢ; θ)|φ(xⱼ; θ)⟩|²

The objective is:

    θ* = argmax_{θ} cKTA(K_θ, Y)
         = argmax_{θ} ⟨H K_θ H, H Y H⟩_F / (‖H K_θ H‖_F · ‖H Y H‖_F)

where H = I - (1/n) 11ᵀ is the centering matrix.

Gradient via parameter-shift rule (Mitarai et al. 2018, Liu et al. 2021):

    ∂cKTA / ∂θ_k ≈ [cKTA(K_{θ + (π/2) eₖ}) − cKTA(K_{θ − (π/2) eₖ})] / 2

This is exact (not approximate) for gates of the form exp(−i θ Pₖ/2) where Pₖ
is a Pauli generator — which is exactly what ZZ and RZ-based feature maps use.

Optimizer: Adam (Kingma & Ba 2015) with gradient *ascent*:
    m_t = β₁ m_{t-1} + (1−β₁) g_t
    v_t = β₂ v_{t-1} + (1−β₂) g_t²
    θ_t = θ_{t-1} + lr · m̂_t / (√v̂_t + ε)

Reference
---------
Glick et al. (2024) "Covariant quantum kernels for data with group structure",
PRX Quantum 5, 020343.
Liu et al. (2021) "A rigorous and robust quantum speed-up in supervised machine
learning", Nature Physics 17, 1013-1017.
Mitarai et al. (2018) "Quantum circuit learning", Physical Review A 98, 032309.
Cortes et al. (2012) "Algorithms for Learning Kernels Based on Centered Alignment",
JMLR 13, 795-828.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Any

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _centered_kta(K: np.ndarray, Y_c: np.ndarray) -> float:
    """cKTA with pre-computed centred target Y_c = H Y H.

    Separating pre-computation of Y_c avoids recomputing it at every gradient
    evaluation — it is constant throughout training.

    Args:
        K: Kernel matrix (n×n).
        Y_c: Pre-centred target kernel (n×n).

    Returns:
        Scalar cKTA ∈ [−1, 1].
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H

    frob_inner = float(np.sum(K_c * Y_c))
    norm_K_c = float(np.linalg.norm(K_c, "fro"))
    norm_Y_c = float(np.linalg.norm(Y_c, "fro"))

    if norm_K_c < 1e-15 or norm_Y_c < 1e-15:
        return 0.0
    return frob_inner / (norm_K_c * norm_Y_c)


def _build_param_map(
    base_circuit: QuantumCircuit,
    pv: ParameterVector,
    values: np.ndarray,
) -> QuantumCircuit:
    """Bind *values* into *base_circuit* using parameter vector *pv*.

    Returns a fully-bound QuantumCircuit with no free parameters.
    """
    assert len(pv) == len(values), (
        f"Parameter vector length {len(pv)} != values length {len(values)}"
    )
    binding = {pv[i]: float(values[i]) for i in range(len(pv))}
    return base_circuit.assign_parameters(binding)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QuantumKernelLearner:
    """Train the parameters of a parameterised quantum feature map by maximising
    cKTA via the parameter-shift rule and Adam gradient ascent.

    Usage
    -----
    >>> from qiskit.circuit.library import ZZFeatureMap
    >>> fm = ZZFeatureMap(feature_dimension=4, reps=1, parameter_prefix="theta")
    >>> qkl = QuantumKernelLearner(
    ...     feature_map=fm,
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     lr=0.05,
    ...     n_epochs=30,
    ... )
    >>> opt_params, history = qkl.fit()
    >>> opt_fm = qkl.get_optimized_feature_map()

    Parameters
    ----------
    feature_map:
        A Qiskit ``QuantumCircuit`` that uses a ``ParameterVector`` for its
        trainable rotation angles.  The circuit may also use a **data**
        ``ParameterVector`` (for input features); QKL will only optimise the
        *trainable* parameters (those whose names start with `parameter_prefix`).
    X_train:
        Training feature matrix (n_samples × n_features).  Must already be
        scaled for angle encoding (e.g. [0, π/2]).
    y_train:
        Binary label vector of length n_samples.
    lr:
        Adam learning rate (default 0.05).  Values in [0.01, 0.1] are typical.
    n_epochs:
        Number of gradient-ascent epochs.
    batch_size:
        Number of training samples used per epoch for kernel evaluation.
        Smaller batches reduce circuit evaluations per step but increase noise.
    seed:
        RNG seed for reproducible mini-batch selection.
    parameter_prefix:
        Name prefix that identifies *trainable* parameters inside the circuit.
        Parameters whose names do NOT start with this prefix are treated as
        data-binding parameters and are excluded from optimisation.
    adam_beta1, adam_beta2, adam_eps:
        Adam hyper-parameters.  Defaults match Kingma & Ba (2015).
    shift:
        Parameter-shift amount.  For standard Pauli generators this is π/2
        (exact gradient formula).
    """

    def __init__(
        self,
        feature_map: QuantumCircuit,
        X_train: np.ndarray,
        y_train: np.ndarray,
        spsa_a: float = 0.2,
        spsa_c: float = 0.1,
        spsa_alpha: float = 0.602,
        spsa_gamma: float = 0.101,
        spsa_A: float = 0.0,
        n_epochs: int = 30,
        batch_size: int = 20,
        seed: int = 42,
        parameter_prefix: str = "θ",
        sampler: Any = None,
    ) -> None:
        self._base_fm = deepcopy(feature_map)
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train)
        self.spsa_a = spsa_a
        self.spsa_c = spsa_c
        self.spsa_alpha = spsa_alpha
        self.spsa_gamma = spsa_gamma
        self.spsa_A = spsa_A
        self.n_epochs = n_epochs
        self.batch_size = min(batch_size, len(X_train))
        self.seed = seed
        self.parameter_prefix = parameter_prefix
        self.sampler = sampler

        # Separate trainable from data parameters
        all_params = sorted(feature_map.parameters, key=lambda p: p.name)
        self._train_params = [
            p for p in all_params if p.name.startswith(parameter_prefix)
        ]
        self._data_params = [
            p for p in all_params if not p.name.startswith(parameter_prefix)
        ]

        if not self._train_params:
            raise ValueError(
                f"No trainable parameters found with prefix '{parameter_prefix}'. "
                f"Available parameters: {[p.name for p in all_params]}. "
                "Ensure your feature map uses a ParameterVector whose names start "
                "with the parameter_prefix."
            )

        self.n_params = len(self._train_params)

        # Initialise parameters: small random values in [−π/4, π/4] to break
        # symmetry while staying near the default circuit structure.
        rng = np.random.default_rng(seed)
        self._params = rng.uniform(-np.pi / 4, np.pi / 4, size=self.n_params)

        # SPSA state
        self._t = 0

        self._history: list[dict[str, Any]] = []
        self._optimized_params: np.ndarray | None = None

        logger.info(
            "QuantumKernelLearner initialised: %d trainable params, %d data params, "
            "n_epochs=%d, batch_size=%d (SPSA)",
            self.n_params, len(self._data_params), n_epochs, self.batch_size,
        )

    # ------------------------------------------------------------------
    # Kernel evaluation helpers
    # ------------------------------------------------------------------

    def _eval_kernel(
        self,
        train_params: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """Evaluate K_θ(X, X) using Qiskit's FidelityQuantumKernel.

        This routes execution through Native Qiskit Primitives (Samplers), resolving
        hardware acceleration bottlenecks and enabling direct QPU training loops.
        
        Returns:
            Kernel matrix of shape (n, n).
        """
        from src.quantum_kernel_engine import create_quantum_kernel
        
        # Bind only the trainable parameters, leaving data variables free
        binding: dict = {
            self._train_params[k]: float(train_params[k])
            for k in range(self.n_params)
        }
        bound_fm = self._base_fm.assign_parameters(binding)
        
        # Instantiate kernel with the configured backend sampler
        kernel = create_quantum_kernel(feature_map=bound_fm, sampler=self.sampler)
        
        # Qiskit primitive implicitly constructs the Gram matrix logic natively
        K = kernel.evaluate(x_vec=X)
        return K

    # ------------------------------------------------------------------
    # Gradient
    # ------------------------------------------------------------------

    def _spsa_gradient(
        self,
        params: np.ndarray,
        X_batch: np.ndarray,
        Y_c: np.ndarray,
        t: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        """SPSA finite-difference gradient.

        Evaluates cKTA exactly twice per step, independent of parameter count.
        
        Returns:
            (grad_vector, current_ckta_approx)
        """
        ak = self.spsa_a / (t + self.spsa_A) ** self.spsa_alpha
        ck = self.spsa_c / (t ** self.spsa_gamma)
        
        # Bernoulli random perturbation vector
        delta = rng.choice([-1.0, 1.0], size=self.n_params)
        
        p_plus = params + ck * delta
        K_plus = self._eval_kernel(p_plus, X_batch)
        ckta_plus = _centered_kta(K_plus, Y_c)
        
        p_minus = params - ck * delta
        K_minus = self._eval_kernel(p_minus, X_batch)
        ckta_minus = _centered_kta(K_minus, Y_c)
        
        # Gradient approximation
        grad = (ckta_plus - ckta_minus) / (2.0 * ck) * delta
        
        # We can approximately track current cKTA without a 3rd evaluation by averaging
        cKTA_approx = float((ckta_plus + ckta_minus) / 2.0)
        
        return grad, cKTA_approx, ak

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Run QKL training loop.

        Returns:
            (optimal_params, history) where:
            - optimal_params: θ* ∈ ℝ^p that achieved maximum cKTA.
            - history: list of dicts with keys
                {"epoch", "ckta", "grad_norm", "wall_time_s"}.
        """
        rng = np.random.default_rng(self.seed)
        n = len(self.X_train)

        # Pre-compute the centred target kernel (constant throughout training)
        unique_y = np.unique(self.y_train)
        y_mapped = np.where(self.y_train == unique_y[0], -1.0, 1.0)
        Y_full = np.outer(y_mapped, y_mapped)
        H_full = np.eye(n) - np.ones((n, n)) / n
        # Pre-centre Y for the FULL training set (reference for reporting)
        Y_c_full = H_full @ Y_full @ H_full

        best_ckta = -np.inf
        best_params = self._params.copy()

        logger.info("Starting QKL training: %d epochs × %d params", self.n_epochs, self.n_params)
        t0_total = time.time()

        for epoch in range(1, self.n_epochs + 1):
            t0_epoch = time.time()

            # Mini-batch selection
            idx = rng.choice(n, size=self.batch_size, replace=False)
            X_batch = self.X_train[idx]
            y_batch = self.y_train[idx]

            # Build centred target for this batch
            nb = len(X_batch)
            y_b_mapped = np.where(y_batch == unique_y[0], -1.0, 1.0)
            Y_batch = np.outer(y_b_mapped, y_b_mapped)
            H_batch = np.eye(nb) - np.ones((nb, nb)) / nb
            Y_c_batch = H_batch @ Y_batch @ H_batch

            # SPSA Gradient
            grad, ckta_val, ak = self._spsa_gradient(self._params, X_batch, Y_c_batch, epoch, rng)
            grad_norm = float(np.linalg.norm(grad))

            # Ascent step
            self._params = self._params + ak * grad

            epoch_time = time.time() - t0_epoch
            entry: dict[str, Any] = {
                "epoch": epoch,
                "ckta": float(ckta_val),
                "grad_norm": float(grad_norm),
                "wall_time_s": float(epoch_time),
            }
            self._history.append(entry)

            if ckta_val > best_ckta:
                best_ckta = ckta_val
                best_params = self._params.copy()

            logger.info(
                "QKL epoch %2d/%d  cKTA=%.4f  |grad|=%.4e  t=%.1fs",
                epoch, self.n_epochs, ckta_val, grad_norm, epoch_time,
            )

        total_time = time.time() - t0_total
        logger.info(
            "QKL training complete: best cKTA=%.4f at epoch — total %.1fs",
            best_ckta, total_time,
        )

        self._optimized_params = best_params
        return best_params, self._history

    def get_optimized_feature_map(self) -> QuantumCircuit:
        """Return a copy of the feature map with trainable parameters bound to θ*.

        The data parameters remain free (as ParameterExpression objects) so
        the returned circuit can still be used as a kernel circuit.

        Raises:
            RuntimeError: If `fit()` has not been called yet.
        """
        if self._optimized_params is None:
            raise RuntimeError("Call fit() before get_optimized_feature_map().")

        fm = deepcopy(self._base_fm)
        binding = {
            self._train_params[k]: float(self._optimized_params[k])
            for k in range(self.n_params)
        }
        return fm.assign_parameters(binding)

    @property
    def history(self) -> list[dict[str, Any]]:
        """Training history: list of per-epoch dicts."""
        return list(self._history)

    @property
    def optimized_params(self) -> np.ndarray | None:
        """Optimal parameter vector θ*, or None before fit()."""
        return self._optimized_params
