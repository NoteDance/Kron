# Kron

**Overview**:

The `Kron` optimizer implements the PSGD Kron algorithm, which uses Kronecker-based preconditioning to accelerate stochastic gradient descent. By maintaining a set of per-parameter preconditioners (built via Kronecker products) and updating them probabilistically during training, Kron adapts the effective gradient direction and scaling. This method is particularly useful for large models where efficient preconditioning can significantly improve convergence while managing memory consumption.

**Parameters**:

- **`learning_rate`** *(float, default=0.0003)*: The base step size for updating model parameters.
- **`weight_decay`** *(float, default=0.0)*: Coefficient for weight decay regularization. When non-zero, weight decay is applied either additively to the gradients or directly to the parameters based on the `decoupled` flag.
- **`b1`** *(float, default=0.9)*: Exponential decay rate used in updating the momentum buffer.
- **`preconditioner_update_probability`** *(callable or float, optional)*: The probability schedule controlling how frequently the preconditioner is updated. If not provided, a default schedule (flat start for 500 steps then exponential annealing) is used.
- **`max_size_triangular`** *(int, default=8192)*: Maximum size for using a full (triangular) preconditioner; dimensions larger than this use a diagonal approximation.
- **`min_ndim_triangular`** *(int, default=2)*: Minimum number of dimensions required for a tensor to receive a triangular (non-diagonal) preconditioner.
- **`memory_save_mode`** *(str, optional)*: Option to control memory usage for preconditioners. Options include `None`, `"smart_one_diag"`, `"one_diag"`, and `"all_diag"`.
- **`momentum_into_precond_update`** *(bool, default=True)*: Determines whether the momentum buffer (updated with decay `b1`) is used when updating the preconditioner.
- **`precond_lr`** *(float, default=0.1)*: Learning rate specifically used for preconditioner updates.
- **`precond_init_scale`** *(float, default=1.0)*: Initial scaling factor for the preconditioners.
- **`mu_dtype`** *(dtype, optional)*: Data type for the momentum buffer; if specified, momentum values are cast to this type.
- **`precond_dtype`** *(dtype, default=tf.float32)*: Data type for the preconditioners and related computations.
- **`clipnorm`** *(float, optional)*: If set, gradients are clipped to this maximum norm.
- **`clipvalue`** *(float, optional)*: If set, gradients are clipped element-wise to this maximum absolute value.
- **`global_clipnorm`** *(float, optional)*: If set, the global norm of all gradients is clipped to this value.
- **`use_ema`** *(bool, default=False)*: Whether to use an Exponential Moving Average (EMA) of the model weights during training.
- **`ema_momentum`** *(float, default=0.99)*: Momentum factor for EMA updates.
- **`ema_overwrite_frequency`** *(int, optional)*: Frequency for overwriting EMA weights.
- **`loss_scale_factor`** *(float, optional)*: A scaling factor for the loss during gradient computation, useful for mixed precision training.
- **`gradient_accumulation_steps`** *(int, optional)*: The number of steps over which gradients are accumulated before updating parameters.
- **`name`** *(str, default="kron")*: The name of the optimizer.

---

**Example Usage**:
```python
import tensorflow as tf
from kron import Kron

# Instantiate the Kron optimizer with default preconditioner schedule.
optimizer = Kron(
    learning_rate=0.0003,
    weight_decay=1e-4,
    b1=0.9,
    # preconditioner_update_probability can be omitted to use the default schedule
    max_size_triangular=8192,
    min_ndim_triangular=2,
    memory_save_mode="smart_one_diag",
    momentum_into_precond_update=True,
    precond_lr=0.1,
    precond_init_scale=1.0,
    mu_dtype=tf.float32,
    precond_dtype=tf.float32,
)

# Compile a Keras model using the Kron optimizer.
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
```
