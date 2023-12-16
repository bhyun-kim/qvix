from typing import Any, Optional, Union

import jax
import optax

from qvix.registry import OptimizerRegistry

ScalarOrSchedule = Union[float, jax.Array, optax.Schedule]


@OptimizerRegistry.register()
def Adam(learning_rate: ScalarOrSchedule,
         b1: float = 0.9,
         b2: float = 0.999,
         eps: float = 1e-8,
         eps_root: float = 0.0,
         mu_dtype: Optional[Any] = None) -> optax.GradientTransformation:
    """Adam optimizer from optax."""

    return optax.adam(learning_rate=learning_rate,
                      b1=b1,
                      b2=b2,
                      eps=eps,
                      eps_root=eps_root,
                      mu_dtype=mu_dtype)

@OptimizerRegistry.register()
def AdamW(learning_rate: ScalarOrSchedule,
          b1: float = 0.9,
          b2: float = 0.999,
          eps: float = 1e-8,
          eps_root: float = 0.0,
          mu_dtype: Optional[Any] = None,
          weight_decay=0.0001, 
          mask=None) -> optax.GradientTransformation:
    """AdamW optimizer from optax."""

    return optax.adamw(learning_rate=learning_rate,
                       b1=b1,
                       b2=b2,
                       eps=eps,
                       eps_root=eps_root,
                       mu_dtype=mu_dtype,
                       weight_decay=weight_decay,
                       mask=mask)

@OptimizerRegistry.register()
def SGD(learning_rate: ScalarOrSchedule,
        momentum: float = 0.0,
        nesterov: bool = False) -> optax.GradientTransformation:
    """SGD optimizer from optax."""

    return optax.sgd(learning_rate=learning_rate,
                     momentum=momentum,
                     nesterov=nesterov)