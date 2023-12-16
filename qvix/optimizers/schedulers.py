from typing import Any, Optional, Union

import jax
import optax

from qvix.registry import SchedulerRegistry

ScalarOrSchedule = Union[float, jax.Array, optax.Schedule]


@SchedulerRegistry.register()
def WarmupCosineDecay(init_value: float,
                      peak_value: float,
                      warmup_steps: int,
                      decay_steps: int,
                      end_value: float = 0.0,
                      exponent: float = 1.0) -> optax.Schedule:
    """Warmup cosine decay schedule from optax."""

    return optax.warmup_cosine_decay_schedule(init_value=init_value,
                                              peak_value=peak_value,
                                              warmup_steps=warmup_steps,
                                              decay_steps=decay_steps,
                                              end_value=end_value,
                                              exponent=exponent)
