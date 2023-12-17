import optax

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