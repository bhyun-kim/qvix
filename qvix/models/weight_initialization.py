import equinox as eqx
import jax


def initialize_layer(model: eqx.Module, key: jax.random.PRNGKey,
                     init_cfg: dict) -> eqx.Module:
    """
    Args:
        model (eqx.Module)
        key (jax.random.PRNGKey)
        init_cfg (dict)
    """

    layer_name = init_cfg['layer']
    if hasattr(eqx.nn, layer_name):
        layer_class = getattr(eqx.nn, layer_name)
        is_target = lambda x: isinstance(x, layer_class)
    else:
        raise NotImplementedError(f"Unknown layer: {layer_name}")

    get_weights = lambda m: [
        x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_target)
        if is_target(x)
    ]
    weights = get_weights(model)

    initializer_name = init_cfg['initializer']
    if hasattr(jax.nn.initializers, initializer_name):
        initializer = getattr(jax.nn.initializers,
                              initializer_name)(**init_cfg['kwargs'])
    else:
        raise ValueError(f"Unknown initializer: {initializer_name}")

    new_weights = [
        initializer(subkey, weight.shape)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    model = eqx.tree_at(get_weights, model, new_weights)

    return model


def initialize_model(model: eqx.Module, key: jax.random.PRNGKey,
                     init_cfgs: list) -> eqx.Module:
    """Initialize model.

    Args:
        model (eqx.Module)
        key (jax.random.PRNGKey)

    Returns:
        model (eqx.Module)
    """
    for init_cfg in init_cfgs:

        model = initialize_layer(model, key, init_cfg)

    return model
