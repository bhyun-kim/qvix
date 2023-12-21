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

    layer_name = init_cfg.pop('layer')
    if hasattr(eqx.nn, layer_name):
        layer_class = getattr(eqx.nn, layer_name)
        is_target = lambda x: isinstance(x, layer_class)
    else:
        raise NotImplementedError(f"Unknown layer: {layer_name}")

    if 'target' not in init_cfg:
        init_cfg['target'] = 'weight'

    if init_cfg['target'] == 'weight':
        init_cfg.pop('target')
        get_targets = lambda m: [
            x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_target)
            if is_target(x)
        ]
    elif init_cfg['target'] == 'bias':
        init_cfg.pop('bias')
        get_targets = lambda m: [
            x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_target)
            if is_target(x)
        ]

    targets = get_targets(model)

    initializer_name = init_cfg['initializer']
    if hasattr(jax.nn.initializers, initializer_name):
        initializer = getattr(jax.nn.initializers,
                              initializer_name)(**init_cfg)
    else:
        raise ValueError(f"Unknown initializer: {initializer_name}")

    new_weights = [
        initializer(subkey, weight.shape)
        for weight, subkey in zip(targets, jax.random.split(key, len(targets)))
    ]
    model = eqx.tree_at(get_targets, model, new_weights)

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
