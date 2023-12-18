import argparse
import os
import time
from pprint import pformat

import equinox as eqx
import jax
import optax

import qvix
from qvix.builder import build_dataloader, build_object
from qvix.registry import BackboneRegistry, OptimizerRegistry, SchedulerRegistry
from qvix.utils import (cvt_cfgPathToDict, evaluate, get_logger,
                        loggin_gpu_info, loggin_system_info, make_step)

parser = argparse.ArgumentParser(description="Train classification model.")
parser.add_argument("cfg", type=str, help="Path to configuration file.")


def main() -> None:
    args = parser.parse_args()
    cfg_path = args.cfg

    cfg = cvt_cfgPathToDict(cfg_path)
    seed = cfg['seed']
    iterations = cfg['iterations']
    work_dir = cfg['work_dir']
    log_interval = cfg['log_interval']
    resume_from_cfg = cfg['resume_from']
    load_from_cfg = cfg['load_from']

    if resume_from_cfg is not None and load_from_cfg is not None:
        raise ValueError(
            "Only one of resume_from and load_from can be specified.")

    model_cfg = cfg['model']
    loss_cfg = cfg['loss']

    trainloader_cfg = cfg['train_loader']
    testloader_cfg = cfg['test_loader']

    if 'optimizer' in cfg:
        optimizer_cfg = cfg['optimizer']
    elif 'optimizer_chain' in cfg:
        optimizer_cfg = cfg['optimizer_chain']

    os.makedirs(cfg['work_dir'], exist_ok=True)

    logger = get_logger(work_dir)
    logger.info(f"Configuration file: {cfg_path}")
    logger.info(f"Configuration: {os.linesep + pformat(cfg)}")
    logger.info(f"JAX devices: {jax.devices()}")

    loggin_system_info(logger)
    loggin_gpu_info(logger)

    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Equinox version: {eqx.__version__}")
    logger.info(f"Optax version: {optax.__version__}")
    logger.info(f"Qvix version: {qvix.__version__}")

    key = jax.random.PRNGKey(seed)

    model_name = model_cfg.pop("name")
    model = BackboneRegistry(model_name)
    model, model_state = eqx.nn.make_with_state(model)(key, **model_cfg)

    logger.info(f"Model: {model_name}")
    logger.info(f"Model Architecture: {model}")

    start_iteration = 1
    if resume_from_cfg is not None:
        resume_from = os.path.join(work_dir, resume_from_cfg)
        model = eqx.tree_deserialise_leaves(resume_from, model)
        stopped_iteration = int(resume_from.split("_")[-1].split(".")[0])
        logger.info(f"Resume from Iteration: {stopped_iteration}")
        start_iteration = stopped_iteration + 1
    elif load_from_cfg is not None:
        load_from = os.path.join(load_from_cfg)
        model = eqx.tree_deserialise_leaves(load_from, model)
        logger.info(f"Load from {load_from}")

    trainloader = build_dataloader(trainloader_cfg)
    testloader = build_dataloader(testloader_cfg)

    
    if 'scheduler' in optimizer_cfg:
        scheduler_cfg = optimizer_cfg.pop('scheduler')
        scheduler = build_object(scheduler_cfg, SchedulerRegistry)
        optimizer_cfg['learning_rate'] = scheduler

    optimizer = build_object(optimizer_cfg, OptimizerRegistry)

    optimizer = optax.chain(
        optax.add_decayed_weights(weight_decay=5e-4),
        optimizer
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


    def init_weight(model,  key):
        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        get_weights = lambda m: [x.weight
                                for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                                if is_linear(x)]
        weights = get_weights(model)

        initializer = jax.nn.initializers.normal(stddev=1e-3)
        new_weights = [initializer(subkey, weight.shape)
                        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
        model = eqx.tree_at(get_weights, model, new_weights)

        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        get_bias = lambda m: [x.bias
                                for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                                if is_linear(x)]
        biases = get_bias(model)
        initializer = jax.nn.initializers.constant(0)
        new_bias = [initializer(subkey, bias.shape)
                        for bias, subkey in zip(biases, jax.random.split(key, len(biases)))]
        model = eqx.tree_at(get_bias, model, new_bias)

        ################################

        is_conv = lambda x: isinstance(x, eqx.nn.Conv2d)
        get_weights = lambda m: [x.weight
                                for x in jax.tree_util.tree_leaves(m, is_leaf=is_conv)
                                if is_conv(x)]
        weights = get_weights(model)

        initializer = jax.nn.initializers.kaiming_normal()
        new_weights = [initializer(subkey, weight.shape)
                        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
        model = eqx.tree_at(get_weights, model, new_weights)

        is_conv = lambda x: isinstance(x, eqx.nn.Linear)
        get_bias = lambda m: [x.bias
                                for x in jax.tree_util.tree_leaves(m, is_leaf=is_conv)
                                if is_linear(x)]
        biases = get_bias(model)
        initializer = jax.nn.initializers.constant(0)
        new_bias = [initializer(subkey, bias.shape)
                        for bias, subkey in zip(biases, jax.random.split(key, len(biases)))]
        model = eqx.tree_at(get_bias, model, new_bias)

        ################################

        is_bn = lambda x: isinstance(x, eqx.nn.Conv2d)
        get_weights = lambda m: [x.weight
                                for x in jax.tree_util.tree_leaves(m, is_leaf=is_bn)
                                if is_bn(x)]
        weights = get_weights(model)

        initializer = jax.nn.initializers.constant(1)
        new_weights = [initializer(subkey, weight.shape)
                        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
        model = eqx.tree_at(get_weights, model, new_weights)

        is_bn = lambda x: isinstance(x, eqx.nn.Linear)
        get_bias = lambda m: [x.bias
                                for x in jax.tree_util.tree_leaves(m, is_leaf=is_bn)
                                if is_linear(x)]
        biases = get_bias(model)
        initializer = jax.nn.initializers.constant(0)
        new_bias = [initializer(subkey, bias.shape)
                        for bias, subkey in zip(biases, jax.random.split(key, len(biases)))]
        model = eqx.tree_at(get_bias, model, new_bias)


        return model
    
    model = init_weight(model, key)



    start_time = time.time()
    logger.info(f"Start training at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def infinite_trainloader():
        while True:
            yield from trainloader

    for i, (x, y) in zip(range(start_iteration, iterations + 1),
                         infinite_trainloader()):

        x = x.numpy()
        y = y.numpy().astype(int)

        model, model_state, opt_state, train_loss, train_acc = make_step(
            model, model_state, opt_state, optimizer, key, loss_cfg, x, y)

        if i % log_interval == 0:
            logger.info(
                f"Iteration: {i}, Loss: {train_loss:.4f}, Accuracy: {train_acc * 100:.2f}%"
            )

        if i % cfg['checkpoint_interval'] == 0:
            save_checkpoint = os.path.join(work_dir, f"iteration_{i}.eqx")
            eqx.tree_serialise_leaves(save_checkpoint, model)

        if i % cfg['validate_interval'] == 0:
            model = eqx.nn.inference_mode(model)
            val_loss, val_acc = evaluate(model, testloader, key, loss_cfg,
                                         model_state)
            logger.info(
                f"Iteration: {i}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc * 100:.2f}%"
            )
            model = eqx.nn.inference_mode(model, value=False)

    end_time = time.time()
    logger.info(f"End training at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    logger.info(
        f"Training time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")


if __name__ == "__main__":
    main()
