import argparse
import os
import time
from pprint import pformat

import equinox as eqx
import jax
import optax

import qvix
from qvix.builder import build_dataloader, build_object
from qvix.registry import BackboneRegistry, OptimizerRegistry
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
    optimizer_cfg = cfg['optimizer']

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

    model = build_object(model_cfg, BackboneRegistry, key)

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
    optimizer = build_object(optimizer_cfg, OptimizerRegistry)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    start_time = time.time()
    logger.info(f"Start training at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def infinite_trainloader():
        while True:
            yield from trainloader

    for i, (x, y) in zip(range(start_iteration, iterations + 1),
                         infinite_trainloader()):

        x = x.numpy()
        y = y.numpy().astype(int)

        model, opt_state, train_loss, train_acc = make_step(
            model, opt_state, optimizer, key, loss_cfg, x, y)

        if i % log_interval == 0:
            logger.info(
                f"Iteration: {i}, Loss: {train_loss:.4f}, Accuracy: {train_acc * 100:.2f}%"
            )

        if i % cfg['checkpoint_interval'] == 0:
            save_checkpoint = os.path.join(work_dir, f"iteration_{i}.eqx")
            eqx.tree_serialise_leaves(save_checkpoint, model)

        if i % cfg['validate_interval'] == 0:
            val_loss, val_acc = evaluate(model, testloader, key, loss_cfg)
            logger.info(
                f"Iteration: {i}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc * 100:.2f}%"
            )

    end_time = time.time()
    logger.info(f"End training at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)

    logger.info(
        f"Training time: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")


if __name__ == "__main__":
    main()
