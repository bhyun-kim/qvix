import argparse
import os
import time
from pprint import pformat

import equinox as eqx
import jax
import optax

import qvix
from qvix.models import initialize_model
from qvix.builder import build_backbone, build_dataloader, build_optimizer_chain, build_optax_object, OptaxLossFunction
from qvix.utils import (cvt_cfgPathToDict, evaluate, get_logger,
                        loggin_gpu_info, loggin_system_info, make_step, check_cfg)

parser = argparse.ArgumentParser(description="Train classification model.")
parser.add_argument("cfg", type=str, help="Path to configuration file.")


def main() -> None:
    args = parser.parse_args()
    cfg_path = args.cfg

    cfg = cvt_cfgPathToDict(cfg_path)
    check_cfg(cfg)

    os.makedirs(cfg['work_dir'], exist_ok=True)

    logger = get_logger(cfg['work_dir'])
    logger.info(f"Configuration file: {cfg_path}")
    logger.info(f"Configuration: {os.linesep + pformat(cfg)}")
    logger.info(f"JAX devices: {jax.devices()}")

    loggin_system_info(logger)
    loggin_gpu_info(logger)

    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Equinox version: {eqx.__version__}")
    logger.info(f"Optax version: {optax.__version__}")
    logger.info(f"Qvix version: {qvix.__version__}")

    key = jax.random.PRNGKey(cfg['seed'])

    model, model_state = build_backbone(cfg['model'], key)

    if cfg['initialization'] is not None:
        model = initialize_model(model, key)

    logger.info(f"Model: {cfg['model']['name']}")
    logger.info(f"Model Architecture: {model}")

    criterion = OptaxLossFunction(cfg['loss'])

    start_iteration = 1
    if cfg['resume_from'] is not None:
        resume_from = os.path.join(cfg['work_dir'], cfg['resume_from'])
        model = eqx.tree_deserialise_leaves(resume_from, model)
        stopped_iteration = int(resume_from.split("_")[-1].split(".")[0])
        logger.info(f"Resume from Iteration: {stopped_iteration}")
        start_iteration = stopped_iteration + 1
    elif cfg['load_from'] is not None:
        load_from = os.path.join(cfg['load_from'])
        model = eqx.tree_deserialise_leaves(load_from, model)
        logger.info(f"Load from {load_from}")

    trainloader = build_dataloader(cfg['train_loader'])
    testloader = build_dataloader(cfg['test_loader'])

    if 'optimizer' in cfg:
        optimizer = build_optax_object(cfg['optimizer'])
    elif 'optimizer_chain' in cfg:
        optimizer = build_optimizer_chain(cfg['optimizer_chain'])

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    start_time = time.time()
    logger.info(f"Start training at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def infinite_trainloader():
        while True:
            yield from trainloader

    for i, (x, y) in zip(range(start_iteration, cfg['iterations'] + 1),
                         infinite_trainloader()):

        x = x.numpy()
        y = y.numpy().astype(int)

        model, model_state, opt_state, train_loss, train_acc = make_step(
            model, model_state, opt_state, optimizer, key, criterion, x, y)

        if i % cfg['log_interval'] == 0:
            logger.info(
                f"Iteration: {i}, Loss: {train_loss:.4f}, Accuracy: {train_acc * 100:.2f}%"
            )

        if i % cfg['checkpoint_interval'] == 0:
            save_checkpoint = os.path.join(cfg['work_dir'], f"iteration_{i}.eqx")
            eqx.tree_serialise_leaves(save_checkpoint, model)

        if i % cfg['validate_interval'] == 0:
            model = eqx.nn.inference_mode(model)
            val_loss, val_acc = evaluate(model, testloader, key, criterion,
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
