import argparse
import logging
import os
import platform
import time
from pprint import pformat
from typing import Optional, Tuple

import equinox as eqx
import GPUtil
import jax
import jax.numpy as jnp
import optax
import psutil
from jaxtyping import Array, PRNGKeyArray
from torch.utils.data import DataLoader

import qvix
from qvix.builder import build_dataloader, build_object, calculate_step
from qvix.registry import BackboneRegistry, OptimizerRegistry
from qvix.utils import cvt_cfgPathToDict

parser = argparse.ArgumentParser(description="Train classification model.")
parser.add_argument("cfg", type=str, help="Path to configuration file.")


def get_logger(
        logger_dir: str,
        filename: Optional[str] = None,
        format: str = '%(asctime)s %(levelname)s %(message)s'
) -> logging.Logger:
    """Get logger.

    Args:
        logger_dir (str)
        filename (Optional[str], optional): Defaults to None. 
            If not specified, the log file name is the current time.
        format (str)
    """
    if filename is None:
        filename = f"{time.strftime('%Y-%m-%d %H%M%S')}.log"

    logging.basicConfig(level=logging.INFO,
                        format=format,
                        handlers=[
                            logging.FileHandler(logger_dir + '/' + filename),
                            logging.StreamHandler()
                        ])

    return logging.getLogger()


@eqx.filter_jit
def make_step(
    model: eqx.Module,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    key: PRNGKeyArray,
    loss_cfg: dict,
    x: Array,
    y: Array,
) -> Tuple[eqx.Module, optax.OptState, float]:
    keys = jax.random.split(key, num=x.shape[0])
    keys = jax.numpy.array(keys)

    (loss, acc), grads = eqx.filter_value_and_grad(
        calculate_step, has_aux=True
        )(model, loss_cfg, keys, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, acc


@eqx.filter_jit
def compute_accuracy(model: eqx.Module,
                     key: PRNGKeyArray,
                     x: Array,
                     y: Array,
                     inference: bool = True) -> float:
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    pred_y = model(x, key, inference=inference)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)


def evaluate(
    model: eqx.Module,
    testloader: DataLoader,
    key: PRNGKeyArray,
    loss_cfg: dict,
) -> Tuple[float, float]:
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """

    avg_loss = 0
    avg_acc = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy().astype(int)

        keys = jax.random.split(key, num=x.shape[0])
        keys = jax.numpy.array(keys)

        loss, acc = calculate_step(model, loss_cfg, keys, x, y, inference=True)

        avg_loss += loss
        avg_acc += acc

    return avg_loss / len(testloader), avg_acc / len(testloader)


def loggin_system_info(logger: logging.Logger) -> None:
    """
    This function retrieves system information including IP address, CPU, RAM, device name, and operating system details.
    
    Args: 
        logger (logging.Logger): Logger object
    """
    try:
        cpu_info = platform.processor()
        ram_info = psutil.virtual_memory()
        os_info = platform.platform()

        logger.info(f"CPU: {cpu_info}")
        logger.info(f"RAM: {ram_info.total / (1024 ** 3):.2f} GB")
        logger.info(f"OS: {os_info}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


def loggin_gpu_info(logger: logging.Logger) -> None:
    """
    This function retrieves GPU information including
         GPU id, name, load, free memory, used memory, total memory, temperature, and uuid.

    Args:
        logger (logging.Logger): Logger object

    References: 
        [1] https://thepythoncode.com/article/get-hardware-system-information-python#GPU_info
    """

    try:
        gpus = GPUtil.getGPUs()
    except Exception as e:
        gpus = []
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("No GPU found. Please check your GPU driver.",
                     exc_info=True)

    for gpu in gpus:

        gpu_id = gpu.id
        gpu_name = gpu.name
        gpu_total_memory = f"{gpu.memoryTotal}MB"
        gpu_uuid = gpu.uuid

        logger.info(f"GPU ID: {gpu_id}")
        logger.info(f"GPU Name: {gpu_name}")
        logger.info(f"GPU Total Memory: {gpu_total_memory}")
        logger.info(f"GPU UUID: {gpu_uuid}")


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
            # logg validation info, accuracy in percentage
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
