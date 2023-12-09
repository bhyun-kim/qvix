import os.path as osp
import sys
from importlib import import_module


def cvt_moduleToDict(mod):
    """
    Args : 
        mod (module)  
    
    Returns :
        cfg (dict)
    
    """
    cfg = {
        name: value
        for name, value in mod.__dict__.items() if not name.startswith('__')
    }

    return cfg


def cvt_cfgPathToDict(path):
    """Convert configuration path to dictionary to
    Args: 
        path (str)

    Returns: 
        cfg (dict)
    """

    abs_path = osp.abspath(path)

    sys.path.append(osp.split(abs_path)[0])
    _mod = import_module(osp.split(abs_path)[1].replace('.py', ''))

    return cvt_moduleToDict(_mod)