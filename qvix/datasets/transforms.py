from qvix.registry import TransformRegistry
import numpy as np
import jax.numpy as jnp


@TransformRegistry.register()
class ToJNP(object):
    def __call__(self, 
                 image: np.ndarray,
                 div_val: jnp.float32 = 255.0,
                 dtype=jnp.float32) -> jnp.ndarray:
        """
        Args:
            image (np.ndarray, (H, W, C)) 

        Returns:
            image (jnp.ndarray, (C, H, W))
        """

        image = jnp.array(image, dtype=dtype)
        image = jnp.transpose(image, (2, 0, 1))
        div_val = jnp.array(div_val, dtype=dtype)    
        image = image / div_val
        
        return image
        


@TransformRegistry.register()
class Normalize(object):
    def __init__(self,
                mean: list[float], 
                std: list[float],
                dtype=np.float32) -> None:
        """
        Args:
            mean (list[float])
            std (list[float])
        """
        self.mean = np.array(mean, dtype=dtype)
        self.std = np.array(std, dtype=dtype)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image (np.ndarray, (C, H, W)) 

        Returns:
            image (np.ndarray, (C, H, W))
        """
        image = (image - self.mean) / self.std
        return image


