from copy import deepcopy
from typing import Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
from jaxtyping import Array, PRNGKeyArray

from qvix.models.backbones.base_backbone import BaseBackbone, call_layer
from qvix.registry import BackboneRegistry


class BasicBlock(eqx.Module):
    layers: list
    shortcut: list
    act: jnn.relu
    expansion: int = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 key: PRNGKeyArray,
                 stride: int = 1):
        super().__init__()

        keys = jax.random.split(key, 2)

        self.layers = nn.Sequential([
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      use_bias=False,
                      key=keys[0]),
            nn.BatchNorm(out_channels, axis_name='batch'),
            nn.Lambda(jnn.relu),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      use_bias=False,
                      key=keys[1]),
            nn.BatchNorm(out_channels, axis_name='batch')
        ])

        self.shortcut = nn.Sequential([nn.Identity()])

        if stride != 1 or in_channels != self.expansion * out_channels:
            keys = jax.random.split(key, 2)
            self.shortcut = nn.Sequential([
                nn.Conv2d(in_channels,
                          self.expansion * out_channels,
                          kernel_size=1,
                          stride=stride,
                          use_bias=False,
                          key=keys[0]),
                nn.BatchNorm(self.expansion * out_channels, axis_name='batch')
            ])

        self.act = jnn.relu

    def __call__(self, 
                 x: Array, 
                 state: nn.State,
                key: Optional[PRNGKeyArray] = None) -> Array:
        """Forward pass.

        Args:
            x (Array): Input tensor.
        """

        out, state = self.layers(x, state=state)
        residual, state = self.shortcut(x, state=state)
        out += residual
        out = self.act(out)

        return out, state


class Bottleneck(eqx.Module):
    layers: nn.Sequential
    shortcut: nn.Sequential
    act: jnn.relu
    expansion: int = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 key: PRNGKeyArray,
                 stride: int = 1):
        super().__init__()

        keys = jax.random.split(key, 3)

        self.layers = nn.Sequential([
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=1,
                      stride=1,
                      use_bias=False,
                      key=keys[0]),
            nn.BatchNorm(out_channels, 'batch'),
            nn.Lambda(jnn.relu),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      use_bias=False,
                      key=keys[1]),
            nn.BatchNorm(out_channels, 'batch'),
            nn.Lambda(jnn.relu),
            nn.Conv2d(out_channels,
                      self.expansion * out_channels,
                      kernel_size=1,
                      stride=1,
                      use_bias=False,
                      key=keys[2]),
            nn.BatchNorm(self.expansion * out_channels, 'batch')
        ])

        self.shortcut = nn.Sequential([nn.Identity()])

        if stride != 1 or in_channels != self.expansion * out_channels:
            keys = jax.random.split(key, 2)
            self.shortcut = nn.Sequential([
                nn.Conv2d(in_channels,
                          self.expansion * out_channels,
                          kernel_size=1,
                          stride=stride,
                          use_bias=False,
                          key=keys[0]),
                nn.BatchNorm(self.expansion * out_channels, 'batch')
            ])

        self.act = jnn.relu

    def __call__(self, x: Array, state:nn.State,
                key: Optional[PRNGKeyArray] = None) -> Array:
        """Forward pass.

        Args:
            x (Array): Input tensor.
        """

        out, state = self.layers(x, state=state)
        residual, state = self.shortcut(x, state=state)
        out += residual
        out = self.act(out)

        return out, state


class ResNet(eqx.Module):
    in_channels: int
    stem: nn.Sequential
    layer1: nn.Sequential
    layer2: nn.Sequential
    layer3: nn.Sequential
    layer4: nn.Sequential
    avg_pool: nn.AdaptiveAvgPool2d
    linear: nn.Linear

    def __init__(self,
                 block: BasicBlock or Bottleneck,
                 num_blocks: int,
                 key: PRNGKeyArray,
                 num_classes: int = 10):
        super().__init__()
        self.in_channels = 64

        keys = jax.random.split(key, 6)

        self.stem = nn.Sequential([
            nn.Conv2d(3,
                      64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      use_bias=False,
                      key=keys[0]),
            nn.BatchNorm(64, axis_name='batch'),
            nn.Lambda(jnn.relu),
        ])

        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, keys[1])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, keys[2])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, keys[3])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, keys[4])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512 * block.expansion,
                                num_classes,
                                key=keys[5])

    def _make_layer(self, block: BasicBlock or Bottleneck, channels: int,
                    num_blocks: int, stride: int, key: PRNGKeyArray):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, key, stride=stride))
            self.in_channels = channels * block.expansion
        return layers

    def __call__(self,
                x: Array,
                state: nn.State = None,
                key: Optional[PRNGKeyArray] = None) -> Array:
        """Forward pass.

        Args:
            x (Array): Input tensor.
            state (nn.State): 
            key (Optional[PRNGKeyArray]): ignored; only for compatibility with other backbones.
        """
        out, state = self.stem(x, state=state)
        

        for layer in self.layer1:
            out, state = layer(out, state=state)
        for layer in self.layer2:
            out, state = layer(out, state=state)
        for layer in self.layer3:
            out, state = layer(out, state=state)
        for layer in self.layer4:
            out, state = layer(out, state=state)

        out = self.avg_pool(out)

        out = out.reshape(-1)
        out = self.linear(out)
        return out, state


@BackboneRegistry.register()
class ResNet18(ResNet):
    """ResNet18"""

    def __init__(self, key: PRNGKeyArray, num_classes: int = -1):
        super().__init__(BasicBlock, [2, 2, 2, 2], key, num_classes)


@BackboneRegistry.register()
class ResNet34(ResNet):
    """ResNet34"""

    def __init__(self, key: PRNGKeyArray, num_classes: int = -1):
        super().__init__(BasicBlock, [3, 4, 6, 3], key, num_classes)


@BackboneRegistry.register()
class ResNet50(ResNet):
    """ResNet50"""

    def __init__(self, key: PRNGKeyArray, num_classes: int = -1):
        super().__init__(Bottleneck, [3, 4, 6, 3], key, num_classes)


@BackboneRegistry.register()
class ResNet101(ResNet):
    """ResNet101"""

    def __init__(self, key: PRNGKeyArray, num_classes: int = -1):
        super().__init__(Bottleneck, [3, 4, 23, 3], key, num_classes)


@BackboneRegistry.register()
class ResNet152(ResNet):
    """ResNet152"""

    def __init__(self, key: PRNGKeyArray, num_classes: int = -1):
        super().__init__(Bottleneck, [3, 8, 36, 3], key, num_classes)
