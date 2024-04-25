from typing import Sequence, Union, Callable, Optional

import jax
import jax.numpy as jnp

import equinox as eqx
from equinox import static_field
from chex import Array, PRNGKey

class StatefulLayer(eqx.Module):
    """
    Base class to define custom spiking neuron types.
    """
    init_fn: Callable = static_field()

    def __init__(self, init_fn: Callable = None):
        if init_fn is None:
            init_fn = lambda x, key, *args, **kwargs: jnp.zeros(x)
        self.init_fn = init_fn

    def init_state(self, 
                    shape: Union[int, Sequence[int]], 
                    key: PRNGKey, 
                    *args, 
                    **kwargs):
        return [self.init_fn(shape, key, *args, **kwargs), jnp.zeros(shape)]

    def init_out(self, 
                shape: Union[int, Sequence[int]], *, 
                key: Optional[PRNGKey] = None):
        # The initial ouput of the layer. Initialize as an array of zeros.
        return jnp.zeros(shape)

    def __call__(self, 
                state: Union[Array, Sequence[Array]], 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None):
        pass
        
class RequiresStateLayer(eqx.Module):
    """
    Base class to define custom spiking neuron types.
    """
    def __call__(self, state):
        raise NotImplementedError


