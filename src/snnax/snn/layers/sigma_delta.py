import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import equinox as eqx

from .stateful import StatefulLayer
from typing import Sequence, Optional, Callable, Union
from equinox import static_field
from ...functional.surrogate import superspike_surrogate
from chex import PRNGKey, Array


# TODO callable is missing
class SigmaDelta(StatefulLayer):

    threshold: float = static_field()
    spike_fn: Callable = static_field()
    
    def __init__(self, threshold: float = 1., 
                 spike_fn: Callable = superspike_surrogate(10.),
                 init_fn: Optional[Callable]=None):
        super().__init__(init_fn)
        self.threshold = threshold
        self.spike_fn = spike_fn

    def init_state(self, 
                   shape: Union[int, Sequence[int]], 
                   key: PRNGKey, 
                   *args, 
                   **kwargs) -> Sequence[Array]:
        sigma = jnp.zeros(shape, dtype=jnp.float32)
        act_new = jnp.zeros(shape, dtype=jnp.float32)
        act = jnp.zeros(shape, dtype=jnp.float32)
        residue = jnp.zeros(shape, dtype=jnp.float32)
        s_out = jnp.zeros(shape, dtype=jnp.float32)
        return [sigma, act_new, act, residue, s_out]

    def sigma_decoder(self, state: Sequence[jnp.ndarray], synaptic_input: jnp.ndarray):
        sigma = state[0]
        act_new = state[1]
        act = state[2]
        residue = state[3]
        s_out = state[4]

        sigma += synaptic_input
        act_new = sigma

        return [sigma, act_new, act, residue, s_out]
    
    def delta_encoder(self, state: Sequence[jnp.ndarray]):
        sigma = state[0]
        act_new = state[1]
        act = state[2]
        residue = state[3]
        s_out = state[4]

        delta = act_new - act + residue
        s_out = self.spike_fn(delta - self.threshold)
        residue = delta - s_out
        act = act_new
        return [sigma, act_new, act, residue, s_out], s_out
  
     

    def __call__(self, state: Sequence[jnp.ndarray], 
                 synaptic_input: jnp.ndarray, 
                 *, key: Optional[jrand.PRNGKey] = None) -> Sequence[jnp.ndarray]:
        state = self.sigma_decoder(state, synaptic_input)
        return self.delta_encoder(state)
        

