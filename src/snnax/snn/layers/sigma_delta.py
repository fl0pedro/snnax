import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

from .stateful import StatefulLayer
from typing import Sequence, Optional, Callable, Union
from equinox import static_field
from ...functional.surrogate import superspike_surrogate
from chex import PRNGKey, Array


# TODO callable is missing
class SigmaDelta(StatefulLayer):
    """
    Implementation of a Sigma-Delta neuron. A Sigma-Delta neuron consists of a Sigma-Decoder
    part and a Delta-Encoder part. 

    Sigma-Decoder accumulates the synaptic inputs of the neuron over the timesteps (sigma-value). 
    
    Delta-Encoder calculates the current timestep activations using the accumulated spikes by the 
    Sigma-Decoder and generates the delta-value by taking the difference of current timestep 
    activations and previous timestep activations. Then previous timestep residue value is added 
    to generate the delta value which is the difference of delta value and neuron spike outputs 
    of previous timestep. This delta-value is used to generate the current timestep spike output 
    of the neuron. The resources are:

    [1]:
    @article{Nair2019AnUP,
    title={An Ultra-Low Power Sigma-Delta Neuron Circuit},
    author={Manu V. Nair and G. Indiveri},
    journal={2019 IEEE International Symposium on Circuits and Systems (ISCAS)},
    year={2019},
    pages={1-5},
    url={https://api.semanticscholar.org/CorpusID:67771396}

    [2]:
    https://github.com/lava-nc/lava/blob/main/src/lava/proc/sdn/process.py
}

    Arguments:
        `threshold` (float): desc
        `spike_fn` (Callable): desc
        `init_fn` (Callable): desc
    """
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

    def sigma_decoder(self, state: Sequence[Array], synaptic_input: Array):
        sigma = state[0]
        act_new = state[1]
        act = state[2]
        residue = state[3]
        s_out = state[4]

        sigma += synaptic_input
        act_new = sigma

        return [sigma, act_new, act, residue, s_out]
    
    def delta_encoder(self, state: Sequence[Array]):
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
  
    def __call__(self, state: Sequence[Array], 
                 synaptic_input: Array, 
                 *, key: Optional[PRNGKey] = None) -> Sequence[Array]:
        state = self.sigma_decoder(state, synaptic_input)
        return self.delta_encoder(state)

