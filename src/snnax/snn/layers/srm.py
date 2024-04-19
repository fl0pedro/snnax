from typing import Sequence, Union, Callable, Optional, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

from equinox import static_field
import equinox as eqx
from .stateful import StatefulLayer
from ...functional.surrogate import superspike_surrogate




class SRM(StatefulLayer):
    """
    TODO
    """
    layer: eqx.Module 
    decay_constants: Union[Sequence[float], jnp.ndarray] = static_field()
    threshold: Union[float, jnp.ndarray] = static_field()
    spike_fn: Callable = static_field()
    reset_val: Optional[Union[float, jnp.ndarray]] = static_field()
    stop_reset_grad: bool = static_field()

    def __init__(self, 
                layer: eqx.Module,
                decay_constants: Union[Sequence[float], jnp.ndarray],
                *args,
                spike_fn: Callable = superspike_surrogate(10.),
                threshold: Union[float, jnp.ndarray] = 1.,
                reset_val: Optional[Union[float, jnp.ndarray]] = None,
                stop_reset_grad: Optional[bool] = True,
                init_fn: Optional[Callable] = None,
                **kwargs) -> None:
        """**Arguments**:

        - `shape`: Shape of the neuron layer.
        - `decay_constants`: Decay constants for the leaky integrate-and-fire neuron.
            Index 0 describes the decay constant of the membrane potential,
            Index 1 describes the decay constant of the synaptic current.
        - `spike_fn`: Spike treshold function with custom surrogate gradient.
        - `threshold`: Spike threshold for membrane potential. Defaults to 1.
        - `reset_val`: Reset value after a spike has been emitted. Defaults to None.
        - `stop_reset_grad`: Boolean to control if the gradient is propagated
            through the refectory potential.
        - `init_fn`: Function to initialize the initial state of the spiking neurons.
            Defaults to initialization with zeros if nothing else is provided.
        """

        super().__init__(init_fn)
        # TODO assert for numerical stability 0.999 leads to errors...
        self.decay_constants = decay_constants
        self.threshold = threshold
        self.spike_fn = spike_fn
        self.reset_val = reset_val
        self.stop_reset_grad = stop_reset_grad
        self.layer = layer

    def init_state(self, 
                   shape: Union[Sequence[int], int], 
                   key: jrand.PRNGKey, 
                   *args, 
                   **kwargs) -> Sequence[jnp.ndarray]:
        init_state_P = jnp.zeros(shape)
        init_state_Q = jnp.zeros(shape) # The synaptic currents are initialized as zeros
        print(init_state_Q.shape)
        mock_input = self.layer(init_state_Q)
        
        init_state_S = jnp.zeros(mock_input.shape) # The synaptic currents are initialized as zeros
        return [init_state_P, init_state_Q, init_state_S]
    
    def init_out(self, 
                shape: Union[int, Sequence[int]], 
                key: Optional[jrand.PRNGKey] = None):
        # The initial ouput of the layer. Initialize as an array of zeros.
        return self.layer(jnp.zeros(shape))
   


    def __call__(self, 
                state: Sequence[jnp.ndarray], 
                synaptic_input: jnp.ndarray,
                *, key: Optional[jrand.PRNGKey] = None) -> Sequence[jnp.ndarray]:

        P, Q = state[0], state[1]

        alpha = self.decay_constants[0]
        beta = self.decay_constants[1]
        
        P = alpha*P + (1.-alpha)*synaptic_input
        Q = beta*Q + P
        membrane_potential = self.layer(Q)
        spike_output = self.spike_fn(membrane_potential - self.threshold)
        
        # if self.reset_val is None:
        #     reset_pot = membrane_potential*spike_output
        # else:
        #     reset_pot = self.reset_val*spike_output

        # optionally stop gradient propagation through refectory potential       
        # refectory_potential = lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        # membrane_potential = membrane_potential - refectory_potential

        state = [P, Q, spike_output]
        return state, spike_output


