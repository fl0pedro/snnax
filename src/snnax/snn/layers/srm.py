from typing import Sequence, Union, Callable, Optional, Tuple
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

from equinox import static_field
import equinox as eqx
from .stateful import StatefulLayer, TrainableArray
from ...functional.surrogate import superspike_surrogate


class SRM(StatefulLayer):
    """
    TODO
    """
    layer: eqx.Module 
    decay_constants: Union[Sequence[float], jnp.ndarray, TrainableArray]     
    r_decay_constants: Union[Sequence[float], jnp.ndarray, TrainableArray]     
    threshold: Union[float, jnp.ndarray] = static_field()
    spike_fn: Callable = static_field()
    reset_val: Optional[Union[float, jnp.ndarray, TrainableArray]]
    stop_reset_grad: bool = static_field()

    def __init__(self, 
                layer: eqx.Module,
                decay_constants: Union[Sequence[float], jnp.ndarray, TrainableArray],
                r_decay_constants: Union[Sequence[float], jnp.ndarray, TrainableArray] = [.9],
                *args,
                spike_fn: Callable = superspike_surrogate(10.),
                threshold: Union[float, jnp.ndarray] = 1.,
                reset_val: Optional[Union[float, jnp.ndarray, TrainableArray]] = None,
                stop_reset_grad: Optional[bool] = True,
                init_fn: Optional[Callable] = None,
                input_shape: Union[Sequence[int],int,None] = None,
                shape: Union[Sequence[int],int,None] = None,
                key: jrand.PRNGKey = jax.random.PRNGKey(0),
                **kwargs) -> None:
        """**Arguments**:
        - `input_shape`: Shape of the neuron layer.
        - `shape`: Shape of the neuron layer.
        - `decay_constants`: Decay constants for the leaky integrate-and-fire neuron.
            Index 0 describes the decay constant of the membrane potential,
            Index 1 describes the decay constant of the synaptic current.
        - `r_decay_constants`: Decay constants for the refractory period.
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
        self.stop_reset_grad = stop_reset_grad
        self.layer = layer

        self.decay_constants = self.init_parameters(decay_constants, input_shape)
        self.r_decay_constants = self.init_parameters(r_decay_constants, shape)
        
        if reset_val is None:
            self.reset_val = self.init_parameters([0], shape, requires_grad=False)
        else:
            self.reset_val = self.init_parameters(reset_val, shape, requires_grad=True)

    def init_state(self, 
                   shape: Union[Sequence[int], int], 
                   key: jrand.PRNGKey, 
                   *args, 
                   **kwargs) -> Sequence[jnp.ndarray]:
        init_state_P = jnp.zeros(shape)
        init_state_Q = jnp.zeros(shape) # The synaptic currents are initialized as zeros
        output = self.layer(init_state_Q)
        
        init_state_R = jnp.zeros(output.shape) # The synaptic currents are initialized as zeros
        init_state_S = jnp.zeros(output.shape) # The synaptic currents are initialized as zeros
        return [init_state_P, init_state_Q, init_state_R, init_state_S]
    
    def init_out(self, 
                shape: Union[int, Sequence[int]], 
                key: Optional[jrand.PRNGKey] = None):
        # The initial ouput of the layer. Initialize as an array of zeros.
        return self.layer(jnp.zeros(shape))
   
    def __call__(self, 
                state: Sequence[jnp.ndarray], 
                synaptic_input: jnp.ndarray,
                *, key: Optional[jrand.PRNGKey] = None) -> Sequence[jnp.ndarray]:

        P, Q, R, S = state

        alpha = jax.lax.clamp(0.5, self.decay_constants.data[0], 1.0)
        beta = jax.lax.clamp(0.5, self.decay_constants.data[1], 1.0)
        gamma = jax.lax.clamp(0.5, self.r_decay_constants.data[0], 1.0)
        reset_val = jax.lax.clamp(0.0, self.reset_val.data[0], 2.0)

        P = alpha*P + (1.-alpha)*synaptic_input
        Q = beta*Q + (1.-beta)*P
        R = gamma*R + reset_val*lax.stop_gradient(S)
        membrane_potential = self.layer(Q)-reset_val*R
        spike_output = self.spike_fn(membrane_potential - self.threshold)

        state = [P, Q, R, spike_output]
        return state, spike_output

