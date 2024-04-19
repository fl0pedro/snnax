from typing import Sequence, Union, Callable, Optional, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

from equinox import static_field

from .stateful import StatefulLayer
from ...functional.surrogate import superspike_surrogate
from chex import Array, PRNGKey

class SimpleLIF(StatefulLayer):
    """
    Simple implementation of a layer of leaky integrate-and-fire neurons 
    which does not make explicit use of synaptic currents.
    Requires one decay constant to simulate membrane potential leak.
    
    Arguments:
        - `decay_constant`: Decay constant of the simple LIF neuron.
        - `spike_fn`: Spike treshold function with custom surrogate gradient.
        - `threshold`: Spike threshold for membrane potential. Defaults to 1.
        - `reset_val`: Reset value after a spike has been emitted.
        - `stop_reset_grad`: Boolean to control if the gradient is propagated
                        through the refectory potential.
        - `init_fn`: Function to initialize the initial state of the 
                    spiking neurons. Defaults to initialization with zeros 
                    if nothing else is provided.
    """
    decay_constants: Union[Sequence[float], Array] = static_field()
    threshold: float = static_field()
    spike_fn: Callable = static_field()
    stop_reset_grad: bool = static_field()
    reset_val: Optional[float] = static_field()

    def __init__(self,
                decay_constant: float,
                spike_fn: Callable = superspike_surrogate(10.),
                threshold: float = 1.,
                stop_reset_grad: bool = True,
                reset_val: Optional[float] = None,
                init_fn: Optional[Callable] = None) -> None:

        super().__init__(init_fn)
        # TODO assert for numerical stability 0.999 leads to errors...
        self.threshold = threshold
        self.decay_constant = decay_constant
        self.spike_fn = spike_fn
        self.reset_val = reset_val if reset_val is not None else None
        self.stop_reset_grad = stop_reset_grad
            
    def __call__(self, 
                state: Array, 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> Sequence[Array]:
        alpha = self.decay_constant
        
        mem_pot = alpha*mem_pot + (1.-alpha)*synaptic_input # TODO with (1-alpha) or without ?
        spike_output = self.spike_fn(mem_pot-self.threshold)
        
        if self.reset_val is None:
            reset_pot = mem_pot*spike_output
        else:
            reset_val = jax.nn.softplus(self.reset_val)
            reset_pot = reset_val * spikes_out
        # optionally stop gradient propagation through refectory potential       
        refectory_potential = lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_potential
        
        mem_pot = alpha*mem_pot + (1.-alpha)*synaptic_input # TODO with (1-alpha) or without ?
        spikes_out = self.spike_fn(mem_pot-threshold)

        output = spikes_out
        state = [mem_pot,  spikes_out]
        return [state, output]

class LIF(StatefulLayer):
    """
    TODO improve docstring
    Implementation of a leaky integrate-and-fire neuron with
    synaptic currents. Requires two decay constants to describe
    decay of membrane potential and synaptic current.
    """
    decay_constants: Union[Sequence[float], Array] = static_field()
    threshold: float = static_field()
    spike_fn: Callable = static_field()
    reset_val: float = static_field()
    stop_reset_grad: bool = static_field()

    def __init__(self, 
                decay_constants: Union[Sequence[float], Array],
                spike_fn: Callable = superspike_surrogate(10.),
                threshold: float = 1.,
                stop_reset_grad: bool = True,
                reset_val: Optional[float] = None,
                init_fn: Optional[Callable] = None) -> None:
        """
        Arguments:
            - `shape`: Shape of the neuron layer.
            - `decay_constants`: Decay constants for the LIF neuron.
                - Index 0 describes the decay constant of the membrane potential,
                - Index 1 describes the decay constant of the synaptic current.
            - `spike_fn`: Spike treshold function with custom surrogate gradient.
            - `threshold`: Spike threshold for membrane potential. Defaults to 1.
            - `reset_val`: Reset value after a spike has been emitted. 
                            Defaults to None.
            - `stop_reset_grad`: Boolean to control if the gradient is propagated
                                through the refectory potential.
            - `init_fn`: Function to initialize the state of the spiking neurons.
                        Defaults to initialization with zeros if 
                        nothing else is provided.
        """

        super().__init__(init_fn)
        # TODO assert for numerical stability 0.999 leads to errors...
        self.decay_constants = decay_constants
        self.threshold = threshold
        self.spike_fn = spike_fn
        self.reset_val = reset_val
        self.stop_reset_grad = stop_reset_grad

    def init_state(self, 
                   shape: Union[int, Sequence[int]], 
                   key: PRNGKey, 
                   *args, 
                   **kwargs) -> Sequence[Array]:
        init_state_mem_pot = self.init_fn(shape, key, *args, **kwargs)
        init_state_syn_curr = jnp.zeros(shape) # The synaptic currents are initialized as zeros
        init_state_spike_output = jnp.zeros(shape) # The synaptic currents are initialized as zeros
        return [init_state_mem_pot, init_state_syn_curr, init_state_spike_output]


    def __call__(self, 
                state: Sequence[jnp.ndarray], 
                synaptic_input: jnp.ndarray,
                *, key: Optional[jax.random.PRNGKey] = None) -> Sequence[jnp.ndarray]:
        mem_pot, syn_curr, spike_output = state[0], state[1], state[2]
        
        if self.reset_val is None:
            reset_pot = mem_pot*spike_output 
        else:
            reset_pot = (mem_pot-self.reset_val)*spike_output 

        # optionally stop gradient propagation through refectory potential       
        refectory_potential = lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_potential

        alpha = self.decay_constants[0]
        beta = self.decay_constants[1]
        
        mem_pot = alpha*mem_pot + (1.-alpha)*syn_curr
        syn_curr = beta*syn_curr + (1-beta)*synaptic_input

        spike_output = self.spike_fn(mem_pot - self.threshold)

        state = [mem_pot, syn_curr, spike_output]
        return [state, spike_output]
    
class LIFSoftReset(LIF):
    """
    Similar to LIF but reset is additive (relative) rather than absolute:
    If the neurons spikes: 
    $V \rightarrow V_{reset}$
    where $V_{reset}$ is the parameter reset_val
    """
    def __call__(self, 
                state: Sequence[jnp.ndarray], 
                synaptic_input: jnp.ndarray,
                *, key: Optional[jax.random.PRNGKey] = None) -> Sequence[jnp.ndarray]:
        mem_pot, syn_curr, spike_output = state[0], state[1], state[2]
        
        if self.reset_val is None:
            reset_pot = spike_output 
        else:
            reset_pot = self.reset_val*spike_output

        # optionally stop gradient propagation through refectory potential       
        refr_pot = lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refr_pot

        alpha = self.decay_constants[0]
        beta = self.decay_constants[1]
        
        mem_pot = alpha*mem_pot + (1.-alpha)*syn_curr
        syn_curr = beta*syn_curr + (1-beta)*synaptic_input

        spike_output = self.spike_fn(mem_pot - self.threshold)

        state = [mem_pot, syn_curr, spike_output]
        return [state, spike_output]

class AdaptiveExponentialLIF(StatefulLayer):
    """
    Implementation of a adaptive exponential leaky integrate-and-fire neuron
    as presented in https://neuronaldynamics.epfl.ch/online/Ch6.S1.html.
    """
    decay_constants: float = static_field()
    threshold: float = static_field()
    spike_fn: Callable = static_field()
    ada_step_val: float = static_field()
    ada_decay_constant: float = static_field()
    ada_coupling_var: float = static_field()
    stop_reset_grad: bool = static_field()
    reset_val: Optional[float] = static_field()

    def __init__(self,
                decay_constant: float,
                ada_decay_constant: float,
                ada_step_val: float,
                ada_coupling_var: float,
                spike_fn: Callable = superspike_surrogate(10.),
                threshold: float = 1.,
                stop_reset_grad: bool = True,
                reset_val: Optional[float] = None,
                init_fn: Optional[Callable] = None) -> None:
        """**Arguments**:

        - `decay_constants`: Decay constants for the adaptive LIF neuron.
            - Index 0 describes the decay constant of the membrane potential,
            - Index 1 describes the decay constant of the synaptic current.
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
        self.threshold = threshold
        self.decay_constant = decay_constant
        self.spike_fn = spike_fn
        self.reset_val = reset_val if reset_val is not None else None
        self.stop_reset_grad = stop_reset_grad
        
        self.ada_decay_constant = ada_decay_constant
        self.ada_step_val = ada_step_val
        self.ada_coupling_var = ada_coupling_var

    def init_state(self, 
                    shape: Union[Sequence[int], int], 
                    key: PRNGKey, 
                    *args, 
                    **kwargs) -> Sequence[Array]:
        init_state_mem_pot = self.init_fn(shape, key, *args, **kwargs)
        init_state_ada = jnp.zeros(shape, key)
        init_state_spikes = jnpzeros(shape, key)
        init_state = jnp.concatenate([init_state_mem_pot, init_state_ada, init_state_spikes])
        return init_state

    def __call__(self, 
                state: Sequence[Array], 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None) -> Sequence[Array]:
        mem_pot, ada_var = state[0], state[1]

        alpha = self.decay_constant

        # Calculation of the membrane potential
        mem_pot = alpha*mem_pot + (synaptic_input+ada_var)
        # membrane_potential_new = alpha*membrane_potential + (1-alpha)*(synaptic_input + ada_var) # TODO with (1-alpha) or without ?
        spike_output = self.spike_fn(mem_pot - self.threshold)
        
        # Calculation of the adaptive part of the dynamics
        ada_var = (1.-self.ada_decay_constant)*self.ada_coupling_var \
                * (mem_pot-self.threshold) \
                + self.ada_decay_constant*ada_var \
                - self.ada_step_val*lax.stop_gradient(spike_output)

        if self.reset_val is None:
            reset_pot = mem_pot*spike_output
        else:
            reset_pot = self.reset_val * spike_output
            
        # optionally stop gradient propagation through refectory potential       
        refectory_pot = lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        mem_pot = mem_pot - refectory_pot

        # state = (membrane_potential_new, ada_var_new)
        state = jnp.concatenate([mem_pot, ada_var_new, spike_output])
        return [state, spike_output]

