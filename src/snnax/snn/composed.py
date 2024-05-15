from typing import Sequence, Tuple, Callable, Union, Optional
from chex import Array, PRNGKey

import jax
import equinox as eqx

from .layers.stateful import StatefulLayer, RequiresStateLayer
from .architecture import StatefulModel, GraphStructure, default_forward_fn


class Sequential(StatefulModel):
    """
    Convenience class to construct a feed-forward spiking neural network in a
    simple manner. It supports the defined StatefulLayer neuron types as well 
    as equinox layers. Under the hood it constructs a connectivity graph 
    with a feed-forward structure and feeds it to the StatefulModel class.
    """

    def __init__(self, 
                *layers: Sequence[eqx.Module],
                forward_fn: Callable = default_forward_fn) -> None:
        """**Arguments**:
        - `layers`: Sequence containing the layers of the network in causal order.
        """
        num_layers = len(list(layers))
        input_connectivity, input_layer_ids, final_layer_ids = gen_feed_forward_struct(num_layers)

        # Constructing the connectivity graph
        graph_structure = GraphStructure(
            num_layers = num_layers,
            input_layer_ids = input_layer_ids,
            final_layer_ids = final_layer_ids,
            input_connectivity = input_connectivity)

        super().__init__(graph_structure, list(layers), forward_fn = forward_fn)

    def __getitem__(self, idx: int) -> eqx.Module:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def __call__(self, state, data, key, **kwargs) -> Tuple[Sequence, Sequence]:
        return super().__call__(state, data, key, **kwargs)

class SequentialFeedback(StatefulModel):
    """
    Convenience class to construct a feed-forward spiking neural network with self recurrent connections in a
    simple manner. It supports the defined StatefulLayer neuron types as well 
    as equinox layers. Under the hood it constructs a connectivity graph 
    with a feed-forward structure and local recurrent connections and feeds it to the StatefulModel class.
    """

    def __init__(self, 
                *layers: Sequence[eqx.Module],
                forward_fn: Callable = default_forward_fn,
                feedback_layers = None,
                ) -> None:
        """**Arguments**:
        - `layers`: Sequence containing the layers of the network in causal order.
        """
        num_layers = len(list(layers))
        input_connectivity, input_layer_ids, final_layer_ids = gen_feed_forward_struct(num_layers)

        # Constructing the connectivity graph
        graph_structure = GraphStructure(
            num_layers = num_layers,
            input_layer_ids = input_layer_ids,
            final_layer_ids = final_layer_ids,
            input_connectivity = input_connectivity)

        if feedback_layers is None:
            for i, l in enumerate(layers):
                input_connectivity[i].append(i)
        else:
            for i, (k, l) in enumerate(feedback_layers.items()):
                input_connectivity[l].append(k)

        super().__init__(graph_structure, list(layers), forward_fn = forward_fn)

    def __getitem__(self, idx: int) -> eqx.Module:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def __call__(self, state, data, key, **kwargs) -> Tuple[Sequence, Sequence]:
        return super().__call__(state, data, key, **kwargs)


def gen_feed_forward_struct(num_layers: int) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """
    Function to construct a simple feed-forward connectivity graph from the
    given number of layers. This means that every layer is just connected to 
    the next one. 
    """
    input_connectivity = [[id] for id in range(-1, num_layers-1)]
    input_connectivity[0] = []
    input_layer_ids = [[] for id in range(0, num_layers)]
    input_layer_ids[0] = [0]
    final_layer_ids = [num_layers-1]
    return input_connectivity, input_layer_ids, final_layer_ids

class Parallel(eqx.Module):
    layers: Sequence[eqx.Module]

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs, key):
        h = [l(x) for l,x in zip(self.layers,inputs)]
        return sum(h)

class CompoundLayer(StatefulLayer):
    '''
    This is a convenience class that groups together several Equinox modules. This 
    is useful for convieniently addressing compound layers. For instance:
    eqx.Linear()
    eqx.LayerNorm()
    snn.LIF()
    '''

    layers: Sequence[eqx.Module]
    def __init__(self,
                 *layers: Sequence[eqx.Module], 
                 init_fn: Callable = None,
                  ) -> None:
        self.layers = layers
        super().__init__(init_fn = init_fn)
        

    def init_state(self,
                   shape: Union[Sequence[Tuple[int]], Tuple[int]],
                   key: PRNGKey) -> Sequence[Array]:
        states = []
        outs = []           
        keys = jax.random.split(key, len(self.layers))
        for ilayer, (key, layer) in enumerate(zip(keys, self.layers)):
            # Check if layer is a StatefulLayer
            if isinstance(layer, StatefulLayer):
                state = layer.init_state(shape = shape, key = key)
                out = layer.init_out(shape = shape, key = key)
                states.append(state)
                outs.append(out)
            # This allows the usage of modules from equinox
            # by calculating the output shape with a mock input
            elif isinstance(layer, RequiresStateLayer):
                mock_input = jax.numpy.zeros(shape)
                out = layer(mock_input)
                states.append([out])
                outs.append(out)
            elif isinstance(layer, Parallel):
                out = layer([jax.numpy.zeros(s) for s in shape], key=key)
                states.append([out])
                outs.append(out)
            elif isinstance(layer, eqx.Module):
                out = layer(jax.numpy.zeros(shape), key=key)
                states.append([out])
                outs.append(out)
            else:
                raise ValueError(f"Layer of type {type(layer)} not supported!")
            shape = out.shape

        return states

    def __call__(self, 
                state: Union[Array, Sequence[Array]], 
                synaptic_input: Array, *, 
                key: Optional[PRNGKey] = None):
        h = synaptic_input
        keys = jax.random.split(key, len(self.layers))
        new_states = []
        outs = []
        for ilayer, (key, old_state, layer) in enumerate(zip(keys, state, self.layers)):
            if isinstance(layer, StatefulLayer):
                new_state, h = layer(state=old_state, synaptic_input=h, key=key) 
                new_states.append(new_state)
                outs.append(h)
            elif isinstance(layer, RequiresStateLayer):
                h = layer(synaptic_input=h, state=old_state, key=key) 
                new_states.append([h])
                outs.append(h)
            else:
                h = layer(h, key=key) 
                new_states.append([h])
                outs.append(h)
        return new_states, outs







 


