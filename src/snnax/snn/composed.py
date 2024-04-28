from typing import Sequence, Tuple, Callable

import jax
import equinox as eqx

from .layers.stateful import StatefulLayer
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

class SequentialLocalFeedback(StatefulModel):
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

#def gen_feed_forward_local_feedback_struct(num_layers: int) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
#    """
#    Function to construct a simple feed-forward connectivity with local recurrent connections graph from the     given number of layers. This means that every layer is just connected the next one and to itself.
#    """
#    input_connectivity = [[i] if i>=0 else [] for i in range(-1, num_layers-1)]
#    for i, l in enumerate(input_connectivity):
#        l.append(i)
#    input_layer_ids = [[] for id in range(0, num_layers)]
#    input_layer_ids[0] = [0]
#    final_layer_ids = [num_layers-1]
#    return input_connectivity, input_layer_ids, final_layer_ids
