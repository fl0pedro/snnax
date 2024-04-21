#!/bin/python
#-----------------------------------------------------------------------------
# File Name : model_snnmlp.py
# Author: Emre Neftci
#
# Creation Date : Tue 28 Mar 2023 12:44:33 PM CEST
# Last Modified : Fri 19 Apr 2024 12:01:55 PM CEST
#
# Copyright : (c) Emre Neftci, PGI-15 Forschungszentrum Juelich
# Licence : GPLv2
#----------------------------------------------------------------------------- 

from .common import *
import snnax.snn as snn
from snnax.functional.surrogate import superspike_surrogate as sr

# In[3]:
class SNNMLP(eqx.Module):
    cell: eqx.Module
    ro_int: int 
    burnin: int 

    def __init__(self, 
                 in_channels: int = 32*32*2,
                 hid_channels: int = 64,
                 out_channels:int = 11, 
                 dt : float = 0.001,
                 alpha = None,
                 beta = None,
                 tau_m: float = .02,
                 tau_s: float = .006,
                 size_factor: float = 4,
                 ro_int: int = -1,
                 burnin: int = 20,
                 norm: bool = False,
                 num_hid_layers: int = 1,
                 use_bias: bool = True,
                 neuron_model: str = 'snnax.snn.LIFSoftReset',
                 key = jrandom.PRNGKey(0),                 
                 **kwargs):
        
        ckey, lkey = jrandom.split(key)
        conn,inp,out = snn.composed.gen_feed_forward_struct(6)
        self.ro_int = ro_int
        self.burnin = burnin
        graph = snn.GraphStructure(6,inp,out,conn)

        if tau_m is None:
            tau_m = dt/np.log(alpha)
        else:
            assert alpha is None, "Only one of alpha or tau_m can be specified"
            alpha = np.exp(-dt/tau_m)

        if tau_s is None:
            tau_s = dt/np.log(beta)
        else:
            assert beta is None, "Only one of beta or tau_s can be specified"
            beta = np.exp(-dt/tau_s)

        
        self.cell = snn.Sequential(
            *make_layers(in_channels,
                         hid_channels,
                         out_channels,
                         key = key,
                         neuron_model = get_method(neuron_model),
                         alpha = alpha,
                         beta = beta,
                         size_factor=size_factor, 
                         use_bias = use_bias,
                         norm = norm, num_hid_layers=num_hid_layers),
            forward_fn = snn.architecture.default_forward_fn) # Remove debug_forward_fn for speed

    def __call__(self, x, key=None):
        state, out = self.get_final_states(x, key)
        return self.multiloss(out)
        
    def multiloss(self, out):
        if self.ro_int == -1:
            ro = out[-1].shape[0]
        else:
            ro = np.minimum(self.ro_int, out[-1].shape[0])
        return out[-1][::-ro]

    def embed(self, x, key):
        state = self.cell.init_state(x[0,:].shape, key)
        state, out = self.cell(state, x, key, burnin=self.burnin)
        return out[-1][-1]
    
    def get_final_states(self, x, key):
        state = self.cell.init_state(x[0,:].shape, key)

        states, out = self.cell(state, x, key, burnin=self.burnin)
        return states, out

def make_layers(in_channels, hid_channels, out_channels, key, neuron_model, size_factor=1, use_bias=True, num_hid_layers=2, alpha=0.95, beta=.85, norm=False):
    surr = sr(beta = 10.0)
    layers = [snn.Flatten()]
    for i in range(num_hid_layers):
        m = []
        init_key, key = jrandom.split(key,2)
        m.append(eqx.nn.Linear(in_channels, hid_channels*size_factor, key=init_key, use_bias=use_bias))
        if norm:
            m.append(eqx.nn.LayerNorm(shape=[hid_channels*size_factor],elementwise_affine=False, eps=1e-4))
        m.append(neuron_model([alpha,beta], spike_fn=surr, reset_val=1))
        layers += m
        in_channels = hid_channels*size_factor

    init_key, key = jrandom.split(key,2)
    layers.append(eqx.nn.Linear(hid_channels*size_factor, out_channels, key=key, use_bias=use_bias))
    return layers

def _model_init(model):
    ## Custom code ensures that only  conv layers are trained
    from snnax.snn.layers.stateful import StatefulLayer
    import jax.tree_util as jtu

    filter_spec = jtu.tree_map(lambda _: False, model)

    # trainable_layers = [i for i, layer in enumerate(model.cell.layers) if hasattr(layer, 'weight')]
    ## or  isinstance(layer, eqx.nn.LayerNorm)
    trainable_layers = [i for i, layer in enumerate(model.cell.layers) if isinstance(layer, eqx.nn.Linear)]

    for idx in trainable_layers:
        if model.cell.layers[idx].bias is not None:
            filter_spec = eqx.tree_at(
                lambda tree: (tree.cell.layers[idx].weight, tree.cell.layers[idx].bias),
                filter_spec,
                replace=(True,True),
            )
        else:
            filter_spec = eqx.tree_at(
                lambda tree: tree.cell.layers[idx].weight ,
                filter_spec,
                replace=True,
            )

    return model, filter_spec

def snn_mlp(in_channels=2*32*32, out_channels=10, key = jax.random.PRNGKey(0), **kwargs):
    return _model_init(SNNMLP(in_channels = in_channels, out_channels=out_channels, key = key, **kwargs))


if __name__ == "__main__":
    key = jrandom.PRNGKey(0)
    batch_key = jrandom.split(key, 36)
    model, filter_spec = snn_mlp(out_channels=10, norm=True, num_hid_layers=3, key = key)

    x = jnp.zeros((36,50,2*32*32)) #batch, time, channels, height, width
    out = jax.vmap(model)(x, batch_key)

