#!/bin/python
#-----------------------------------------------------------------------------
# File Name : utils.py
# Author: Emre Neftci
#
# Creation Date : Tue 14 Mar 2023 02:02:35 PM CET
# Last Modified : Tue 23 Apr 2024 09:27:00 PM CEST
#
# Copyright : (c) Emre Neftci, PGI-15 Forschungszentrum Juelich
# Licence : GPLv2
#----------------------------------------------------------------------------- 

import equinox as eqx
import numpy as np
from typing import *
import functools as ft
import jax
import jax.numpy as jnp
from typing import Callable,Type
from jaxtyping import PyTree
from chex import Array, PRNGKey

class TrainableArray(eqx.Module):
    data: Array
    requires_grad : bool

    def __init__(self, array):
        self.data = array
        self.requires_grad = True

def sum_pytrees(models):
    return jax.tree_map(lambda *models: sum(models), *models)



def prepare_data(data, key, shard=None):
    '''
    Assigns data to devices along the first dimension (batch dimension)
    If shard is not None, data parellel training will be enabled and the batch dimension will be divided among the shards
    '''
    datap = []
    for d in data:
        if hasattr(d, 'shape'):
            ds_ =  jnp.array(d)
            if shard is not None:
                datap.append(jax.device_put(ds_, shard.reshape(shard.shape[0],*np.ones_like(ds_.shape[1:]))))
            else:
                datap.append(ds_)
    jrandom = jax.random
    bk = jrandom.split(key, data[0].shape[0])
    _, key = jrandom.split(bk[-1], 2)
    if shard is not None:
        bkp = jax.device_put(bk, shard.reshape(shard.shape[0],*np.ones_like(bk.shape[1:])))
    else:
        bkp = bk
    return datap, bkp, key


def init_LSUV_actrate(act_rate, threshold=0., var=1.0):
    from scipy.stats import norm
    import scipy.optimize
    return scipy.optimize.fmin(lambda loc: (act_rate-(1-norm.cdf(threshold,loc,var)))**2, x0=0.)[0]

def apply_to_tree_leaf(pytree: PyTree, identifier: str, replace_fn: Callable) -> PyTree:
    """
    Apply a function all leaves in the given pytree with the given identifier.
    To simply replace values, use `replace_fn=lambda _: value`

    **Arguments**
    - `pytree`: The pytree where we want to modify the leaves.
    - `identifier`: A string used to identify the name/field of the leaves.
    - `replace_fn`: Callable which is applied to the leaf values.
    """
    _has_identifier = (
        lambda leaf: hasattr(leaf, identifier) and getattr(leaf, identifier) is not None
    )

    def _identifier(pytree):
        return tuple(
            getattr(leaf, identifier)
            for leaf in jax.tree_util.tree_leaves(pytree, is_leaf=_has_identifier)
            if _has_identifier(leaf)
        )

    return eqx.tree_at(_identifier, pytree, replace_fn=replace_fn)


def apply_to_tree_leaf_bytype(pytree: PyTree, 
                        typ: None, 
                        identifier: str, 
                        replace_fn: Callable) -> PyTree:
    """
    Apply a function all leaves in the given pytree with the given identifier.
    To simply replace values, use `replace_fn=lambda _: value`
    
    **Arguments**
    - 'typ': the type, can be none
    - `pytree`: The pytree where we want to modify the leaves.
    - `identifier`: A string used to identify the name/field of the leaves.
    - `replace_fn`: Callable which is applied to the leaf values.
    """
    if typ is None:
        _has_identifier = lambda leaf: hasattr(leaf, identifier)
    elif type(typ) is str:
        _has_identifier = lambda leaf: hasattr(leaf, typ) and hasattr(leaf, identifier)
    else:
        _has_identifier = lambda leaf: (type(leaf) == typ) and hasattr(leaf, identifier)
    def _identifier(pytree):
        a = tuple( 
                  getattr(leaf,identifier) 
                  for leaf in jax.tree_util.tree_leaves(pytree, is_leaf=_has_identifier) 
                  if _has_identifier(leaf) and getattr(leaf, identifier) is not None )
        return a

    return eqx.tree_at(_identifier, pytree, replace_fn=replace_fn)

def vid_to_patch(x, patch_size=4, flatten_channels=True, sort=False):
    """
    Inputs:
        x - torch.Tensor representing the video of shape [T, H, W, C]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    #x = x.swapaxes(0,1).swapaxes(2,4)
    T, C, H, W = x.shape
    x = x.reshape(T, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.transpose(0,2,4,3,5,1)
    x = x.reshape(T, -1, patch_size, patch_size, C)   # [H'*W', p_H, p_W, C]
    if flatten_channels:
        x = x.transpose([1,0,4,3,2])
        res = x.reshape(x.shape[0], x.shape[1], -1)
    else:
        res = x.transpose([1,0,4,3,2])
    if sort:
        idxs = np.argsort(res.sum(axis=np.arange(1,len(res.shape)))) 
        return res[idxs]
    else:
        return res

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    x = x.transpose(1, 2, 0) # [H, W, C] -> [C, H, W]
    H, W, C = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, f"Image dimensions must be divisible by patch size. Got {H}x{W} image and {patch_size} patch size."
    x = x.reshape(H//patch_size, patch_size, W//patch_size, patch_size, C)
    x = x.transpose(0, 2, 1, 3, 4)    # [H', W', p_H, p_W, C]
    x = x.reshape(-1, *x.shape[2:])   # [H'*W', p_H, p_W, C]
    if flatten_channels:
        x = x.reshape(x.shape[0], -1) # [H'*W', p_H*p_W*C]
    return x

def img_batch_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, H, W, C]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
#    B, H, W, C = x.shape
#    x = x.reshape(B, H//patch_size, patch_size, W//patch_size, patch_size, C)
#    x = x.transpose(0, 1, 3, 2, 4, 5)    # [B, H', W', p_H, p_W, C]
#    x = x.reshape(B, -1, *x.shape[3:])   # [B, H'*W', p_H, p_W, C]
#    if flatten_channels:
#        x = x.reshape(B, x.shape[1], -1) # [B, H'*W', p_H*p_W*C]
#    return x
    f = lambda x: img_to_patch(x, patch_size=patch_size, flatten_channels=flatten_channels)
    return jax.vmap(f, x)


def standardize(x: np.ndarray, eps=1e-7):
    mi = x.min(axis=0)
    ma = x.max(axis=0)
    return (x-mi)/(ma-mi+eps)

def show_filter_spec(model, filter_spec):
    from jax.tree_util import tree_map
    jtu.tree_map(lambda x, y: print(type(x),jnp.shape(y),x), filter_spec, vit);  

def sum_models(models):
    return jax.tree_map(lambda *models: sum(models), *models)

def xent(y_pred, y_hat, num_classes=10):
    """
    y_pred: jnp array of predictions
    y_hat: jnp array of targets (categorical)
    """
    y1h = jax.nn.one_hot(y_hat, num_classes=num_classes)
    return  -jnp.sum(y1h*jax.nn.log_softmax(y_pred)) / len(y_pred)

def mse(y_pred, y_hat):
    """
    y_pred: jnp array of predictions
    y_hat: jnp array of targets (dense)
    """
    return jnp.mean((y_hat-y_pred)**2)

def mse_cat(y_pred, y_hat, num_classes=10):
    """
    y_pred: jnp array of predictions
    y_hat: jnp array of targets (categorical)
    """
    y1h = jax.nn.one_hot(y_hat, num_classes=num_classes)
    return jnp.mean((y1h-y_pred)**2)

def prng_batch(key, batch_size):
    batch_key = jax.random.split(key, batch_size)
    _, key = jax.random.split(batch_key[-1], 2)
    return key, batch_key

def default_init(model, init_key = None, custom_init=True):
    if init_key is None:
        init_key = jax.random.PRNGKey(0)

    if custom_init:
        w_key, b_key = jax.random.split(init_key,2)
        kaiming_init_fn = lambda leaf: jax.nn.initializers.kaiming_normal()(w_key, leaf.shape, leaf.dtype)
        zeros_init_fn = lambda leaf: jax.nn.initializers.zeros(b_key, leaf.shape, leaf.dtype)
        ones_init_fn = lambda leaf: jax.nn.initializers.ones(b_key, leaf.shape, leaf.dtype)

        model = apply_to_tree_leaf_bytype(model, eqx.nn.GroupNorm, 'weight', ones_init_fn)
        model = apply_to_tree_leaf_bytype(model, eqx.nn.GroupNorm, 'bias', zeros_init_fn)   
        model = apply_to_tree_leaf_bytype(model, eqx.nn.LayerNorm, 'weight', ones_init_fn)
        model = apply_to_tree_leaf_bytype(model, eqx.nn.LayerNorm, 'bias', zeros_init_fn)   
        model = apply_to_tree_leaf_bytype(model, eqx.nn.Conv2d, 'weight', kaiming_init_fn)
        model = apply_to_tree_leaf_bytype(model, eqx.nn.Linear, 'weight', kaiming_init_fn)
        model = apply_to_tree_leaf_bytype(model, eqx.nn.Linear, 'weight', kaiming_init_fn)
        
    filter_spec = jax.tree_util.tree_map(lambda _: False, model)

    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.Conv2d, 'weight', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.Conv2d, 'bias', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.Linear, 'weight', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.Linear, 'bias', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.LayerNorm, 'weight', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.LayerNorm, 'bias', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.GroupNorm, 'weight', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, eqx.nn.GroupNorm, 'bias', lambda _: True)
    filter_spec = apply_to_tree_leaf_bytype(filter_spec, 'requires_grad', 'data', lambda _: True)
    
    return model, filter_spec


## Following for inspecting the gradients
#             for n,g in enumerate(grads.cell.layers):
#                 name = str(n)+'.'+str(g).split('(')[0]
#                 if hasattr(g,'weight'):
#                     if g.weight is not None:
#                         wandb.log({'hist/grad_weight/'+name: wandb.Histogram(g.weight.flatten()),'epoch':e})
#                 if hasattr(g,'bias'):
#                     if g.bias is not None:
#                         wandb.log({'hist/grad_bias/'+name: wandb.Histogram(g.bias.flatten()),'epoch':e})

#             for n,g in enumerate(updates.cell.layers):
#                 name = str(n)+'.'+str(g).split('(')[0]
#                 if hasattr(g,'weight'):
#                     if g.weight is not None:
#                         wandb.log({'hist/upd_weight/'+name: wandb.Histogram(g.weight.flatten()),'epoch':e})
#                 if hasattr(g,'bias'):
#                     if g.bias is not None:
#                         wandb.log({'hist/upd_bias/'+name: wandb.Histogram(g.bias.flatten()),'epoch':e})

#             batch_key = jax.random.split(key, len(x))
#             s,o= jax.vmap(model.get_final_states)(jnp.array(x), batch_key)

#             for n,g in enumerate(updates.cell.layers):
#                 name = str(n)+'.'+str(g).split('(')[0]
#                 wandb.log({'hist/state0/'+name: wandb.Histogram(s[n][0].flatten()),'epoch':e})

def lsuv_withnorm(model, 
        data: Sequence[Array], 
        key: PRNGKey, 
        tgt_mean: float=-.85, 
        tgt_var: float=1., 
        mean_tol: float=.1, 
        var_tol: float=.1, 
        max_iters: Optional[int] = None):
    """
    LSUV = Layer Sequential Unit Variance
    Initialization inspired from Mishkin D and Matas J. 
    All you need is a good init. arXiv:1511.06422 [cs], February 2016.
    This is a datadriven initialization scheme for possibly stateful 
    networks which adjusts the mean and variance of the membrane output to the 
    given values for each layer. The weights and biases are required to be 
    pytree leaves with attributes `weight` and `bias`.
    It operates on an entire minibatch.
    """
    # TODO maybe write JIT compiled version?
    init_key, key = jax.random.split(key)
    shape = data[0,0].shape
    init_states = model.init_state(shape, init_key)

    from snnax.snn.layers.stateful import StatefulLayer
    from jax.nn.initializers import zeros, ones
    w_key, b_key, key = jax.random.split(key, 3)
    ones_init_fn = lambda leaf: ones(b_key, leaf.shape, leaf.dtype)
    zero_init_fn = lambda leaf: zeros(b_key, leaf.shape, leaf.dtype)

    # Initialize all layers with `weight` or `bias` attribute with 
    # random orthogonal matrices or zeros respectively 
    model = apply_to_tree_leaf_bytype(model, eqx.nn.LayerNorm, 'weight', ones_init_fn)
    model = apply_to_tree_leaf_bytype(model, eqx.nn.LayerNorm, "bias", zero_init_fn)
    model = apply_to_tree_leaf_bytype(model, eqx.nn.GroupNorm, 'weight', ones_init_fn)
    model = apply_to_tree_leaf_bytype(model, eqx.nn.GroupNorm, "bias", zero_init_fn)

    adjust_var = lambda weight: weight *jnp.sqrt(tgt_var) / jnp.sqrt(jnp.maximum(mem_pot_var, 1e-2))
    adjust_mean_bias = lambda bias: bias -.2*(mem_pot_mean-tgt_mean) 
    # (1. - .2*(mem_pot_mean - tgt_mean)/norm) # TODO Further investigation!!!
    adjust_mean_weight = lambda eps, weight: weight*(1.-eps) 

    spike_layers = [i for i, layer in enumerate(model.layers) if isinstance(layer, StatefulLayer)]
    vmap_model = jax.vmap(model, in_axes=(None, 0, 0))
    alldone = False

    iters = 0
    while not alldone and iters<max_iters:
        alldone = True
        iters += 1

        keys = jax.random.split(key, data.shape[0])
        states, outs = vmap_model(init_states, data, keys)
        spike_layer_idx = 0
        for ilayer, layer in enumerate(model.layers):
            # Filter for layers that have weight attribute

            if isinstance(layer, StatefulLayer) :
                spike_layer_idx += 1
                continue
                
            if not(isinstance(layer, eqx.nn.LayerNorm) or isinstance(layer, eqx.nn.GroupNorm)):
                continue


            #handles corner case where lif is not the final layer
            if spike_layer_idx >= len(spike_layers): 
                continue
                
            idx = spike_layers[spike_layer_idx]
            
            # Sum of spikes over entire time horizon
            spike_sum = jnp.array(states[idx][-1]).mean(axis=0).sum() #last column is the spike output
            mem_pot_var = jnp.var(states[idx][0]) 
            mem_pot_mean = jnp.mean(states[idx][0])
            spike_mean = jnp.mean(spike_sum)
            
            
            assert spike_mean>=0
            
            print(f"Layer: {idx}, Variance of U: {mem_pot_var:.3}, \
                    Mean of U: {mem_pot_mean:.3}, \
                    Mean of S: {spike_mean:.3}")
            
            if jnp.isnan(mem_pot_var) or jnp.isnan(mem_pot_mean):
                done = False
                raise ValueError("NaN encountered during init!")
        
            if jnp.abs(mem_pot_var-tgt_var) > var_tol:
                model.layers[ilayer] = apply_to_tree_leaf(layer, 
                                                        "weight", 
                                                        adjust_var)
                done = False
            else:
                done = True
            alldone *= done
            
                
            if jnp.abs(mem_pot_mean-tgt_mean) > mean_tol:
                _has_bias = lambda leaf: hasattr(leaf, "bias")
                # Initialization with or without bias
                # TODO simplify this expression and refine case w/o bias
                bias_list = [getattr(leaf, "bias") is not None for leaf in jax.tree_util.tree_leaves(layer, is_leaf=_has_bias) if _has_bias(leaf)]
                if all(bias_list):
                    model.layers[ilayer] = apply_to_tree_leaf(layer, 
                                                            "bias", 
                                                            adjust_mean_bias)
                else:
                    eps = -.05*jnp.sign(mem_pot_mean-tgt_mean)/jnp.abs(mem_pot_mean-tgt_mean)**2
                    adjust_weight = ft.partial(adjust_mean_weight, eps)
                    model.layers[ilayer] = apply_to_tree_leaf(layer, 
                                                            "weight", 
                                                            adjust_weight)
                done = False
            else:
                done = True
            alldone *= done
        iters += 1
            
    return model

def get_method(name):
    from importlib import import_module

    try:
        p, m = name.rsplit('.', 1)
        mod = import_module(p)
        met = getattr(mod, m)
    except ImportError:
        print(name + ' not found, trying to replace model_ with model.')
        newname = name.replace('model_','model.')
        p, m = newname.rsplit('.', 1)
        mod = import_module(p)
        met = getattr(mod, m)
    return met


def create_cls_func_xent(model, optim, filter_spec, num_classes):
    '''
    Default classification functions using the cross entropy loss for classical neural networks (no time component)
    Y 
    ^ 
    X 
    '''
    import equinox as eqxac
    import jax
    
    @eqx.filter_value_and_grad
    def compute_loss(diff_model, static_model, batch, rng_key):
        x, y, *z = batch
        model = eqx.combine(diff_model, static_model)
        pred_y = jax.vmap(model)(x, key = rng_key)
        y1h = jax.nn.one_hot(y, num_classes=num_classes)
        return -jnp.mean(y1h*jax.nn.log_softmax(pred_y))

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        x, y, *z = batch
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def _run_accuracy(model, batch, rng_key):
        #Needed because counter is not jittable
        x, y, *z = batch
        return jax.vmap(model)(x, rng_key) 

    def accuracy(model, batch, rng_key):
        x, y, *z = batch
        pred_y = _run_accuracy(model, batch, rng_key).argmax(axis=-1)
        return np.mean(np.array(pred_y) == y)
    return compute_loss, make_step, accuracy

def create_cls_func_multixent(model, optim, filter_spec, num_classes, sparsity_loss = False):
    '''
    Sequence classification functions using the cross entropy loss for classical neural networks, loss applied every output
     Y  Y
     ^  ^
    X1 X2  
    '''
    import equinox as eqx
    import jax
    from collections import Counter, defaultdict 
    
    
    if sparsity_loss:
        @eqx.filter_value_and_grad
        def compute_loss(diff_model, static_model, batch, rng_key):
            x, y, *z = batch
            model = eqx.combine(diff_model, static_model)
            state, out = jax.vmap(model.get_final_states)(x, key = rng_key)
            pred_y = jax.vmap(model.multiloss)(out)
            y1h = jax.nn.one_hot(y, num_classes=num_classes)
            y11h = jnp.tile(y1h,(pred_y.shape[1],1,1)).swapaxes(0,1) 
            act = [o[-1].mean() for o in out[:-1]]
            vact = [jnp.mean((o[-2])**2) for o in out[:-1]]
            xent_loss = -jnp.mean(y11h*jax.nn.log_softmax(pred_y))
            return xent_loss +sum(act)+.2*sum(vact)

    else:
        @eqx.filter_value_and_grad
        def compute_loss(diff_model, static_model, batch, rng_key):
            x, y, *z = batch
            model = eqx.combine(diff_model, static_model)
            pred_y = jax.vmap(model)(x, key = rng_key)
            y1h = jax.nn.one_hot(y, num_classes=num_classes)
            y11h = jnp.tile(y1h,(pred_y.shape[1],1,1)).swapaxes(0,1) 
            xent_loss = -jnp.mean(y11h*jax.nn.log_softmax(pred_y)) 
            return xent_loss

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        x, y, *z = batch
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def _run_accuracy(model, batch, rng_key):
        x, y, *z = batch
        #Needed because counter is not jittable
        return jax.vmap(model)(x, rng_key) 

    def accuracy(model, batch, rng_key):
        x, y, *z = batch
        pred_y = _run_accuracy(model, batch, rng_key)
        maxs = pred_y.argmax(axis=-1) 
        pred = [] 
        for m in maxs: 
            pred.append(Counter(np.array(m)).most_common(1)[0][0])
        return np.mean(np.array(pred) == y)
    return compute_loss, make_step, accuracy

def create_func_seq2seq_xent(model, optim, filter_spec, num_classes):
    import equinox as eqx
    import jax
    from collections import Counter, defaultdict 
    
    @eqx.filter_value_and_grad
    def compute_loss(diff_model, static_model, batch, rng_key):
        x, y, *z = batch
        model = eqx.combine(diff_model, static_model)
        pred_y = jax.vmap(model)(x, key = rng_key)
        y1h = jax.nn.one_hot(y, num_classes=num_classes)
        xent_loss = -jnp.mean(y1h*jax.nn.log_softmax(pred_y)) 
        return xent_loss

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        x, y, *z = batch
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def _run_accuracy(model, batch, rng_key):
        x, y, *z = batch
        #Needed because counter is not jittable
        return jax.vmap(model)(x, rng_key) 

    def accuracy(model, batch, rng_key):
        x, y, *z = batch
        pred_y = _run_accuracy(model, batch, rng_key)
        maxs = pred_y.argmax(axis=-1) 
        
        return np.mean(maxs == y)
    return compute_loss, make_step, accuracy

def create_func_seq2seq_mse(model, optim, filter_spec, output_size):
    import equinox as eqx
    import jax
    from collections import Counter, defaultdict 
    
    @eqx.filter_value_and_grad
    def compute_loss(diff_model, static_model, batch, rng_key):
        x, y, *z = batch
        model = eqx.combine(diff_model, static_model)
        pred_y = jax.vmap(model)(x, key = rng_key)
        mse_loss = jnp.mean((y-pred_y)**2) 
        return mse_loss

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        x, y, *z = batch
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def _run_accuracy(model, batch, rng_key):
        x, y, *z = batch
        #Needed because counter is not jittable
        return jax.vmap(model)(x, rng_key) 

    def accuracy(model, batch, rng_key):
        x, y, *z = batch
        pred_y = _run_accuracy(model, batch, rng_key)
        errors = (pred_y-y)**2 
        
        return np.mean(errors)
    return compute_loss, make_step, accuracy

def create_func_seq2seq_mse_lastpointacc(model, optim, filter_spec, output_size):
    import equinox as eqx
    import jax
    from collections import Counter, defaultdict 
    
    @eqx.filter_value_and_grad
    def compute_loss(diff_model, static_model, batch, rng_key):
        x, y, *z = batch
        model = eqx.combine(diff_model, static_model)
        pred_y = jax.vmap(model)(x, key = rng_key)
        mse_loss = jnp.mean((y-pred_y)**2) 
        return mse_loss

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        x, y, *z = batch
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def _run_accuracy(model, batch, rng_key):
        x, y, *z = batch
        #Needed because counter is not jittable
        return jax.vmap(model)(x, rng_key) 

    def accuracy(model, batch, rng_key):
        x, y, *z = batch
        pred_y = _run_accuracy(model, batch, rng_key)
        errors = (pred_y[:,-1]-y[:,-1])**2 
        return np.mean(errors)
    return compute_loss, make_step, accuracy

def create_func_seq2seq_xent(model, optim, filter_spec, num_classes):
    import equinox as eqx
    import jax
    from collections import Counter, defaultdict 
    from optax import softmax_cross_entropy_with_integer_labels as xent

    def _loss_fn(model, x, labels, rng_key):
        logits = model(x, key=rng_key)
        return xent(logits, labels, num_classes=num_classes).sum(), logits
    
    
    @eqx.filter_value_and_grad
    def compute_loss(diff_model, static_model, batch, rng_key):
        x, labels, *z = batch
        model = eqx.combine(diff_model, static_model)
        losses, outs = jax.vmap(_loss_fn, in_axes=(None, 0, 0, 0))(model, x, labels, rng_key)
        return losses.mean()

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        x, y, *z = batch
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def accuracy(model, batch, rng_key, normalize=True):
        x, labels, *z = batch
        logits = jax.vmap(model)(x, key = rng_key)

        if logits.ndim == 3:
            assert labels.ndim == 2
            logits = logits.reshape((-1, logits.shape[-1]))
            labels = labels.reshape((-1,))
        reduce_fn = jnp.mean if normalize else jnp.sum
        return reduce_fn(jnp.argmax(logits, axis=-1) == labels) 

    return compute_loss, make_step, accuracy

def create_cls_func_xent_lastpoint(model, optim, filter_spec, num_classes):
    '''
    Sequence classification functions using the cross entropy loss for classical neural networks
        Y
     ^  ^
    X1 X2
    '''
    import equinox as eqxac
    import jax
    
    @eqx.filter_value_and_grad
    def compute_loss(diff_model, static_model, batch, rng_key):
        x, y, *z = batch
        model = eqx.combine(diff_model, static_model)
        pred_y = jax.vmap(model)(x, key = rng_key)
        y1h = jax.nn.one_hot(y, num_classes=num_classes)
        return -jnp.mean(y1h*jax.nn.log_softmax(pred_y[:,-1]))

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        x, y, *z = batch
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def _run_accuracy(model, batch, rng_key):
        #Needed because counter is not jittable
        x, y, *z = batch
        return jax.vmap(model)(x, rng_key)[:,-1] 

    def accuracy(model, batch, rng_key):
        x, y, *z = batch
        pred_y = _run_accuracy(model, batch, rng_key).argmax(axis=-1)
        return np.mean(np.array(pred_y) == y)
    return compute_loss, make_step, accuracy

def create_cls_func_xent_variable_seqlen(model, optim, filter_spec, num_classes):
    '''
    Sequence classification functions using the cross entropy loss for neural networks, where the sequence length is variable and the datset returns the sequence length in Z
               Y
     ^  ^      ^
    X1 X2 ... XZ

    assumes first element of z is the sequence length
    assumes model will return element at seqlen, by passing seqlen
    '''
    import equinox as eqxac
    import jax

    
    @eqx.filter_value_and_grad
    def compute_loss(diff_model, static_model, batch, rng_key):
        x, y, *z = batch
        model = eqx.combine(diff_model, static_model)
        pred_y = jax.vmap(model)(x, key = rng_key, seqlen=z[0])
        y1h = jax.nn.one_hot(y, num_classes=num_classes)
        return -jnp.mean(y1h*jax.nn.log_softmax(pred_y))

    @eqx.filter_jit
    def make_step(model, batch, opt_state, rng_key):    
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = compute_loss(diff_model, static_model, batch, rng_key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads, updates

    @eqx.filter_jit
    def _run_accuracy(model, batch, rng_key):
        #Needed because counter is not jittable
        x, y, *z = batch
        return jax.vmap(model)(x, key = rng_key, seqlen=z[0]) 

    def accuracy(model, batch, rng_key):
        pred_y = _run_accuracy(model, batch, rng_key).argmax(axis=-1)
        x, y, *z = batch
        return np.mean(np.array(pred_y) == y)
    return compute_loss, make_step, accuracy


