#!/bin/python
#-----------------------------------------------------------------------------
# File Name : train_dataset_model.py
# Author: Emre Neftci
#
# Creation Date : 
# Last Modified : Fri 19 Apr 2024 10:36:51 AM CEST
#
# Copyright : (c) Emre Neftci, PGI-15 Forschungszentrum Juelich
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import sys

import math, tqdm, time
import jax
import jax.lax as lax
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import optax  # https://github.com/deepmind/optax
import functools as ft
import equinox as eqx
from utils import get_method, prepare_data
import argparse 
import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as sharding

import wandb
import tqdm
import yaml 

parser = argparse.ArgumentParser(description='Train model (-m) on dataset (-d) with config (-c) and functions (-l)')
parser.add_argument('-c','--config', default='parameters/config_default.yaml', help='YAML config file')
parser.add_argument('-m','--model', help='model path', required=True)
parser.add_argument('-n','--nowb', help='disable wandb', action='store_true')
parser.add_argument('-i','--norun', help='disable run', action='store_true')
parser.add_argument('-s','--no_parallel', help='disable data parallelism', action='store_true')
parser.add_argument('-d','--dataset', default='', help='Dataset selected from datasets.py')
parser.add_argument('-l','--funcs', default='utils.create_cls_func_xent', help='path to functions used in training')
parser.add_argument('-t','--notesteval', help='disable run', action='store_true')
parser.add_argument('-u','--notraineval', help='disable run', action='store_true')
parser.add_argument('-e','--seed', help='seed', default=0)
parser.add_argument('-r','--train_iter_reset', help='reset training iterator', action='store_true')
args = vars(parser.parse_args())


config = yaml.safe_load(open(args['config']))
if args['nowb']:
    wandb.init(project=args['dataset'], config=config, mode='disabled')
else:
    wandb.init(project=args['dataset'], config=config) 
w_c = wandb.config

dt = w_c['dt']

## Datasets
create_dataloaders = get_method(args['dataset'])
dataloader_train, dataloader_test, dataloader_val, input_size, output_size = create_dataloaders(dt = dt, **w_c['dataset_kwargs'])
try:
    num_train_iters = len(dataloader_train)
except TypeError:
    num_train_iters = w_c['num_test_iters']

try:
    num_test_iters = len(dataloader_test)
except TypeError:
    num_test_iters = w_c['num_test_iters']

SEED = int(args['seed'])
key, model_key = jrandom.split(jrandom.PRNGKey(SEED), 2)

model, filter_spec = get_method(args['model'])(
        input_size = input_size,
        out_channels = output_size,
        key = model_key,        
        dt = dt,
        **w_c['model_kwargs']
        )

epochs = w_c['epochs']
learning_rate = w_c['optim_kwargs']['learning_rate']
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=learning_rate/5,
    peak_value=learning_rate,
    warmup_steps=10,
    decay_steps=int(epochs * num_train_iters),
    end_value=1e-4*learning_rate
)

optim = optax.chain(optax.clip_by_global_norm(w_c['optim_kwargs']['clip_norm']),
                    optax.adamw(lr_schedule, weight_decay=w_c['optim_kwargs']['weight_decay']))
opt_state = optim.init(filter_spec)

compute_loss, make_step, accuracy = get_method(args['funcs'])(model, optim, filter_spec, output_size)

#Create mesh/shard for parallel run
if args['no_parallel']:
    shard = None
else:
    num_devices = len(jax.devices())
    devices = mesh_utils.create_device_mesh((num_devices, 1))
    shard = sharding.PositionalSharding(devices)

if __name__ == "__main__":
    if args['norun']:
        xs, ys, *zs = next(iter(dataloader_train))
        sys.exit(0)

    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)

    for e in tqdm.tqdm(range(epochs)):
        ### Train 
        loss_sum = 0
        ms_key, key = jrandom.split(key,2)
        model = eqx.tree_inference(model, value=False)
        for i in tqdm.tqdm(range(0,num_train_iters),leave=False):
            batch = next(train_iter)
            datap, bkp, key = prepare_data(batch,key,shard)
            xsp, ysp, *zsp = datap
            loss, model, opt_state, grads, updates = make_step(model, datap, opt_state, bkp)
            loss_sum += loss.item()
        
        if not args['train_iter_reset']: train_iter = iter(dataloader_train)

        wandb.log({'train/loss':loss_sum, "epoch":e})
        wandb.log({'optim/count':opt_state[-1][0].count, "epoch":e})
        #wandb.log({'optim/lr':lr_schedule(opt_state[-1][0].count), "epoch":e})

        acc_ = []
        model = eqx.tree_inference(model, value=True)


        ### Train Dataset Evaluation
        if not args['notraineval']:
            for i in tqdm.tqdm(range(0,num_train_iters),leave=False):
                batch = next(train_iter)
                datap, bkp, key = prepare_data(batch,key,shard)
                xsp, ysp, *zsp = datap
                acc_.append(accuracy(model, datap, bkp).item())
            if not args['train_iter_reset']: train_iter = iter(dataloader_train)

            acc_train = np.array(acc_).mean()
            wandb.log({'acc/train':acc_train, "epoch":e})
            tqdm.tqdm.write('Accuracy Train: {0:.4f} | Loss: {1:.2f}'.format(acc_train,loss_sum))

        ### Test Dataset Evaluation
        if not args['notesteval']:
            test_iter = iter(dataloader_test)
            acc_ = []
            model = eqx.tree_inference(model, value=True)

            for i in tqdm.tqdm(range(0,num_test_iters),leave=False):
                batch = next(test_iter)
                datap, bkp, key = prepare_data(batch, key, shard)
                xsp, ysp, *zsp = datap
                acc_.append(accuracy(model, datap, bkp).item())
            test_iter = iter(dataloader_test)

            acc_test = np.array(acc_).mean()
            wandb.log({'acc/val':acc_test, "epoch":e})
            tqdm.tqdm.write('Accuracy Test : {0:.4f} | Loss: {1:.2f}'.format(acc_test,loss_sum))






