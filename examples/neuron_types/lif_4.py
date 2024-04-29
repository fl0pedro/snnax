# %% [markdown]
# # SNNAX Framework Demonstration on DVS Gestures Dataset
# This notebook demonstrates how SNNAX Spiking Neural Network (SNN) Framework (https://iffgit.fz-juelich.de/pgi-15/snnax/) can be used to train an SNN on an event-based dataset (DVS Gestures). A convolutional neural network with 3 convolutional layers and one fully connected layer is used as the network model. Each of these layers are appended with a stateful LIF layer to obtain the spike outputs from the membrane potentials and synaptic currents of the neurons. Tonic package (https://tonic.readthedocs.io/en/latest/#) is used to manage the DVS Gestures dataset. 

# %%
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'   # So that XLA does not allocate 90% of the GPU memory at once.
os.environ['CUDA_VISIBLE_DEVICES'] = '2'    # Select the GPU. Use 'nvidia-smi' terminal command.

# %%
import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.nn as nn
from jax.tree_util import tree_map

import optax
import equinox as eqx

import snnax as snx
import snnax.snn as snn
from snnax.functional import one_hot_cross_entropy


import tonic
from tonic.transforms import Compose, Downsample, ToFrame
from tonic import SlicedDataset
from tonic.slicers import SliceByTime 

from utils import calc_accuracy, DVSGestures, RandomSlice
from torch.utils.data import DataLoader

#import wandb
from tqdm import tqdm
import numpy as np

# %%
MODEL_NAME = 'test_model'
LEARNING_RATE = 2e-3
N_EPOCHS = 300
BATCH_SIZE = 16
T = 1000 # Number of timesteps
SENSOR_HEIGHT = 32
SENSOR_WIDTH = 32
SENSOR_DIM = (2, SENSOR_WIDTH, SENSOR_HEIGHT)
N_CLASSES = 11  # Number of output classes
LABELS = ["hand clap",
        "right hand wave",
        "left hand wave",
        "right arm clockwise",
        "right arm counterclockwise",
        "left arm clockwise",
        "left arm counterclockwise",
        "arm roll",
        "air drums",
        "air guitar",
        "other gestures"]


TEST_T = 1798   # Duration of the shortest sample.
SCALING = .25
slicing_time_window = 2000000   # Length of each slice of a training example, i.t.o. microseconds.
T_SCALE = 10000000/slicing_time_window     # Time scale in ms.

# %%
SEED = 42
key = jrand.PRNGKey(SEED)
init_key, key = jrand.split(key, 2)

# %% [markdown]
# ### Dataset
# DVS Gestures dataset is used in this example notebook. This is an event-based dataset captured by an event-based camera and contains 10 gesture classes performed by hand, and an extra class for the gestures that do not fit to the 10 labeled gestures. Further detail on this dataset can be found on: https://research.ibm.com/interactive/dvsgesture/.
# Each 10 second training example is sliced into 2 second chunks, to make the dataset larger. Since the gestures contained in the DVS Gestures dataset are repetitive, slicing them does not harm the temporal dependency of the motions to be learned by the network. There are 1000 time bins for each training example, each time bin containing events accumulated in 2 milliseconds.

# %%
train_transforms = Compose([
        Downsample(time_factor=1., spatial_factor=SCALING), 
        ToFrame(sensor_size=(SENSOR_HEIGHT, SENSOR_WIDTH, 2), n_time_bins=T)])

test_transforms = Compose([
        Downsample(time_factor=1., spatial_factor=SCALING), 
        ToFrame(sensor_size=(SENSOR_HEIGHT, SENSOR_WIDTH, 2), n_time_bins=T)])


dataset_train = tonic.datasets.DVSGesture(save_to='./dvs_dataset_train', train=True)
dataset_test = tonic.datasets.DVSGesture(save_to='./dvs_dataset_test', train=False)


slicer = SliceByTime(time_window = slicing_time_window)
dataset_train = SlicedDataset(dataset_train, slicer=slicer, transform=train_transforms, metadata_path='./metadata/dvs_train')
dataset_test = SlicedDataset(dataset_test, slicer=slicer, transform=test_transforms, metadata_path='./metadata/dvs_test')


train_dataloader = DataLoader(dataset_train, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)
test_dataloader = DataLoader(dataset_test, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)

# %%
key1, key2, key3, key4, key5, key = jrand.split(key, 6)

# %%
# Parameters stored for wandb
#print("Saving model as", MODEL_NAME)
#wandb.init(project="jax-gestures")
#wandb.run.name = MODEL_NAME
#wandb.config = {
#    "epochs": N_EPOCHS,
#    "batchsize": BATCH_SIZE,
#    "learning_rate": LEARNING_RATE,
#}

# %%
def linear_layer_input_size(input_w_h: list[int, int], last_conv_layer_channels: int, \
        conv_layer_kernel_sizes: list[int, ...], \
        conv_layer_strides: list[int, ...], conv_layer_paddings: list[int, ...], ) -> int:
    hzt_dim = input_w_h[0]
    vert_dim = input_w_h[1]

    for i in range(len(conv_layer_kernel_sizes)):
        hzt_dim = (hzt_dim + 2 * conv_layer_paddings[i] - conv_layer_kernel_sizes[i]) \
                // conv_layer_strides[i] + 1

        vert_dim = (vert_dim + 2 * conv_layer_paddings[i] - conv_layer_kernel_sizes[i]) \
                // conv_layer_strides[i] + 1
    return hzt_dim * vert_dim * last_conv_layer_channels


lin_layer_ip_size = linear_layer_input_size([32, 32], 32, [7, 7, 7], [2, 1, 1], [0, 0, 0])

# %% [markdown]
# ### Defining The Network
# Spiking Neural Network layers are stateful layers. They contain membrane potential (U), synaptic current (I) and neuron spike output (S) variables. Fully connected and convolutional layers are used to obtain the synaptic current on the present timestep by accepting the previous neuron spike outputs as inputs. Thus, we can define the SNN as a stack of fully connected/convolutional and stateful layers, as can be seen below. Leaky integrate and fire neuron model is used in this case.

# %%
model = snn.Sequential(
    # 2x32x32
    eqx.nn.Conv2d(2, 32, 7, 2, key=key1, use_bias=False),
    snn.BatchNormLayer(eps = 1e-10, forget_weight=0.2, gamma=0.7),
    snn.SigmaDelta(),
    eqx.nn.Dropout(p=0.2),


    # 32x13x13
    eqx.nn.Conv2d(32, 32, 7 ,1, key=key4, use_bias=False),
    snn.BatchNormLayer(eps = 1e-10, forget_weight=0.2, gamma=0.7),
    snn.SigmaDelta(),
    eqx.nn.Dropout(p=0.2),

    # 32x7x7
    eqx.nn.Conv2d(32, 32, 7 ,1, key=key4, use_bias=False),
    snn.BatchNormLayer(eps = 1e-10, forget_weight=0.2, gamma=0.7),
    snn.SigmaDelta(),
    eqx.nn.Dropout(p=0.2),

    # 32x1x1
    snn.Flatten(),
    eqx.nn.Linear(lin_layer_ip_size, 11, key=key5, use_bias=False),
    snn.BatchNormLayer(eps = 1e-10, forget_weight=0.2, gamma=0.7),
    snn.SigmaDelta()
)
# %%
def calc_loss(model, init_state, data, target, loss_fn, key):
    """
    Here we define how the loss is exacly calculated, i.e. whether
    we use a sum of spikes or spike-timing for the calculation of
    the cross-entropy.
    """
    # data shape: (num_timesteps, num_channels, image_height, image_width)
    carry, outs = model(init_state, data, key=key)
    # output of last layer
    final_layer_out = outs[-1]
    # sum over all spikes
    pred = tree_map(lambda x: jnp.sum(x, axis=0), final_layer_out)

    loss = loss_fn(pred, target)
    return loss

# %%
# Vectorization of the loss function calculation
vmap_calc_loss = jax.vmap(calc_loss, in_axes=(None, None, 0, 0, None, 0), axis_name='batch_axis')

# %%
def calc_batch_loss(model, init_state, input_batch, target, loss_fn, key):
    """
    The vectorized version of calc_loss is used to 
    to get the batch loss.
    """
    # input_batch shape: (batch_size, num_timesteps, num_channels, image_height, image_width)
    keys = jrand.split(key, input_batch.shape[0])
    # Vector map loss calculation on batch_size dimension:
    loss_batch = vmap_calc_loss(model, 
                                init_state, 
                                input_batch, 
                                target, 
                                loss_fn, 
                                keys)
    loss = loss_batch.sum()
    return loss

# %%
def calc_loss_and_grads(model, init_state, input_batch, target, loss_fn, key):
    """
    This function uses the filter_value_and_grad feature of equinox to 
    calculate the value of the batch loss as well as it's gradients w.r.t. 
    the models parameters.
    """
    loss, grad = eqx.filter_value_and_grad(calc_batch_loss)(model, # HAS AUX ?????????????????
                                                            init_state, 
                                                            input_batch, 
                                                            target, 
                                                            loss_fn, 
                                                            key)
    return loss, grad

# %%
def update(model,
            optim, 
            opt_state, 
            input_batch, 
            target_batch, 
            loss_fn, 
            key):
    """
    Function to calculate the update of the model and the optimizer based
    on the calculated updates.
    """
    init_key, grad_key = jax.random.split(key)
    states = model.init_state(SENSOR_DIM, init_key)
    loss_value, grads = calc_loss_and_grads(model, 
                                            states, 
                                            input_batch, 
                                            target_batch, 
                                            loss_fn, 
                                            grad_key)    

    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value

# %%
init_batch = jnp.asarray(next(iter(train_dataloader))[0], dtype=jnp.float32)
#model = snn.init.lsuv(model, init_batch, init_key, max_iters=50)
    
optim = optax.adam(LEARNING_RATE)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

nbar = tqdm(range(N_EPOCHS))
for epoch in nbar:
    losses, test_accuracies, train_accuracies = [], [], []
    
    pbar = tqdm(train_dataloader, leave=False)
    for input_batch, target_batch in pbar:
        
        model_key, batch_key, key = jrand.split(key, 3)
        input_batch = jnp.asarray(input_batch.numpy(), dtype=jnp.float32)
        #print('let input batch size: ', input_batch.shape)
        target_batch = jnp.asarray(target_batch.numpy(), dtype=jnp.float32)
        one_hot_target_batch = jnp.asarray(nn.one_hot(target_batch, N_CLASSES), 
                                            dtype=jnp.float32)

        model, opt_state, loss = \
            eqx.filter_jit(update)(model, 
                                    optim,
                                    opt_state,  
                                    input_batch,
                                    one_hot_target_batch,
                                    one_hot_cross_entropy, 
                                    model_key)
            
        losses.append(loss/BATCH_SIZE)
        
        #wandb.log({"loss": loss/BATCH_SIZE})
        pbar.set_description(f"loss: {loss/BATCH_SIZE}")

    # Turn off dropout for model tests
    model = eqx.tree_inference(model, True)
    pbar = tqdm(train_dataloader, leave=False)
    for input_batch, target_batch in pbar:
        batch_key, key = jrand.split(key, 2)
        input_batch = jnp.asarray(input_batch.numpy(), dtype=jnp.float32)
        target_batch = jnp.asarray(target_batch.numpy(), dtype=jnp.float32)
        train_acc = calc_accuracy(model, 
                                model.init_state(SENSOR_DIM, batch_key), 
                                input_batch, 
                                target_batch,
                                key)

        train_accuracies.append(train_acc)

    tbar = tqdm(test_dataloader, leave=False)    
    for input_test, target_test in tbar:
        batch_key, key = jrand.split(key, 2)
        input_batch = jnp.asarray(input_test.numpy(), dtype=jnp.float32)
        target_batch = jnp.asarray(target_test.numpy(), dtype=jnp.float32)
        test_acc = calc_accuracy(model, 
                                model.init_state(SENSOR_DIM, batch_key), 
                                input_batch, 
                                target_batch,
                                key)
        test_accuracies.append(test_acc)

    model = eqx.tree_inference(model, False)
    #wandb.log({"train_accuracy": np.mean(train_accuracies)})
    #wandb.log({"test_accuracy": np.mean(test_accuracies)})
    nbar.set_description(f"epoch: {epoch}, loss = {np.mean(losses)}, train_accuracy = {np.mean(train_accuracies):.2f}, test_accuracy = {np.mean(test_accuracies):.2f}")



