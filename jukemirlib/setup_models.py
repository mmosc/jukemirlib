import os
from pathlib import Path
from tqdm import tqdm

# imports and set up Jukebox's multi-GPU parallelization
import jukebox
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.make_models import MODELS, make_prior, make_vqvae

from accelerate import init_empty_weights

import torch.nn as nn
import torch

# this is a huggingface accelerate method, all we do is just
# remove the type hints that we don't want to import in the header
def set_module_tensor_to_device(
    module: nn.Module, tensor_name: str, device, value=None
):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).
    Args:
        module (`torch.nn.Module`): The module in which the tensor we want to move lives.
        param_name (`str`): The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`): The device on which to set the tensor.
        value (`torch.Tensor`, *optional*): The value of the tensor (useful when going from the meta device to any
            other device).
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    with torch.no_grad():
        if value is None:
            new_value = old_value.to(device)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device)
        else:
            new_value = torch.tensor(value, device=device)

        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif value is not None or torch.device(device) != module._parameters[tensor_name].device:
            param_cls = type(module._parameters[tensor_name])
            kwargs = module._parameters[tensor_name].__dict__
            new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(device)
            module._parameters[tensor_name] = new_value

def setup_models(cache_dir="/juice/scr/rjcaste/jukemirlib-testing/model_cache",
                 verbose=True,
                 device="cuda"):
    # caching preliminaries
    VQVAE_CACHE_PATH = cache_dir + '/vqvae.pth.tar'
    PRIOR_CACHE_PATH = cache_dir + '/prior_level_2.pth.tar'
    os.makedirs(cache_dir, exist_ok=True)

    if verbose:
        print("Importing jukebox and associated packages...")

    # Set up MPI
    #rank, local_rank, device = setup_dist_from_mpi()

    # Set up VQVAE
    if verbose:
        print("Setting up the VQ-VAE...")
    model = "5b"  # or "1b_lyrics"
    hps = Hyperparams()
    hps.sr = 44100
    hps.n_samples = 3 if model == "5b_lyrics" else 8
    hps.name = "samples"
    chunk_size = 16 if model == "5b_lyrics" else 32
    max_batch_size = 3 if model == "5b_lyrics" else 16
    hps.levels = 3
    hps.hop_fraction = [0.5, 0.5, 0.125]
    vqvae, *priors = MODELS[model]

    hparams = setup_hparams(vqvae, dict(sample_length=1048576))

    # hparams.restore_vqvae = VQVAE_CACHE_PATH

    # don't actually load any weights in yet,
    # leave it for later. memory optimization
    with init_empty_weights():
        vqvae = make_vqvae(
            hparams, 'meta'
        )

    # Set up language model
    if verbose:
        print("Setting up the top prior...")
    hparams = setup_hparams(priors[-1], dict())

    # IMPORTANT LINE: only include layers UP TO prior_depth
    #hparams["prior_depth"] = 72

    # hparams.restore_prior = PRIOR_CACHE_PATH

    # don't actually load any weights in yet,
    # leave it for later. memory optimization
    with init_empty_weights():
        top_prior = make_prior(hparams, vqvae, 'meta')

    # flips a bit that tells the model to return activations
    # instead of projecting to tokens and getting loss for
    # forward pass
    top_prior.prior.only_encode = True

    ##############################################
    # actually loading in the model weights now! #
    ##############################################
    if verbose:
        print("Now actually loading in the model weights...")

    if verbose:
        print("Loading the top prior weights into memory...")
    top_prior_weights = torch.load(PRIOR_CACHE_PATH, map_location='cpu')

    # load_state_dict, basically
    if verbose:
        print("Loading the top prior weights into our empty model...")
    for k in tqdm(top_prior_weights['model'].keys()):
        set_module_tensor_to_device(top_prior, k, device, value=top_prior_weights['model'][k])

    del top_prior_weights

    import gc
    gc.collect()

    if verbose:
        print("Loading the VQ-VAE weights into memory...")
    vqvae_weights = torch.load(VQVAE_CACHE_PATH, map_location='cpu')

    if verbose:
        print("Loading the VQ-VAE weights into our empty model...")
    for k in tqdm(vqvae_weights['model'].keys()):
        set_module_tensor_to_device(vqvae, k, device, value=vqvae_weights['model'][k])

    return vqvae, top_prior