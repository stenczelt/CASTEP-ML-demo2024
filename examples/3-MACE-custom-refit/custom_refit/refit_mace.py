#  Hybrid MD decision making package
#
#  Copyright (c) Tamas K. Stenczel 2021-2024.
"""Refitting of MACE models on the fly"""
import json
from argparse import Namespace
from typing import Optional

import ase.io

from mace.cli.run_train import run as train_mace_model
from hybrid_md.refit_mace import ExtendedCheckpointIO, choose_device, convert_to_lammps
from hybrid_md.state_objects import HybridMD


RANDOM_SEED = 123


def refit_generic(
    state: HybridMD,
    device=None,
    random_seed=RANDOM_SEED,
    model_kwargs: Optional[dict] = None,
):
    """Refit a MACE model

    Parameters
    ----------
    state: HybridMD
    device: device for torch
    random_seed
    model_kwargs
    """

    print("&" * 200)

    # extract main settings
    refit_settings = state.settings.refit

    # settings
    model_name = refit_settings.gp_name  # JIT-compiled model which we can run in LAMMPS
    epochs_initial = 10
    epochs_additional = 6
    epoch_swa_from = 0.8  # fraction of the total new ones

    # see if first one or not
    checkpoint_io = ExtendedCheckpointIO(
        directory="checkpoints",
        tag=f"MACE_model_run-{RANDOM_SEED}",
        keep=True,
    )
    last_epoch = checkpoint_io.get_last_epoch_number()
    if last_epoch is None:
        # no checkpoint, so we are starting from scratch
        max_num_epochs = epochs_initial
        start_swa = int(max_num_epochs * epoch_swa_from)
        print("Starting MACE training from scratch")
    else:
        max_num_epochs = last_epoch + epochs_additional
        start_swa = last_epoch + int(epochs_additional * epoch_swa_from)
        print(f"Continuing MACE training from epoch {last_epoch}")

    # gather training set
    train_frames = state.get_previous_data() + ase.io.read(state.xyz_filename, ":")

    # we need at least 1 train + 1 validation + 1 test structure.
    if len(train_frames) < 3:
        print(f"Fewer than 3 structures available for fine-tuning.")
        raise RuntimeError

    # split structures - as simple as it gets, 1:1:1
    ase.io.write("train.xyz", train_frames[::3])
    ase.io.write("test.xyz", train_frames[1::3])
    ase.io.write("validation.xyz", train_frames[1::3])

    # arguments & overwrites
    train_args = default_args()
    if device is None:
        train_args["device"] = choose_device()
    else:
        train_args["device"] = device
    train_args.update(
        {
            "max_num_epochs": max_num_epochs,
            "start_swa": start_swa,
            "seed": random_seed,
            "E0s": "{6:-148.1738815378, 14:-163.4954402249}",  # Si & C Example
        }
    )
    if model_kwargs:
        train_args.update(model_kwargs)

    # training
    print("Starting MACE training with parameters:" + json.dumps(train_args))
    model = train_mace_model(Namespace(**train_args))
    print("Training completed.")

    # JIT-compiled model for C++/F90 to evaluators
    convert_to_lammps(model, model_name)


def default_args():
    return {
        "name": "MACE_model",
        "batch_size": 10,
        "valid_batch_size": 10,
        "max_num_epochs": 10,
        "swa": True,
        "start_swa": 8,
        "E0s": "average",
        "keep_isolated_atoms": True,
        "work_dir": ".",
        "save_all_checkpoints": False,
        "restart_latest": True,
        # ------------------------------------
        "foundation_model": "/projappl/project_2010950/castep-ml/mace-cache/mace_agnesi_mediummodel",
        "foundation_model_readout": True,
        "multiheads_finetuning": True,
        "foundation_filter_elements": True,
        "weight_pt_head": 1.0,
        "num_samples_pt": 10,  # TODO: use min 50, this is for demonstration
        "subselect_pt": "random",
        "pt_train_file": None,
        "pt_valid_file": None,
        "heads": None,
        # ------------------------------------
        "train_file": "train.xyz",
        "valid_file": "validation.xyz",
        "test_file": "test.xyz",
        "config": None,
        # ------------------------------------
        # new parameters
        "pair_repulsion": False,
        "distance_transform": "None",
        "distributed": False,
        "test_dir": None,
        "multi_processed_test": False,
        "num_workers": 0,
        "pin_memory": True,
        "atomic_numbers": None,
        "mean": None,
        "std": None,
        "statistics_file": None,
        "beta": 0.9,
        # ------------------------------------
        "model": "MACE",
        "device": "cpu",
        "r_max": 5.0,
        "seed": RANDOM_SEED,
        "log_dir": "./logs",
        "model_dir": ".",
        "checkpoints_dir": "./checkpoints",
        "results_dir": "./results",
        "downloads_dir": "./downloads",
        "default_dtype": "float64",
        "log_level": "INFO",
        "error_table": "PerAtomRMSE",
        "radial_type": "bessel",
        "num_radial_basis": 8,
        "num_cutoff_basis": 5,
        "interaction": "RealAgnosticResidualInteractionBlock",
        "interaction_first": "RealAgnosticResidualInteractionBlock",
        "max_ell": 3,
        "correlation": 3,
        "num_interactions": 2,
        "MLP_irreps": "16x0e",
        "radial_MLP": "[64, 64, 64]",
        "hidden_irreps": "128x0e + 128x1o",
        "num_channels": 128,
        "max_L": 1,
        "gate": "silu",
        "scaling": "rms_forces_scaling",
        "avg_num_neighbors": 1,
        "compute_avg_num_neighbors": True,
        "compute_stress": False,
        "compute_forces": True,
        "valid_fraction": 0.1,
        "energy_key": "QM_energy",
        "forces_key": "QM_forces",
        "virials_key": "QM_virials",
        "stress_key": "QM_stress",
        "dipole_key": "QM_dipole",
        "charges_key": "QM_charges",
        "loss": "weighted",
        "forces_weight": 100.0,
        "swa_forces_weight": 100.0,
        "energy_weight": 1.0,
        "swa_energy_weight": 1000.0,
        "virials_weight": 1.0,
        "swa_virials_weight": 10.0,
        "stress_weight": 1.0,
        "swa_stress_weight": 10.0,
        "dipole_weight": 1.0,
        "swa_dipole_weight": 1.0,
        "config_type_weights": '{"Default":1.0}',
        "huber_delta": 0.01,
        "optimizer": "adam",
        "lr": 0.01,
        "swa_lr": 0.001,
        "weight_decay": 5e-07,
        "amsgrad": True,
        "scheduler": "ReduceLROnPlateau",
        "lr_factor": 0.8,
        "scheduler_patience": 50,
        "lr_scheduler_gamma": 0.9993,
        "ema": False,
        "ema_decay": 0.99,
        "patience": 2048,
        "eval_interval": 1,
        "keep_checkpoints": False,
        "save_cpu": False,
        "clip_grad": 10.0,
        "wandb": False,
        "wandb_dir": None,
        "wandb_project": "",
        "wandb_entity": "",
        "wandb_name": "",
        "wandb_log_hypers": [
            "num_channels",
            "max_L",
            "correlation",
            "lr",
            "swa_lr",
            "weight_decay",
            "batch_size",
            "max_num_epochs",
            "start_swa",
            "energy_weight",
            "forces_weight",
        ],
    }
