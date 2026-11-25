"""Central experiment and method definitions.

Keep scientific choices here so training/evaluation scripts do not silently diverge.
"""

DEFAULT_METHODS = [
    "baseline_plain",
    "baseline_sn",
    "zc_gp",
    "zc_gp_sn",
    "asym_tik",
    "asym_tik_sn",
]

METHOD_CONFIGS = {
    "baseline_plain": {
        "use_sn": False,
        "use_asym_gp": False,
        "use_tikh_d": False,
        "use_tikh_g": False,
        "gamma_gp": 0.0,
        "gamma_tik": 0.0,
    },
    "baseline_sn": {
        "use_sn": True,
        "use_asym_gp": False,
        "use_tikh_d": False,
        "use_tikh_g": False,
        "gamma_gp": 0.0,
        "gamma_tik": 0.0,
    },
    "zc_gp": {
        "use_sn": False,
        "use_asym_gp": True,
        "use_tikh_d": False,
        "use_tikh_g": False,
        "gamma_gp": 0.5,
        "gamma_tik": 0.0,
    },
    "zc_gp_sn": {
        "use_sn": True,
        "use_asym_gp": True,
        "use_tikh_d": False,
        "use_tikh_g": False,
        "gamma_gp": 0.5,
        "gamma_tik": 0.0,
    },
    "asym_tik": {
        "use_sn": False,
        "use_asym_gp": False,
        "use_tikh_d": True,
        "use_tikh_g": False,
        "gamma_gp": 0.0,
        "gamma_tik": 1e-6,
    },
    "asym_tik_sn": {
        "use_sn": True,
        "use_asym_gp": False,
        "use_tikh_d": True,
        "use_tikh_g": False,
        "gamma_gp": 0.0,
        "gamma_tik": 1e-6,
    },
    # Controls retained for the original/full CIFAR study, but not run by default
    # in the new cross-domain/cross-architecture validation.
    "sym_tik": {
        "use_sn": False,
        "use_asym_gp": False,
        "use_tikh_d": True,
        "use_tikh_g": True,
        "gamma_gp": 0.0,
        "gamma_tik": 1e-6,
    },
    "sym_tik_sn": {
        "use_sn": True,
        "use_asym_gp": False,
        "use_tikh_d": True,
        "use_tikh_g": True,
        "gamma_gp": 0.0,
        "gamma_tik": 1e-6,
    },
}

EXPERIMENTS = {
    "cifar10_dcgan32": {
        "dataset": "cifar10",
        "architecture": "dcgan",
        "image_size": 32,
        "batch_size": 64,
        "z_dim": 128,
        "lr_g": 2e-4,
        "lr_d": 2e-4,
        "betas": (0.0, 0.9),
        "split_seed": 123,
        "train_fraction": 1.0,
        "compute_is": True,
    },
    "plantvillage_dcgan64": {
        "dataset": "plantvillage",
        "architecture": "dcgan",
        "image_size": 64,
        "batch_size": 64,
        "z_dim": 128,
        "lr_g": 2e-4,
        "lr_d": 2e-4,
        "betas": (0.0, 0.9),
        "split_seed": 123,
        "train_fraction": 0.80,
        "split_protocol": "official_leaf_grouped_80_20",
        "compute_is": False,
    },
    "cifar10_resnet32": {
        "dataset": "cifar10",
        "architecture": "resnet",
        "image_size": 32,
        "batch_size": 64,
        "z_dim": 128,
        "lr_g": 2e-4,
        "lr_d": 2e-4,
        "betas": (0.0, 0.9),
        "split_seed": 123,
        "train_fraction": 1.0,
        "compute_is": True,
    },
}

STAGES = {
    # Fast wiring check. Never use these results in the paper.
    "smoke": {
        "seeds": [999],
        "methods": ["baseline_plain"],
        "max_steps": 200,
    },
    # One dedicated tuning/diagnostic seed. Exclude seed 999 from final results.
    "pilot": {
        "seeds": [999],
        "methods": DEFAULT_METHODS,
        "max_steps": 10_000,
    },
    # Final validation uses three untouched seeds. The full update budget must be
    # chosen and frozen after the pilot, so max_steps is intentionally None.
    "final": {
        "seeds": [0, 1, 2],
        "methods": DEFAULT_METHODS,
        "max_steps": None,
    },
}
