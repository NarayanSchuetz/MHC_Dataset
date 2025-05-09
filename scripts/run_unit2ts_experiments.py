#!/usr/bin/env python3

import os
from pathlib import Path
from uni2ts_experiment import run_uni2ts_experiment

# Default configuration
DEFAULT_CONFIG = {
    'paths': {
        'data_dir': '/mnt/nvme/mhc_dataset/',
        'dataset_file': '/mnt/shared/mhc_dataset_out/splits/test_dataset.parquet',
        'output_dir': None,
        'cache_dir': None
    },
    'experiment_params': {
        'n_features': 6,
        'use_nans': False,
        'experiment_name': 'default_experiment'
    }
}

def setup_directories(paths):
    """Create necessary directories for the experiment."""
    for dir_path in paths.values():
        Path(dir_path).parent.mkdir(parents=True, exist_ok=True)

def run_experiment(config=None):
    """
    Run the uni2ts experiment with the given configuration.
    
    Args:
        config (dict): Configuration dictionary. If None, uses DEFAULT_CONFIG.
                      Any keys in config will override DEFAULT_CONFIG values.
    """
    # Merge default config with provided config
    experiment_config = DEFAULT_CONFIG.copy()
    if config:
        # Deep merge for nested dictionaries
        if 'paths' in config:
            experiment_config['paths'].update(config['paths'])
        if 'experiment_params' in config:
            experiment_config['experiment_params'].update(config['experiment_params'])
    
    # Setup directories
    setup_directories(experiment_config['paths'])
    
    # Create experiment-specific output directory
    experiment_output_dir = Path(experiment_config['paths']['output_dir']) / experiment_config['experiment_params']['experiment_name']
    experiment_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the experiment
    run_uni2ts_experiment(
        root_dir=experiment_config['paths']['data_dir'],
        dataset_df_path=experiment_config['paths']['dataset_file'],
        output_hf_dataset_path=str(experiment_output_dir / 'uni2ts_dataset'),
        cache_dir=experiment_config['paths']['cache_dir'],
        n_features=experiment_config['experiment_params']['n_features'],
        use_nans=experiment_config['experiment_params']['use_nans']
    )

def main():
    experiments = {
        # 'baseline': {
        #     'paths': {
        #         'output_dir': '/mnt/shared/mhc_test_hf_nan',
        #         'cache_dir': '/mnt/shared/mhc_test_hf_nan_cache'
        #     },
        #     'experiment_params': {
        #         'experiment_name': 'baseline',
        #         'n_features': 6,
        #         'use_nans': True
        #     }
        # }
        'full_dataset_test_6var': {
            'paths': {
                'dataset_file': '/mnt/shared/mhc_dataset_out/splits/test_dataset.parquet',
                'output_dir': '/mnt/shared/mhc_test_hf_full_6var',
                'cache_dir': '/mnt/shared/mhc_test_hf_full_6var_cache'
            },
            'experiment_params': {
                'experiment_name': 'full_dataset_test_6var',
                'n_features': 6,
                'use_nans': True
            }
        },

        'full_dataset_train_6var': {
            'paths': {
                'dataset_file': '/mnt/shared/mhc_dataset_out/splits/train_final_dataset.parquet',
                'output_dir': '/mnt/shared/mhc_train_hf_full_6var',
                'cache_dir': '/mnt/shared/mhc_train_hf_full_6var_cache'
            },
            'experiment_params': {
                'experiment_name': 'full_dataset_train_6var',
                'n_features': 6,
                'use_nans': True
            }
        },

        'full_dataset_val_6var': {
            'paths': {
                'dataset_file': '/mnt/shared/mhc_dataset_out/splits/validation_dataset.parquet',
                'output_dir': '/mnt/shared/mhc_val_hf_full_6var',
                'cache_dir': '/mnt/shared/mhc_val_hf_full_6var_cache'
            },
            'experiment_params': {
                'experiment_name': 'full_dataset_val_6var',
                'n_features': 6,
                'use_nans': True
            }
        }
    }
    
    # Run all experiments
    for exp_name, exp_config in experiments.items():
        print(f"\nRunning experiment: {exp_name}")
        run_experiment(exp_config)

if __name__ == "__main__":
    main()
