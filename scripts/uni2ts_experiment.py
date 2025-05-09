import sys
import os
import logging
import datasets
import argparse
from torch_dataset import FlattenedMhcDataset
from huggingface_dataset import create_and_save_hf_dataset_as_gluonTS_style
import pandas as pd

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger(__name__)


def run_uni2ts_experiment(root_dir, dataset_df_path, output_hf_dataset_path, cache_dir, n_features, use_nans):
    dataset_df = pd.read_parquet(dataset_df_path)
    dataset_df.file_uris = dataset_df.file_uris.apply(eval)
    selected_features = list(range(n_features))

    torch_dataset = FlattenedMhcDataset(
        dataframe=dataset_df, # Using the dataframe loaded in the previous cell
        root_dir=root_dir,
        include_mask=True,
        feature_indices=selected_features,
        use_cache=False, # Disable caching for this potentially small/test dataset
        #feature_stats=feature_stats
        #postprocessors=[p0, p1]
    )

    create_and_save_hf_dataset_as_gluonTS_style(
        torch_dataset=torch_dataset,
        save_path=output_hf_dataset_path,
        # num_features will be inferred if None, otherwise provide it:
        # num_features=len(selected_features) if selected_features else None, 
        num_features=None, # Let the function try to infer
        include_mask_as_dynamic_feature=False,
        # Optional arguments (uncomment/adjust if needed):
        cache_dir=cache_dir, 
        # num_proc=8, 
        keep_in_memory=False,
        set_masked_target_to_nan=use_nans,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Uni2TS experiment')
    parser.add_argument('--root_dir', type=str,
                      help='Root directory containing the dataset')
    parser.add_argument('--output_hf_dataset_path', type=str,
                      help='Path to save the Hugging Face dataset')
    parser.add_argument('--cache_dir', type=str,
                      help='Cache directory for Hugging Face datasets')
    parser.add_argument('--dataset_df_path', type=str,
                      help='Path to the dataset dataframe')
    parser.add_argument('--n_features', type=int, default=6,
                      help='Number of features to use')
    parser.add_argument('--use_nans', type=bool, default=False,
                      help='Whether to use nans')
    
    args = parser.parse_args()
    
    run_uni2ts_experiment(
        root_dir=args.root_dir,
        dataset_df_path=args.dataset_df_path,
        output_hf_dataset_path=args.output_hf_dataset_path,
        cache_dir=args.cache_dir,
        n_features=args.n_features,
        use_nans=args.use_nans
    )