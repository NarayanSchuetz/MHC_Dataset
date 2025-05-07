import datasets
from datasets import Features, Sequence, Value, DatasetInfo
import pandas as pd
import numpy as np
from typing import Generator, Any, Optional, List, Callable
import logging
import os
# import json # No longer needed for direct JSON manipulation here

# Assuming FlattenedMhcDataset is accessible, adjust import path if necessary
# from .torch_dataset import FlattenedMhcDataset
# For demonstration, let's assume it's in the same directory or package
try:
    from .torch_dataset import FlattenedMhcDataset, BaseMhcDataset # Use Flattened for (F, D*T) shape
except ImportError:
    # Fallback if running as a script directly
    from torch_dataset import FlattenedMhcDataset, BaseMhcDataset

logger = logging.getLogger(__name__)

def mhc_timeseries_generator(
    torch_dataset: BaseMhcDataset,
    target_key: str = 'data',
    # New parameters for explicit mask handling:
    input_mask_key: Optional[str] = None,
    apply_nan_using_mask: bool = False,
    add_mask_to_payload_as_feat_dynamic_real: bool = False
) -> Generator[dict[str, Any], None, None]:
    """
    Generates samples in the format expected by Hugging Face Datasets
    for multivariate time series, using data from an MhcDataset.

    Args:
        torch_dataset: An instantiated MhcDataset.
        target_key: The key in the torch_dataset sample dict holding the main time series data.
        input_mask_key: The optional key for the mask in the torch_dataset sample.
        apply_nan_using_mask: If True and mask is available, set masked values in target to np.nan.
        add_mask_to_payload_as_feat_dynamic_real: If True and mask is available, add mask to payload.

    Yields:
        dict: A dictionary for each sample.
    """
    if not isinstance(torch_dataset, FlattenedMhcDataset):
        logger.warning(f"Input dataset type is {type(torch_dataset)}. "
                       f"Expected FlattenedMhcDataset for (Features, Time) target shape. "
                       f"Proceeding, but ensure '{target_key}' has the correct shape.")

    for i in range(len(torch_dataset)):
        logger.debug(f"Generator processing sample index: {i}") # Log start of processing
        try:
            sample = torch_dataset[i]
            metadata = sample['metadata']
            health_code = metadata['healthCode']
            time_range_str = metadata['time_range']
            logger.debug(f"  Sample {i}: HC={health_code}, TimeRange={time_range_str}")
            
            # Target data: expected shape (num_features, total_time_points)
            target_data = sample.get(target_key)
            if target_data is None:
                logger.warning(f"  Sample {i}: Skipping - missing target key '{target_key}'.")
                continue
            if not isinstance(target_data, np.ndarray):
                 # BaseMhcDataset returns torch tensors, convert if needed
                 try:
                     target_data = target_data.numpy()
                 except AttributeError:
                     logger.warning(f"  Sample {i}: Skipping - target data is not numpy array or torch tensor (type: {type(target_data)})." )
                     continue

            if target_data.ndim != 2:
                logger.warning(f"  Sample {i}: Skipping - target data has incorrect dimensions "
                               f"({target_data.ndim}, shape {target_data.shape}). Expected 2D (features, time).")
                continue
            logger.debug(f"  Sample {i}: Target data shape OK: {target_data.shape}")

            # Process mask if a key is provided
            processed_mask_data = None
            if input_mask_key:
                raw_mask_data = sample.get(input_mask_key)
                if raw_mask_data is not None:
                    logger.debug(f"  Sample {i}: Found potential mask with key '{input_mask_key}'")
                    current_mask = raw_mask_data
                    if not isinstance(current_mask, np.ndarray):
                        try:
                            current_mask = current_mask.numpy()
                        except AttributeError:
                            logger.warning(f"  Sample {i}: Mask data for key '{input_mask_key}' is not numpy or torch tensor (type: {type(current_mask)}). Mask will not be used.")
                            current_mask = None
                    
                    if current_mask is not None: # If conversion was successful or it was already numpy
                        if current_mask.shape == target_data.shape:
                            processed_mask_data = current_mask # Valid mask obtained
                            logger.debug(f"  Sample {i}: Mask data (key '{input_mask_key}') shape OK: {processed_mask_data.shape}")
                        else:
                            logger.warning(f"  Sample {i}: Mask (key '{input_mask_key}') shape {current_mask.shape} "
                                           f"doesn't match target shape {target_data.shape}. Mask will not be used.")
                elif input_mask_key in sample: # Key exists but value is None
                    logger.debug(f"  Sample {i}: Mask key '{input_mask_key}' present in sample but data is None.")
                else: # Key not in sample
                    logger.debug(f"  Sample {i}: Mask key '{input_mask_key}' not found in sample keys: {list(sample.keys())}")

            # Apply NaN to masked values in target_data if requested and mask is available
            if apply_nan_using_mask and processed_mask_data is not None:
                if not np.issubdtype(target_data.dtype, np.floating):
                    logger.debug(f"  Sample {i}: Converting target_data from {target_data.dtype} to np.float32 for NaN application.")
                    target_data = target_data.astype(np.float32)
                logger.debug(f"  Sample {i}: Applying NaNs to target_data using mask (key '{input_mask_key}'). Assumes mask == 0 for masked regions.")
                target_data[processed_mask_data == 0] = np.nan

            # Start timestamp: Use the start date from the time_range
            try:
                start_date_str = time_range_str.split('_')[0]
                start_timestamp = pd.Timestamp(start_date_str, tz='UTC') # Assign UTC timezone
                logger.debug(f"  Sample {i}: Parsed start timestamp: {start_timestamp}")
            except (IndexError, ValueError):
                logger.warning(f"  Sample {i}: Skipping - invalid time_range format '{time_range_str}'.")
                continue

            # Frequency: Assuming minute-level data ('T') for the flattened dataset
            freq = 'T'

            # Item ID: Unique identifier for the time series sample
            item_id = f"{health_code}_{i}" # Or use time_range_str for more specificity if needed

            # Construct the payload
            payload = {
                "target": target_data.astype(np.float32), # Ensure float32
                "start": start_timestamp,
                "freq": freq,
                "item_id": item_id,
                # feat_static_cat is always present
                #"feat_static_cat": [health_code],
            }

            # Add labels as static real features if label_cols are defined for the dataset
            if hasattr(torch_dataset, 'label_cols') and torch_dataset.label_cols:
                num_defined_labels = len(torch_dataset.label_cols)
                # Initialize with NaNs, which is a safe default for float features
                current_static_real_features = [np.float32(np.nan)] * num_defined_labels

                sample_labels = sample.get('labels')
                if sample_labels: # If sample_labels dict exists
                    for i, label_col_name_with_suffix in enumerate(torch_dataset.label_cols):
                        # label_cols from BaseMhcDataset are sorted and have '_value'
                        label_key = label_col_name_with_suffix.replace('_value', '')
                        value = sample_labels.get(label_key, np.nan) # Get value, default to NaN
                        current_static_real_features[i] = np.float32(value) # Ensure float32
                
                payload["feat_static_real"] = current_static_real_features
            # If torch_dataset.label_cols is empty/None, feat_static_real is NOT added to payload,
            # matching the schema definition in create_and_save_hf_dataset.
            
            # Optionally include the processed mask as a dynamic real feature
            if add_mask_to_payload_as_feat_dynamic_real and processed_mask_data is not None:
                payload["feat_dynamic_real"] = processed_mask_data.astype(np.float32)
                logger.debug(f"  Sample {i}: Added mask data (from key '{input_mask_key}') as feat_dynamic_real.")
            elif add_mask_to_payload_as_feat_dynamic_real: # If requested but mask wasn't available/valid
                 logger.debug(f"  Sample {i}: feat_dynamic_real (mask) requested but no valid mask data was processed from key '{input_mask_key}'. It will not be added to payload.")

            logger.debug(f"  Sample {i}: Yielding payload for item_id {item_id}")
            yield payload

        except Exception as e:
            logger.error(f"Error processing sample index {i}: {e}", exc_info=True)
            # Decide whether to skip or re-raise
            continue


def create_and_save_hf_dataset_as_gluonTS_style(
    torch_dataset: BaseMhcDataset,
    save_path: str,
    num_features: Optional[int] = None,
    include_mask_as_dynamic_feature: bool = False,
    set_masked_target_to_nan: bool = False, # New parameter
    cache_dir: Optional[str] = None, # Optional cache directory
    num_proc: Optional[int] = None, # Optional number of processes
    keep_in_memory: bool = False # Optional flag to keep the dataset in memory
):
    """
    Creates a Hugging Face Dataset from an MhcDataset and saves it to disk.

    Args:
        torch_dataset: An instantiated MhcDataset (ideally FlattenedMhcDataset).
        save_path (str): The directory path where the dataset will be saved.
        num_features (Optional[int]): The number of features in the 'target' data.
                                      If None, it tries to infer from the first sample.
        include_mask_as_dynamic_feature (bool): If True, assumes a 'mask' key exists
                                                in the torch_dataset output and includes
                                                it as 'feat_dynamic_real' in the HF dataset.
        set_masked_target_to_nan (bool): If True, masked values in the 'target' data
                                         (identified by the mask) will be set to np.nan.
                                         Requires a mask to be available in torch_dataset.
        cache_dir (Optional[str]): Optional cache directory for Hugging Face datasets.
        num_proc (Optional[int]): Optional number of processes for dataset processing.
        keep_in_memory (bool): Optional flag to keep the dataset in memory.
    """
    # Infer number of features if not provided (use with caution)
    if num_features is None:
        try:
            sample_0 = torch_dataset[0]
            target_0 = sample_0['data']
            if hasattr(target_0, 'numpy'): # Convert if torch tensor
                target_0 = target_0.numpy()
            num_features = target_0.shape[0] # Assumes shape (F, T)
            logger.info(f"Inferred number of features: {num_features}")
        except Exception as e:
            logger.error(f"Could not infer number of features from the first sample: {e}")
            raise ValueError("Failed to infer num_features. Please provide it explicitly.")

    # Define the Hugging Face Features schema
    feature_dict = {
        # target shape: (num_features, sequence_length) -> Sequence of Sequences
        "target": Sequence(Sequence(Value("float32")), length=num_features),
        "start": Value("timestamp[us]", id='start'), # Use microsecond precision compatible with pd.Timestamp UTC
        "freq": Value("string"),
        "item_id": Value("string"),
        # For health_code
        #"feat_static_cat": Sequence(Value("string"), length=1),
    }
    
    # Add schema for feat_static_real if labels are present in torch_dataset
    if hasattr(torch_dataset, 'label_cols') and torch_dataset.label_cols:
        num_labels = len(torch_dataset.label_cols)
        if num_labels > 0:
            feature_dict["feat_static_real"] = Sequence(Value("float32"), length=num_labels)

    if include_mask_as_dynamic_feature:
        # Mask shape should match target: (num_features, sequence_length)
         feature_dict["feat_dynamic_real"] = Sequence(Sequence(Value("float32")), length=num_features)

    features = Features(feature_dict)

    # Define arguments for the generator function
    # Determine if the generator needs to access the mask from the input sample.
    # It needs to if we're applying NaNs OR if we're including it as a feature in output.
    # Assuming the key for mask in torch_dataset samples is 'mask'.
    mask_key_for_generator = None
    if set_masked_target_to_nan or include_mask_as_dynamic_feature:
        mask_key_for_generator = 'mask' # Standard key name

    gen_kwargs = {
        "torch_dataset": torch_dataset,
        # "target_key": "data", # Uses generator default
        "input_mask_key": mask_key_for_generator,
        "apply_nan_using_mask": set_masked_target_to_nan,
        "add_mask_to_payload_as_feat_dynamic_real": include_mask_as_dynamic_feature
    }

    # --- Custom metadata handling removed for now --- 

    # --- Create DatasetInfo object (minimal) --- 
    dataset_info = DatasetInfo(
        description="MHC Time Series Dataset", # Basic description
        features=features
        # No custom metadata passed here for now
    )
    logger.info(f"Prepared DatasetInfo with features (no custom metadata): {dataset_info}")

    # Create the Hugging Face dataset using the info object
    logger.info(f"Creating Hugging Face dataset using DatasetInfo...")
    hf_dataset = datasets.Dataset.from_generator(
        mhc_timeseries_generator, # Pass the generator FUNCTION
        gen_kwargs=gen_kwargs, # Pass arguments via gen_kwargs
        info=dataset_info, # Pass the minimal DatasetInfo object HERE
        keep_in_memory=keep_in_memory,
        cache_dir=cache_dir if cache_dir else None,
        num_proc=num_proc if num_proc else None 
    )

    # ---> Add logging here to check the dataset object <--- 
    logger.info(f"Dataset created. Info from dataset object: {hf_dataset.info}")
    if len(hf_dataset) == 0:
        logger.warning("The created dataset is empty. Check the generator function and input data.")
    # <------------------------------------------------------>
    
    # Save the dataset (core data and minimal DatasetInfo)
    logger.info(f"Saving dataset to disk at: {save_path}")
    # Try saving with num_proc=1 to simplify potential issues
    hf_dataset.save_to_disk(save_path, num_proc=1) 
    logger.info("Dataset successfully saved (with minimal DatasetInfo).")

    # --- Custom metadata JSON manipulation logic removed --- 

    # ---> List directory contents immediately after saving <---    
    try:
        dir_contents = os.listdir(save_path)
        logger.info(f"Contents of save directory '{save_path}': {dir_contents}")
    except Exception as list_e:
        logger.error(f"Could not list contents of save directory '{save_path}': {list_e}")
    # <--------------------------------------------------------->


# --- Example Usage ---
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Dummy Data Setup (Replace with actual data loading) ---
    # Create a dummy dataframe
    dummy_data = {
        'healthCode': ['participant_1', 'participant_2', 'participant_1'],
        'time_range': ['2023-01-01_2023-01-02', '2023-01-05_2023-01-05', '2023-01-03_2023-01-04'], # 2 days, 1 day, 2 days
        'file_uris': [
            ['participant_1/2023-01-01.npy', 'participant_1/2023-01-02.npy'],
            ['participant_2/2023-01-05.npy'],
            ['participant_1/2023-01-03.npy', 'participant_1/missing_day.npy'] # One missing file
        ],
        'happiness_value': [5.0, 7.0, np.nan],
        'sleep_value': [8.0, 6.5, 7.0]
    }
    dummy_df = pd.DataFrame(dummy_data)

    # Create dummy npy files in a temporary directory
    root_dir = "./temp_mhc_data"
    num_features_raw = 24
    num_time_points = 1440
    selected_features = [0, 5, 10] # Example feature selection
    num_selected_features = len(selected_features)

    os.makedirs(os.path.join(root_dir, "participant_1"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "participant_2"), exist_ok=True)

    # File 1: p1, day 1
    np.save(os.path.join(root_dir, "participant_1", "2023-01-01.npy"),
            np.random.rand(2, num_features_raw, num_time_points).astype(np.float32)) # (Mask+Data, F, T)
    # File 2: p1, day 2
    np.save(os.path.join(root_dir, "participant_1", "2023-01-02.npy"),
            np.random.rand(2, num_features_raw, num_time_points).astype(np.float32))
     # File 3: p2, day 1
    np.save(os.path.join(root_dir, "participant_2", "2023-01-05.npy"),
            np.random.rand(2, num_features_raw, num_time_points).astype(np.float32))
    # File 4: p1, day 3
    np.save(os.path.join(root_dir, "participant_1", "2023-01-03.npy"),
            np.random.rand(2, num_features_raw, num_time_points).astype(np.float32))
    # Note: participant_1/missing_day.npy is intentionally not created for the 4th day of the last sample

    # --- Instantiate FlattenedMhcDataset ---
    try:
        flattened_torch_dataset = FlattenedMhcDataset(
            dataframe=dummy_df,
            root_dir=root_dir,
            include_mask=True, # Set to True if you want mask data
            feature_indices=selected_features, # Use selected features
            use_cache=False # Disable caching for this example
        )
        
        # Check the output shape of the first sample
        sample0 = flattened_torch_dataset[0]
        logger.info(f"Sample 0 data shape (Flattened): {sample0['data'].shape}") # Should be (num_selected_features, num_days * 1440)
        logger.info(f"Sample 0 mask shape (Flattened): {sample0['mask'].shape}")

        # --- Create and Save Hugging Face Dataset ---
        output_hf_dataset_path = "./mhc_hf_dataset_flattened"
        create_and_save_hf_dataset_as_gluonTS_style(
            torch_dataset=flattened_torch_dataset,
            save_path=output_hf_dataset_path,
            num_features=num_selected_features, # Pass the number of *selected* features
            include_mask_as_dynamic_feature=True, # Include mask as feat_dynamic_real
            set_masked_target_to_nan=True # New: Set masked values in target to NaN
        )

        # --- Verification (Optional) ---
        logger.info(f"Verifying saved dataset at {output_hf_dataset_path}...")
        reloaded_dataset = datasets.load_from_disk(output_hf_dataset_path)
        print(reloaded_dataset)
        print(reloaded_dataset[0]) # Print the first sample

    except Exception as e:
         logger.error(f"An error occurred during example execution: {e}", exc_info=True)
    finally:
        # Clean up dummy files
        import shutil
        if os.path.exists(root_dir):
             logger.info(f"Cleaning up dummy data directory: {root_dir}")
             shutil.rmtree(root_dir)
        # if os.path.exists(output_hf_dataset_path):
        #     logger.info(f"Cleaning up dummy HF dataset directory: {output_hf_dataset_path}")
        #     shutil.rmtree(output_hf_dataset_path)
