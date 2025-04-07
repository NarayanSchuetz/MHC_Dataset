import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from pathlib import Path


# NOTE: the way of accessing the data is not very optimized (pandas iloc is not the fastest, we could speed this up by even just using a dict).


class BaseMhcDataset(Dataset):
    """
    PyTorch Dataset for loading MHC data from a denormalized DataFrame.

    Expects a DataFrame like the one produced by `labelled_dataset.py`, containing:
    - 'healthCode': Participant identifier.
    - 'time_range': The time window for the data.
    - 'file_uris': Path to the .npy file containing the time-series data, relative to root_dir.
    - Label columns ending in '_value' (e.g., 'happiness_value').
    - Corresponding label date columns ending in '_date' (optional, not used by default loader).
    """

    def __init__(self, dataframe: pd.DataFrame, root_dir: str):
        """
        Args:
            dataframe (pd.DataFrame): The denormalized dataframe containing metadata and labels.
            root_dir (str): The root directory where the .npy files specified in 'file_uris' are located.
        """
        super().__init__()
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        if 'file_uris' not in dataframe.columns:
            raise ValueError("DataFrame must contain a 'file_uris' column.")
        if 'healthCode' not in dataframe.columns:
             print("Warning: 'healthCode' column not found in DataFrame.")

        self.df = dataframe.reset_index(drop=True) # Ensure consistent indexing
        self.root_dir = Path(os.path.expanduser(root_dir))

        # Identify label columns (those ending in _value)
        self.label_cols = sorted([col for col in self.df.columns if col.endswith('_value')])
        print(f"Initialized Dataset with {len(self.df)} samples.")
        print(f"Found label columns: {self.label_cols}")
        print(f"Root directory set to: {self.root_dir.resolve()}")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'data' (torch.Tensor): The time-series data loaded from the .npy file (as FloatTensor).
                - 'labels' (dict): A dictionary where keys are label names (without '_value')
                                   and values are the corresponding label values (as float, np.nan if missing).
                - 'metadata' (dict): Contains 'healthCode', 'time_range', and 'file_uri' for reference.
        """
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.df)}")

        row = self.df.iloc[idx]
        file_uri = row['file_uris']
        data_path = self.root_dir / file_uri

        try:
            # Load the numpy array
            # Ensure data is loaded as float32 for consistency with PyTorch FloatTensors
            data_array = np.load(data_path).astype(np.float32)
            # Convert to PyTorch tensor
            data_tensor = torch.from_numpy(data_array)
        except FileNotFoundError:
            print(f"ERROR: File not found at {data_path}. Check root_dir and file_uris.")
            # Return None or raise error, depending on desired handling
            # For now, raising an error might be better during development
            raise FileNotFoundError(f"Could not load data for index {idx}. File not found: {data_path}")
        except Exception as e:
            print(f"ERROR: Could not load or process file {data_path}. Error: {e}")
            raise IOError(f"Failed to load or process data for index {idx} from {data_path}") from e


        # Extract labels
        labels = {}
        for label_col in self.label_cols:
            label_name = label_col.replace('_value', '')
            # Ensure labels are float, preserving NaN
            label_value = float(row[label_col]) if pd.notna(row[label_col]) else np.nan
            labels[label_name] = label_value

        # Include metadata which might be useful
        metadata = {
            'healthCode': row.get('healthCode', 'N/A'),
            'time_range': row.get('time_range', 'N/A'),
            'file_uri': file_uri
        }

        return {
            'data': data_tensor,
            'labels': labels,
            'metadata': metadata
        }



class FilteredMhcDataset(BaseMhcDataset):
    """
    A subclass of BaseMhcDataset that filters the dataframe to only include samples
    where a specified label of interest is not NaN.
    
    This dataset is useful when training models for a specific prediction task
    where you only want samples that have the target label available.
    
    Usage Example:
        >>> # Assume 'full_df' is your loaded denormalized dataframe
        >>> # Assume 'data_root' is the path to the directory with .npy files
        >>> happiness_dataset = FilteredMhcDataset(
        ...     dataframe=full_df, 
        ...     root_dir=data_root, 
        ...     label_of_interest='happiness'
        ... )
        >>> # Now happiness_dataset only contains samples with a non-NaN happiness_value
        >>> # You can use it with a DataLoader:
        >>> # from torch.utils.data import DataLoader
        >>> # data_loader = DataLoader(happiness_dataset, batch_size=32)
    """

    def __init__(self, dataframe: pd.DataFrame, root_dir: str, label_of_interest: str):
        """
        Args:
            dataframe (pd.DataFrame): The denormalized dataframe containing metadata and labels.
            root_dir (str): The root directory where the .npy files specified in 'file_uris' are located.
            label_of_interest (str): The label to filter by (without '_value' suffix).
                                     Only samples where this label is not NaN will be included.
        """
        # Ensure the label_of_interest has the '_value' suffix for filtering
        label_col = f"{label_of_interest}_value" if not label_of_interest.endswith('_value') else label_of_interest
        
        # Check if the label exists in the dataframe
        if label_col not in dataframe.columns:
            raise ValueError(f"Label column '{label_col}' not found in the dataframe.")
        
        # Filter the dataframe to only include rows where the label is not NaN
        filtered_df = dataframe[dataframe[label_col].notna()].copy()
        
        if len(filtered_df) == 0:
            raise ValueError(f"No samples found with non-NaN values for label '{label_of_interest}'.")
        
        print(f"Filtered dataset from {len(dataframe)} to {len(filtered_df)} samples with non-NaN '{label_of_interest}' values.")
        
        # Initialize the parent class with the filtered dataframe
        super().__init__(filtered_df, root_dir)
        
        # Store the label of interest for reference
        self.label_of_interest = label_of_interest
