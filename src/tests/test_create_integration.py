import os
import shutil
import tempfile
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, List, Optional

from constants import FileType
from create import create_dataset


# Define specific file paths for real data
HEALTHKIT_PATH = '/Users/narayanschuetz/tmp_data/MHC_healthkit/Wmzvhl88mFNfAXCDqy-dUR9Y.parquet'
MOTION_PATH = None # Set Motion Path to None
SLEEP_PATH = '/Users/narayanschuetz/tmp_data/healthkit_sleep.parquet'
WORKOUT_PATH = '/Users/narayanschuetz/tmp_data/healthkit_workout.parquet'

# List of paths to check for existence
REQUIRED_PATHS = [
    p for p in [HEALTHKIT_PATH, MOTION_PATH, SLEEP_PATH, WORKOUT_PATH] if p is not None
]


@pytest.mark.skipif(
    not all(os.path.exists(p) for p in REQUIRED_PATHS),
    reason="One or more required data files not found. Test skipped."
)
def test_create_dataset_from_real_files_aug_2017():
    """
    Integration test using real data files, filtered for August 2017.
    Skips the test if any of the required files don't exist.
    """
    # Create a temporary directory for output
    output_dir = tempfile.mkdtemp()

    try:
        # Call function to process the real data files, filtered for Aug 2017
        create_dataset_from_paths(
            healthkit_path=HEALTHKIT_PATH,
            motion_path=MOTION_PATH,
            sleep_path=SLEEP_PATH,
            workout_path=WORKOUT_PATH,
            output_dir=output_dir,
            force_recompute=True,
            filter_start_date="2017-08-01",
            filter_end_date="2017-08-31"
        )

        # Check if any output was generated
        npy_files = list(Path(output_dir).glob("*.npy"))

        # Check that metadata was created (even if no daily files were made)
        metadata_file = os.path.join(output_dir, "metadata.parquet")
        assert os.path.exists(metadata_file), "Metadata file was not created"

        if npy_files:
             # Check first output file format if files were generated
             first_output = npy_files[0]
             data = np.load(first_output)
             assert data.shape[0] == 2, "Output should have 2 channels (mask and data)"

             # Verify metadata format
             metadata = pd.read_parquet(metadata_file)
             assert "date" in metadata.columns, "Metadata missing 'date' column"
             assert "data_coverage" in metadata.columns, "Metadata missing 'data_coverage' column"

        else:
             # If no npy files, metadata should be empty or reflect no processed days
             metadata = pd.read_parquet(metadata_file)
             assert metadata.empty or 'date' not in metadata.columns or metadata['date'].empty, "Metadata should be empty if no files processed"


    finally:
        pass
        # Clean up
        shutil.rmtree(output_dir)


def create_dataset_from_paths(
    output_dir: str,
    healthkit_path: Optional[str] = None,
    motion_path: Optional[str] = None,
    sleep_path: Optional[str] = None,
    workout_path: Optional[str] = None,
    skip: List[FileType] = [],
    force_recompute: bool = False,
    force_recompute_metadata: bool = False,
    filter_start_date: Optional[str] = None,
    filter_end_date: Optional[str] = None,
):
    """
    Create a dataset by loading DataFrames from file paths, optionally filtering by date,
    and passing them to create_dataset.
    """
    dfs = {}

    date_cols_map = {
        FileType.HEALTHKIT: ['startTime', 'endTime'],
        FileType.MOTION: ['startTime', 'endTime'],
        FileType.SLEEP: ['startTime'],
        FileType.WORKOUT: ['startTime', 'endTime']
    }

    path_map = {
        FileType.HEALTHKIT: healthkit_path,
        FileType.MOTION: motion_path,
        FileType.SLEEP: sleep_path,
        FileType.WORKOUT: workout_path,
    }

    start_ts = pd.Timestamp(filter_start_date) if filter_start_date else None
    end_ts = pd.Timestamp(filter_end_date) + pd.Timedelta(days=1) if filter_end_date else None

    for file_type, path in path_map.items():
        if path and file_type not in skip:
            df = pd.read_parquet(path)
            if df is not None and 'startTime' in df.columns:
                # Set index to startTime and remove tz info if present
                df.index = pd.to_datetime(df['startTime'], errors='coerce')
                if pd.api.types.is_datetime64tz_dtype(df.index):
                    df.index = df.index.tz_localize(None)
                # Now filter using the index (which is tz-naive)
                if start_ts or end_ts:
                    if start_ts:
                        df = df[df.index >= start_ts]
                    if end_ts:
                        df = df[df.index < end_ts]

            if not df.empty:
                if 'healthCode' in df.columns and "Wmzvhl88mFNfAXCDqy-dUR9Y" in df['healthCode'].values:
                    df = df[df.healthCode == "Wmzvhl88mFNfAXCDqy-dUR9Y"]
                dfs[file_type] = df

    create_dataset(
        dfs=dfs,
        output_root_dir=output_dir,
        skip=skip,
        force_recompute=force_recompute,
        force_recompute_metadata=force_recompute_metadata
    )


if __name__ == "__main__":
    # Example command-line usage
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--real-data":
        all_files_exist = all(os.path.exists(p) for p in REQUIRED_PATHS)

        if all_files_exist:
            print("Running test with real data filtered for August 2017...")
            output_dir = tempfile.mkdtemp()
            try:
                create_dataset_from_paths(
                    healthkit_path=HEALTHKIT_PATH,
                    motion_path=MOTION_PATH,
                    sleep_path=SLEEP_PATH,
                    workout_path=WORKOUT_PATH,
                    output_dir=output_dir,
                    force_recompute=True,
                    filter_start_date="2017-08-01",
                    filter_end_date="2017-08-31"
                )
                print(f"Processing complete. Output saved to {output_dir}")
            except Exception as e:
                print(f"Error processing real data: {e}")
            finally:
                 # Clean up even on error
                 if os.path.exists(output_dir):
                     shutil.rmtree(output_dir)
        else:
            print("One or more required data files not found. Test skipped.")
    else:
        print("Running test with sample data filtered for August 2017...")
        test_create_dataset_from_sample_files()
