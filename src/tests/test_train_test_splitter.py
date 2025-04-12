import pandas as pd
import numpy as np
import pytest
import os # Needed for checking file existence

# Import the function to test (adjust path if necessary)
# from src.train_test_splitter import stratify_and_split_by_nan_labels # Removed as no longer used
from src.train_test_splitter import  stratified_split_cluster, stratified_split_advanced

# Define paths to real data files (replace with actual paths if different)
REAL_DEMO_PATH = "/Users/narayanschuetz/tmp_data/mhc_demographics.parquet"
REAL_INFO_PATH = "/Users/narayanschuetz/tmp_data/mhc_full_participant_info.parquet"
REAL_DATASET_PATH = "/Users/narayanschuetz/Downloads/global_records.parquet" # Corrected filename

# Check if all real data files exist
real_data_exists = (
    os.path.exists(REAL_DEMO_PATH) and 
    os.path.exists(REAL_INFO_PATH) and 
    os.path.exists(REAL_DATASET_PATH)
)

@pytest.mark.skipif(not real_data_exists, reason="Real data files not found at specified paths")
def test_cluster_split_real_data():
    """Integration test for cluster-based split using real data files."""
    print(f"\nRunning cluster split integration test with real data:")
    print(f"  Demographics: {REAL_DEMO_PATH}")
    print(f"  Info:         {REAL_INFO_PATH}")
    print(f"  Dataset:      {REAL_DATASET_PATH}")
    
    # Load real data
    demographic_df = pd.read_parquet(REAL_DEMO_PATH)
    info_df = pd.read_parquet(REAL_INFO_PATH)
    dataset_df = pd.read_parquet(REAL_DATASET_PATH)
    print("Successfully loaded real data files.")
        
    test_size = 0.2
    random_state = 456 # Use a different random state 
    n_clusters = 8 
    
    # Run the cluster split function
    train_df, test_df = stratified_split_cluster(
        dataset_df=dataset_df,
        demographic_df=demographic_df,
        info_df=info_df,
        test_size=test_size,
        n_clusters=n_clusters,
        random_state=random_state
    )
    print(f"Successfully ran stratified_split_cluster with n_clusters={n_clusters} on real data.")

    # --- Assertions --- 
    
    # 1. Check for overlap
    train_hc = set(train_df['healthCode'].unique())
    test_hc = set(test_df['healthCode'].unique())
    assert len(train_hc.intersection(test_hc)) == 0, "Overlap detected between train/test healthCodes"
    print("Assertion Passed: No healthCode overlap.")

    # 2. Check if all dataset healthCodes are present (including those without demographic data)
    all_dataset_hcs = set(dataset_df['healthCode'].unique())
    split_hc_set = train_hc | test_hc
    
    # Check for missing healthCodes
    missing_from_split = all_dataset_hcs - split_hc_set
    assert len(missing_from_split) == 0, f"{len(missing_from_split)} healthCodes from dataset are missing from the split"
    
    # Check for extra healthCodes (shouldn't be any)
    extra_in_split = split_hc_set - all_dataset_hcs
    assert len(extra_in_split) == 0, f"{len(extra_in_split)} healthCodes in split are not from the dataset"
    
    print(f"Assertion Passed: All {len(all_dataset_hcs)} dataset healthCodes are included in the split (train: {len(train_hc)}, test: {len(test_hc)}).")
    
    # Get the demographic healthCodes for comparison calculations
    demographic_df_copy = demographic_df.copy()
    demographic_df_copy['nan_count'] = demographic_df_copy.isna().sum(axis=1)
    deduped_demo = demographic_df_copy.sort_values('nan_count').drop_duplicates(subset=['healthCode'], keep='first')
    
    # Get healthCodes with and without demographic data
    demo_hcs = set(deduped_demo['healthCode'].unique())
    dataset_with_demo_hcs = all_dataset_hcs & demo_hcs
    dataset_without_demo_hcs = all_dataset_hcs - demo_hcs
    
    print(f"Dataset healthCodes with demographic data: {len(dataset_with_demo_hcs)}")
    print(f"Dataset healthCodes without demographic data: {len(dataset_without_demo_hcs)}")
    
    # 3. Check split shapes (approximate)
    expected_test_hc = int(round(len(all_dataset_hcs) * test_size))
    expected_train_hc = len(all_dataset_hcs) - expected_test_hc
    
    # Allow a small tolerance due to stratification and potential edge cases in clustering/splitting
    # Sklearn's train_test_split with stratification might slightly deviate
    tolerance = max(2, int(len(all_dataset_hcs) * 0.02)) # Tolerance of 2 or 2%, whichever is larger
    assert abs(len(train_hc) - expected_train_hc) <= tolerance, f"Train set size deviates too much ({len(train_hc)} vs {expected_train_hc}, tolerance {tolerance})"
    assert abs(len(test_hc) - expected_test_hc) <= tolerance, f"Test set size deviates too much ({len(test_hc)} vs {expected_test_hc}, tolerance {tolerance})"
    print(f"Assertion Passed: Split shapes are approximately correct (Train: {len(train_hc)}, Test: {len(test_hc)}).")

    # --- 4. Detailed Distribution Comparison ---
    print("\n--- Comparing Feature Distributions ---")
    print("Note: Distribution comparisons only include healthCodes with demographic data")
    print(f"({len(dataset_with_demo_hcs)} of {len(all_dataset_hcs)} total healthCodes, {len(dataset_without_demo_hcs)} healthCodes lack demographic data)")

    # Recalculate features needed for comparison for healthCodes with demographic data
    # This involves merging demo, info, and average label data for healthCodes with demo data
    
    # Start with deduped demo data for healthCodes with demographic data
    comparison_features_df = deduped_demo[deduped_demo['healthCode'].isin(dataset_with_demo_hcs)].copy()
    
    # Add age from info_df
    info_df_copy = info_df.copy()
    info_df_copy['birthdate'] = pd.to_datetime(info_df_copy['birthdate'], errors='coerce', utc=True)
    info_df_copy['createdOn'] = pd.to_datetime(info_df_copy['createdOn'], errors='coerce', utc=True)
    info_df_copy['age'] = (info_df_copy['createdOn'] - info_df_copy['birthdate']).dt.days / 365.25
    info_df_copy['age'] = info_df_copy['age'].round().astype('Int64')
    comparison_features_df = comparison_features_df.merge(
        info_df_copy[['healthCode', 'age']], on='healthCode', how='left'
    )
    
    # Add Height and Weight (use original columns from demo)
    height_col = 'HeightCentimeters'
    weight_col = 'WeightKilograms'
    cols_to_merge = ['healthCode']
    if height_col in demographic_df.columns:
        cols_to_merge.append(height_col)
    else:
        print(f"Warning: Column '{height_col}' not found in demographic data. Skipping height comparison.")
        height_col = None # Set to None if not found
        
    if weight_col in demographic_df.columns:
        cols_to_merge.append(weight_col)
    else:
        print(f"Warning: Column '{weight_col}' not found in demographic data. Skipping weight comparison.")
        weight_col = None # Set to None if not found

    if len(cols_to_merge) > 1: # Only merge if height or weight columns exist
        comparison_features_df = comparison_features_df.merge(
            demographic_df[cols_to_merge], on='healthCode', how='left'
        )

    # Calculate average label values
    label_cols = [col for col in dataset_df.columns if col.endswith('_value')]
    avg_label_values = dataset_df[dataset_df['healthCode'].isin(dataset_with_demo_hcs)].groupby('healthCode')[label_cols].mean(numeric_only=True).reset_index()
    comparison_features_df = comparison_features_df.merge(avg_label_values, on='healthCode', how='left')

    # --- Binning for comparison ---
    # Age bins
    age_bins = [18, 25, 35, 45, 55, 65, 75, float('inf')]
    age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
    comparison_features_df['age_group'] = pd.cut(comparison_features_df['age'], bins=age_bins, labels=age_labels, right=False)
    comparison_features_df['age_group'] = comparison_features_df['age_group'].cat.add_categories('Unknown').fillna('Unknown')

    # Height bins (only if height_col exists)
    if height_col and height_col in comparison_features_df.columns:
        height_bins = [0, 150, 160, 170, 180, 190, 200, float('inf')]
        height_labels = ['<150cm', '150-160cm', '160-170cm', '170-180cm', '180-190cm', '190-200cm', '>200cm']
        comparison_features_df['height_group'] = pd.cut(comparison_features_df[height_col], bins=height_bins, labels=height_labels, right=False)
        comparison_features_df['height_group'] = comparison_features_df['height_group'].cat.add_categories('Unknown').fillna('Unknown')
    else:
        comparison_features_df['height_group'] = 'Not Available'

    # Weight bins (only if weight_col exists)
    if weight_col and weight_col in comparison_features_df.columns:
        weight_bins = [0, 50, 60, 70, 80, 90, 100, float('inf')]
        weight_labels = ['<50kg', '50-60kg', '60-70kg', '70-80kg', '80-90kg', '90-100kg', '>100kg']
        comparison_features_df['weight_group'] = pd.cut(comparison_features_df[weight_col], bins=weight_bins, labels=weight_labels, right=False)
        comparison_features_df['weight_group'] = comparison_features_df['weight_group'].cat.add_categories('Unknown').fillna('Unknown')
    else:
        comparison_features_df['weight_group'] = 'Not Available'
    
    # Fill missing BiologicalSex
    comparison_features_df['BiologicalSex'] = comparison_features_df['BiologicalSex'].fillna('Unknown')

    # Separate features for train and test sets
    train_features_df = comparison_features_df[comparison_features_df['healthCode'].isin(train_hc)]
    test_features_df = comparison_features_df[comparison_features_df['healthCode'].isin(test_hc)]

    # --- Function to compare categorical/binned distributions ---
    def compare_categorical_distribution(feature_name):
        print(f"\n--- Distribution for: {feature_name} ---")
        original_dist = comparison_features_df[feature_name].value_counts(normalize=True).sort_index()
        train_dist = train_features_df[feature_name].value_counts(normalize=True).sort_index()
        test_dist = test_features_df[feature_name].value_counts(normalize=True).sort_index()
        
        comparison_df = pd.DataFrame({
            'Original (%)': original_dist * 100,
            'Train (%)': train_dist * 100,
            'Test (%)': test_dist * 100
        }).fillna(0).round(2)
        print(comparison_df)

    # --- Function to compare continuous distributions ---
    def compare_continuous_distribution(feature_name):
         print(f"\n--- Distribution for: {feature_name} ---")
         original_desc = comparison_features_df[feature_name].describe()
         train_desc = train_features_df[feature_name].describe()
         test_desc = test_features_df[feature_name].describe()

         comparison_df = pd.DataFrame({
             'Original': original_desc,
             'Train': train_desc,
             'Test': test_desc
         }).round(3)
         # Add non-NaN count for clarity
         comparison_df.loc['non_nan_count'] = [
             comparison_features_df[feature_name].notna().sum(),
             train_features_df[feature_name].notna().sum(),
             test_features_df[feature_name].notna().sum()
         ]
         print(comparison_df)

    # Compare demographic distributions
    compare_categorical_distribution('age_group')
    compare_categorical_distribution('BiologicalSex')
    if height_col: # Only compare if column existed
        compare_categorical_distribution('height_group')
    if weight_col: # Only compare if column existed
        compare_categorical_distribution('weight_group')

    # Compare average label distributions (continuous)
    print("\n--- Comparing Average Label Value Distributions ---")
    for label_col in label_cols:
        # Only compare if there are non-NaN values
        if comparison_features_df[label_col].notna().any():
             compare_continuous_distribution(label_col)
        else:
             print(f"\n--- Distribution for: {label_col} ---")
             print("Skipping comparison (all values are NaN).")


    # Note: Stratification check based on original features is less direct here,
    # as stratification happens on derived cluster labels. 
    # The primary checks are overlap, completeness, and size.

    # Print the final length of train, test sets and input dataset
    print("\n--- Dataset Size Comparison ---")
    print(f"Input dataset size: {len(dataset_df)} rows, {len(set(dataset_df['healthCode'].unique()))} unique healthCodes")
    print(f"Train set size: {len(train_df)} rows, {len(set(train_df['healthCode'].unique()))} unique healthCodes")
    print(f"Test set size: {len(test_df)} rows, {len(set(test_df['healthCode'].unique()))} unique healthCodes")
    
    # Calculate and print percentages
    train_pct = len(train_df) / len(dataset_df) * 100
    test_pct = len(test_df) / len(dataset_df) * 100
    print(f"Train set: {train_pct:.2f}% of input dataset")
    print(f"Test set: {test_pct:.2f}% of input dataset")
    print("\nCluster split real data integration test completed successfully.")

@pytest.mark.skipif(not real_data_exists, reason="Real data files not found at specified paths")
def test_advanced_split_real_data():
    """Integration test for advanced split using real data files."""
    print(f"\nRunning advanced split integration test with real data:")
    print(f"  Demographics: {REAL_DEMO_PATH}")
    print(f"  Info:         {REAL_INFO_PATH}")
    print(f"  Dataset:      {REAL_DATASET_PATH}")
    
    # Load real data
    try:
        demographic_df = pd.read_parquet(REAL_DEMO_PATH)
        info_df = pd.read_parquet(REAL_INFO_PATH)
        dataset_df = pd.read_parquet(REAL_DATASET_PATH)
        print("Successfully loaded real data files.")
    except Exception as e:
        pytest.fail(f"Failed to load real data files: {e}")
        
    test_size = 0.2
    random_state = 789  # Use a different random state
    
    # Run the advanced split function
    try:
        train_df, test_df = stratified_split_advanced(
            dataset_df=dataset_df,
            demographic_df=demographic_df,
            info_df=info_df,
            test_size=test_size,
            random_state=random_state
        )
        print(f"Successfully ran stratified_split_advanced on real data.")
    except Exception as e:
         pytest.fail(f"stratified_split_advanced failed with real data: {e}")

    # --- Assertions --- 
    
    # 1. Check for overlap
    train_hc = set(train_df['healthCode'].unique())
    test_hc = set(test_df['healthCode'].unique())
    assert len(train_hc.intersection(test_hc)) == 0, "Overlap detected between train/test healthCodes"
    print("Assertion Passed: No healthCode overlap.")

    # 2. Check if all dataset healthCodes are present
    all_dataset_hcs = set(dataset_df['healthCode'].unique())
    split_hc_set = train_hc | test_hc
    
    # Check for missing healthCodes
    missing_from_split = all_dataset_hcs - split_hc_set
    assert len(missing_from_split) == 0, f"{len(missing_from_split)} healthCodes from dataset are missing from the split"
    
    # Check for extra healthCodes (shouldn't be any)
    extra_in_split = split_hc_set - all_dataset_hcs
    assert len(extra_in_split) == 0, f"{len(extra_in_split)} healthCodes in split are not from the dataset"
    
    print(f"Assertion Passed: All {len(all_dataset_hcs)} dataset healthCodes are included in the split (train: {len(train_hc)}, test: {len(test_hc)}).")
    
    # 3. Check split shapes (approximate)
    expected_test_hc = int(round(len(all_dataset_hcs) * test_size))
    expected_train_hc = len(all_dataset_hcs) - expected_test_hc
    
    # Allow a small tolerance due to stratification constraints
    tolerance = max(2, int(len(all_dataset_hcs) * 0.02)) # Tolerance of 2 or 2%, whichever is larger
    assert abs(len(train_hc) - expected_train_hc) <= tolerance, f"Train set size deviates too much ({len(train_hc)} vs {expected_train_hc}, tolerance {tolerance})"
    assert abs(len(test_hc) - expected_test_hc) <= tolerance, f"Test set size deviates too much ({len(test_hc)} vs {expected_test_hc}, tolerance {tolerance})"
    print(f"Assertion Passed: Split shapes are approximately correct (Train: {len(train_hc)}, Test: {len(test_hc)}).")

    # --- 4. Calculate Feature Distributions for Comparison ---
    print("\n--- Comparing Feature Distributions ---")
    
    # Process info data to get age
    info_df_copy = info_df.copy()
    info_df_copy['birthdate'] = pd.to_datetime(info_df_copy['birthdate'], errors='coerce', utc=True)
    info_df_copy['createdOn'] = pd.to_datetime(info_df_copy['createdOn'], errors='coerce', utc=True)
    info_df_copy['age'] = (info_df_copy['createdOn'] - info_df_copy['birthdate']).dt.days / 365.25
    info_df_copy['age'] = info_df_copy['age'].round().astype('Int64')
    
    # Age bins for comparison
    age_bins = [18, 25, 35, 45, 55, 65, 75, float('inf')]
    age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
    info_df_copy['age_group'] = pd.cut(info_df_copy['age'], bins=age_bins, labels=age_labels, right=False)
    info_df_copy['age_group'] = info_df_copy['age_group'].astype(str)
    info_df_copy.loc[info_df_copy['age_group'] == 'nan', 'age_group'] = 'Unknown_Age'
    
    # Get age data for each healthCode
    hc_age = info_df_copy[['healthCode', 'age', 'age_group']].drop_duplicates('healthCode')
    
    # Calculate average label values
    label_cols = [col for col in dataset_df.columns if col.endswith('_value')]
    print(f"Found {len(label_cols)} label columns")
    
    # Group by healthCode to get mean label values
    avg_labels = dataset_df.groupby('healthCode')[label_cols].mean().reset_index()
    
    # Merge age data with average labels
    comparison_df = hc_age.merge(avg_labels, on='healthCode', how='outer')
    
    # Add demographic data
    demo_cols = ['healthCode', 'BiologicalSex', 'WeightKilograms', 'HeightCentimeters']
    demo_data = demographic_df[demo_cols].drop_duplicates('healthCode')
    comparison_df = comparison_df.merge(demo_data, on='healthCode', how='outer')

    # Split into train/test for comparison
    train_features = comparison_df[comparison_df['healthCode'].isin(train_hc)]
    test_features = comparison_df[comparison_df['healthCode'].isin(test_hc)]

    # --- Functions to compare distributions ---
    def compare_categorical_distribution(feature_name):
        print(f"\n--- Distribution for: {feature_name} ---")
        try:
            original_dist = comparison_df[feature_name].value_counts(normalize=True).sort_index()
            train_dist = train_features[feature_name].value_counts(normalize=True).sort_index()
            test_dist = test_features[feature_name].value_counts(normalize=True).sort_index()
            
            comparison_table = pd.DataFrame({
                'Original (%)': original_dist * 100,
                'Train (%)': train_dist * 100,
                'Test (%)': test_dist * 100
            }).fillna(0).round(2)
            print(comparison_table)
        except Exception as e:
            print(f"Error comparing distributions for {feature_name}: {e}")

    def compare_continuous_distribution(feature_name):
        print(f"\n--- Distribution for: {feature_name} ---")
        try:
            original_desc = comparison_df[feature_name].describe([0.25, 0.5, 0.75])
            train_desc = train_features[feature_name].describe([0.25, 0.5, 0.75])
            test_desc = test_features[feature_name].describe([0.25, 0.5, 0.75])

            comparison_table = pd.DataFrame({
                'Original': original_desc,
                'Train': train_desc,
                'Test': test_desc
            }).round(3)
            # Add non-NaN count for clarity
            comparison_table.loc['non_nan_count'] = [
                comparison_df[feature_name].notna().sum(),
                train_features[feature_name].notna().sum(),
                test_features[feature_name].notna().sum()
            ]
            print(comparison_table)
        except Exception as e:
            print(f"Error comparing distributions for {feature_name}: {e}")

    # Compare demographic distributions
    print("\n=== Demographic Distribution Comparison ===")
    compare_categorical_distribution('age_group')
    compare_categorical_distribution('BiologicalSex')
    compare_continuous_distribution('WeightKilograms')
    compare_continuous_distribution('HeightCentimeters')
    
    # Compare label distributions
    print("\n=== Label Distribution Comparison ===")
    for label in label_cols:
        if comparison_df[label].notna().any():
            compare_continuous_distribution(label)
        else:
            print(f"\n{label}: All values are NaN, skipping comparison")

    print("\nAdvanced split real data integration test completed successfully.")

    



