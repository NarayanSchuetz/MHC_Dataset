import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import argparse
import os
from pathlib import Path

def stratify_and_split_by_nan_labels(df, test_size=0.2, num_bins=5, random_state=42):
    """
    Performs a train-test split stratified by the number of non-NaN label types per healthCode.

    Args:
        df (pd.DataFrame): The input dataframe with 'healthCode' and label columns ending in '_value', as produced by the `labelled_dataset.py` script.
        test_size (float): The proportion of the dataset to include in the test split.
        num_bins (int): The number of bins to create for stratification based on label counts.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: A tuple containing the train DataFrame and the test DataFrame (train_df, test_df).
    """
    print("Starting stratification and split...")

    # 1. Calculate the number of distinct non-NaN label types per healthCode
    label_value_cols = [col for col in df.columns if col.endswith('_value')]
    if not label_value_cols:
        raise ValueError("No label columns found ending with '_value'. Cannot stratify.")

    # Create a boolean dataframe indicating where labels are non-NaN
    label_present_df = df[['healthCode'] + label_value_cols].copy()
    for col in label_value_cols:
        label_present_df[col] = label_present_df[col].notna()

    # Group by healthCode and count how many label *types* have at least one True value
    non_nan_counts = label_present_df.groupby('healthCode')[label_value_cols].any().sum(axis=1)
    non_nan_counts = non_nan_counts.reset_index(name='non_nan_label_count')

    print("Distribution of non-NaN label counts per healthCode:")
    print(non_nan_counts['non_nan_label_count'].value_counts().sort_index())

    # 2. Bin the counts
    # Using qcut ensures roughly equal numbers of healthCodes per bin
    # Handle cases where there might be too few unique counts for the requested number of bins
    try:
        non_nan_counts['count_bin'] = pd.qcut(non_nan_counts['non_nan_label_count'], q=num_bins, labels=False, duplicates='drop')
        actual_bins = non_nan_counts['count_bin'].nunique()
        if actual_bins < num_bins:
             print(f"\nWarning: Could only create {actual_bins} bins due to data distribution.")
        else:
             print(f"\nBinned counts into {actual_bins} bins.")
    except ValueError:
        # If qcut fails (e.g., all counts are the same), just use the raw count for stratification
        print(f"\nWarning: Could not create {num_bins} bins for counts. Using raw counts for stratification.")
        non_nan_counts['count_bin'] = non_nan_counts['non_nan_label_count']
        actual_bins = non_nan_counts['count_bin'].nunique()


    print("\nDistribution of binned counts:")
    print(non_nan_counts['count_bin'].value_counts(normalize=True).sort_index())

    # 3. Perform stratified split on the healthCodes based on the bins
    # Ensure we only split healthCodes that were part of the count calculation
    valid_healthcodes = non_nan_counts['healthCode']
    stratify_labels = non_nan_counts['count_bin']

    # Handle cases where stratification might not be possible (e.g., only one bin)
    if actual_bins <= 1:
        print("\nWarning: Only one stratification bin. Performing regular group split instead.")
        unique_hc = valid_healthcodes.unique()
        train_hc, test_hc = train_test_split(
            unique_hc,
            test_size=test_size,
            random_state=random_state
        )
    else:
         train_hc, test_hc = train_test_split(
            valid_healthcodes,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=random_state
        )


    print(f"\nSplit into {len(train_hc)} train healthCodes and {len(test_hc)} test healthCodes.")

    # 4. Create final train/test DataFrames
    train_df = df[df['healthCode'].isin(train_hc)].copy()
    test_df = df[df['healthCode'].isin(test_hc)].copy()

    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Test DataFrame shape: {test_df.shape}")

    # Verification
    common_hc = set(train_df['healthCode'].unique()) & set(test_df['healthCode'].unique())
    print(f"Common healthCodes between train/test: {len(common_hc)}")
    if len(common_hc) > 0:
        print("WARNING: Overlap detected in healthCodes between train and test sets!")

    print("Stratification and split complete.")
    return train_df, test_df


def stratified_split_advanced(dataset_df, demographic_df, info_df, test_size=0.2, random_state=42, sharing_subset="all-researchers"):
    """
    Performs a stratified split on the health data by healthCode, stratifying by age and label values.
    After splitting, adds diagnostic information about demographics and label distributions.

    Args:
        dataset_df (pd.DataFrame): The input dataframe with 'healthCode' and label columns ending in '_value'.
        demographic_df (pd.DataFrame): The demographic dataframe with 'healthCode' and demographic columns.
        info_df (pd.DataFrame): The info dataframe with 'healthCode' and info columns.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.
        sharing_subset (str): The subset of participants to include in the split.

    Returns:
        tuple: (train_df, test_df) DataFrames with the split data.
    """
    print("Starting advanced stratified split...")
    
    # Process info data to get age
    info_df = info_df.copy()
    # Convert birthdate and createdOn to datetime
    info_df['birthdate'] = pd.to_datetime(info_df['birthdate'], errors='coerce', utc=True)
    info_df['createdOn'] = pd.to_datetime(info_df['createdOn'], errors='coerce', utc=True)
    # Calculate age at the time the record was created
    info_df['age'] = (info_df['createdOn'] - info_df['birthdate']).dt.days / 365.25
    info_df['age'] = info_df['age'].round().astype('Int64')  # Round to nearest integer and use nullable integer type
    
    # Bin age for stratification
    age_bins = [18, 25, 35, 45, 55, 65, float('inf')]
    age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    info_df['age_group'] = pd.cut(info_df['age'], bins=age_bins, labels=age_labels, right=False)
    info_df['age_group'] = info_df['age_group'].astype(str)
    info_df.loc[info_df['age_group'] == 'nan', 'age_group'] = 'Unknown_Age'
    
    # Get unique healthCodes from dataset and merge with age info
    dataset_hcs = pd.DataFrame({'healthCode': dataset_df['healthCode'].unique()})
    hc_age = dataset_hcs.merge(
        info_df[['healthCode', 'age_group']],
        on='healthCode',
        how='left'
    )
    hc_age['age_group'] = hc_age['age_group'].fillna('Unknown_Age')
    
    # Print age distribution
    print("\nAge Group Distribution:")
    print(len(hc_age))
    print(len(dataset_hcs))
    print(hc_age['age_group'].value_counts())
    
    # Find all label columns in the dataset
    label_cols = [col for col in dataset_df.columns if col.endswith('_value')]
    print(f"\nFound {len(label_cols)} label columns: {label_cols}")
    
    # For each healthCode, calculate average value for each label and round to integer
    label_values = []
    for hc in dataset_df['healthCode'].unique():
        hc_data = dataset_df[dataset_df['healthCode'] == hc]
        
        # For each label, calculate the average value (or NaN if not available)
        avg_values = {'healthCode': hc}
        for col in label_cols:
            # Calculate mean, ignoring NaN values, and round to nearest integer
            if hc_data[col].notna().any():
                avg_value = round(hc_data[col].mean(skipna=True))
                
                # For happiness-related labels, bin values below 5 into a single bin
                if "happiness" in col.lower() and avg_value < 5:
                    avg_value = 4  # Use 4 to represent all values below 5
                
                # For feel_worthwhile1_value, bin values below 4 into a single bin
                if col == "feel_worthwhile1_value" and avg_value < 4:
                    avg_value = 3  # Use 3 to represent all values below 4
                
                # For feel_worthwhile2_value, bin values below 4 into a single bin
                if col == "feel_worthwhile2_value" and avg_value < 4:
                    avg_value = 3  # Use 3 to represent all values below 4
                
                # For feel_worthwhile3_value, bin values 9 and 10 together
                if col == "feel_worthwhile3_value" and avg_value == 10:
                    avg_value = 9  # Use 9 to represent both 9 and 10
                
                # Heart disease handling - keep 10 as is, everything else as "other"
                if "heart_disease" in col.lower() and avg_value != 10:
                    avg_value = 0  # Use 0 to represent "Other Heart Disease"
                
                avg_values[col] = avg_value
            else:
                avg_values[col] = np.nan
        
        label_values.append(avg_values)
    
    # Convert to DataFrame
    label_df = pd.DataFrame(label_values)
    
    # Convert label values to strings with clear descriptive labels for stratification
    label_str_cols = []
    for col in label_cols:
        str_col = f'{col}_str'
        label_str_cols.append(str_col)
        
        # Create a mapping function based on the column
        def map_value_to_label(val):
            if pd.isna(val):
                return f'Missing'
                
            val_int = int(val)  # Convert to integer
            
            # Specific mapping for happiness-related columns
            if "happiness" in col.lower():
                if val_int == 4:
                    return f'≤4 (Low)'
                else:
                    return f'{val_int} (High)'
                    
            # Specific mapping for feel_worthwhile1
            elif col == "feel_worthwhile1_value":
                if val_int == 3:
                    return f'≤3 (Low)'
                else:
                    return f'{val_int} (High)'
            
            # Specific mapping for feel_worthwhile2
            elif col == "feel_worthwhile2_value":
                if val_int == 3:
                    return f'≤3 (Low)'
                else:
                    return f'{val_int} (High)'
            
            # Specific mapping for feel_worthwhile3
            elif col == "feel_worthwhile3_value":
                if val_int == 9:
                    return f'9-10 (High)'
                else:
                    return f'{val_int}'
            
            # Heart disease handling
            elif "heart_disease" in col.lower():
                if val_int == 10:
                    return "10"
                elif val_int == 0:  # Our marker for "Other Heart Disease"
                    return "Other Heart Disease"
                else:
                    return "Other Heart Disease"  # Redundant but for clarity
            
            # Default mapping for other columns
            else:
                return f'{val_int}'
        
        # Apply the mapping function
        label_df[str_col] = label_df[col].apply(map_value_to_label)
        
        # Handle missing values
        nan_representations = ['nan', 'NaN', '<NA>', 'None', 'NA', 'NaT']
        label_df.loc[label_df[str_col].isin(nan_representations), str_col] = 'Missing'
    
    # Display the distribution of string values
    for col in label_str_cols:
        base_col = col.replace('_str', '')
        col_name = base_col.replace('_value', '').replace('_', ' ').title()
        print(f"\n{col_name} Distribution:")
        value_counts = label_df[col].value_counts()
        print(value_counts)
        
        # Check for rare classes (less than 2 members)
        rare_classes = value_counts[value_counts < 2].index.tolist()
        if rare_classes:
            print(f"Warning: '{col_name}' has rare classes with <2 members: {rare_classes}")
    
    # Merge age data with string label values
    stratification_df = hc_age.merge(
        label_df[['healthCode'] + label_str_cols],
        on='healthCode',
        how='inner'
    )
    
    # Get features for stratification
    strat_features = [] + label_str_cols
    
    print(f"\nUsing features for stratification: {strat_features}")
    
    # Check for rare combinations in stratification
    strat_combo_counts = stratification_df.groupby(strat_features).size()
    rare_combos = strat_combo_counts[strat_combo_counts < 2]
    
    if not rare_combos.empty:
        print("\nERROR: Cannot perform stratified split.")
        print(f"Found {len(rare_combos)} class combinations with only 1 member.")
        print("Detailed information about rare combinations:")
        
        # Show the top N rare combinations (limiting output to avoid overwhelming)
        max_to_show = min(10, len(rare_combos))
        print(f"\nShowing {max_to_show} of {len(rare_combos)} rare combinations:")
        
        # Convert the MultiIndex to a DataFrame for better display
        rare_combos_df = rare_combos.reset_index()
        # Add the count column
        rare_combos_df['count'] = rare_combos.values
        
        # Display the rare combinations with their counts
        for i in range(max_to_show):
            print("\nCombination #{}:".format(i+1))
            for col in strat_features:
                col_name = col.replace('_str', '').replace('_value', '').replace('_', ' ').title()
                print(f"  {col_name}: {rare_combos_df.iloc[i][col]}")
            print(f"  Count: {rare_combos_df.iloc[i]['count']}")
        
        # Suggestion for potential solutions
        print("\nPossible solutions:")
        print("1. Remove some stratification features")
        print("2. Further bin/combine rare categories")
        print("3. Use a different stratification approach like clustering")
        
        # Raise an error to stop execution
        raise ValueError("Cannot perform stratified split due to rare class combinations")
        
    # Perform stratified split using all stratification features directly
    try:
        train_hc, test_hc = train_test_split(
            stratification_df['healthCode'],
            test_size=test_size,
            stratify=stratification_df[strat_features],
            random_state=random_state
        )
        print(f"\nSuccessfully split using stratification with {len(strat_features)} separate features")
    except ValueError as e:
        print(f"\nERROR: Failed to perform stratified split: {str(e)}")
        
        # Additional diagnostics for unexpected errors
        print("\nAdditional diagnostics:")
        for col in strat_features:
            print(f"Column '{col}' has {stratification_df[col].nunique()} unique values")
            unique_values_counts = stratification_df[col].value_counts()
            single_member_values = unique_values_counts[unique_values_counts == 1]
            if not single_member_values.empty:
                print(f"  Values with only one occurrence: {list(single_member_values.index)}")
        
        # Raise the error to fail the test
        raise
    
    print(f"\nSplit into {len(train_hc)} train and {len(test_hc)} test healthCodes")
    
    # Create the final train and test datasets
    train_df = dataset_df[dataset_df['healthCode'].isin(train_hc)].copy()
    test_df = dataset_df[dataset_df['healthCode'].isin(test_hc)].copy()
    
    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Test DataFrame shape: {test_df.shape}")
    
    # Verify no overlap between train and test
    common_hc = set(train_df['healthCode'].unique()) & set(test_df['healthCode'].unique())
    print(f"Common healthCodes between train/test: {len(common_hc)}")
    if len(common_hc) > 0:
        print("WARNING: Overlap detected in healthCodes between train and test sets!")
    
    # Add diagnostic information
    print("\n=== Diagnostic Information ===")
    
    # Prepare demographic data for diagnostics
    demo_cols = ['healthCode', 'BiologicalSex', 'WeightKilograms', 'HeightCentimeters']
    demo_data = demographic_df[demo_cols].drop_duplicates('healthCode')
    
    # Prepare age data for diagnostics
    age_data = info_df[['healthCode', 'age']].drop_duplicates('healthCode')
    
    # Combine all diagnostic data
    diag_data = demo_data.merge(age_data, on='healthCode', how='outer')
    
    # Calculate statistics for train set
    train_diag = diag_data[diag_data['healthCode'].isin(train_hc)]
    print("\nTrain Set Statistics:")
    print("Age (years):")
    print(train_diag['age'].describe(percentiles=[0.25, 0.5, 0.75]))
    print("\nWeight (kg):")
    print(train_diag['WeightKilograms'].describe(percentiles=[0.25, 0.5, 0.75]))
    print("\nHeight (cm):")
    print(train_diag['HeightCentimeters'].describe(percentiles=[0.25, 0.5, 0.75]))
    print("\nSex Distribution:")
    print(train_diag['BiologicalSex'].value_counts())
    
    # Calculate statistics for test set
    test_diag = diag_data[diag_data['healthCode'].isin(test_hc)]
    print("\nTest Set Statistics:")
    print("Age (years):")
    print(test_diag['age'].describe(percentiles=[0.25, 0.5, 0.75]))
    print("\nWeight (kg):")
    print(test_diag['WeightKilograms'].describe(percentiles=[0.25, 0.5, 0.75]))
    print("\nHeight (cm):")
    print(test_diag['HeightCentimeters'].describe(percentiles=[0.25, 0.5, 0.75]))
    print("\nSex Distribution:")
    print(test_diag['BiologicalSex'].value_counts())
    
    # Calculate label statistics for train set
    print("\nTrain Set Label Statistics:")
    for col in label_cols:
        col_name = col.replace('_value', '').replace('_', ' ').title()
        print(f"\n{col_name}:")
        print(train_df[col].describe(percentiles=[0.25, 0.5, 0.75]))
    
    # Calculate label statistics for test set
    print("\nTest Set Label Statistics:")
    for col in label_cols:
        col_name = col.replace('_value', '').replace('_', ' ').title()
        print(f"\n{col_name}:")
        print(test_df[col].describe(percentiles=[0.25, 0.5, 0.75]))
    
    return train_df, test_df


def stratified_split_cluster(dataset_df, demographic_df, info_df, test_size=0.2, n_clusters=10, random_state=42):
    """
    Performs a stratified split on the health data by healthCode, using clustering
    on demographic features and average label values for stratification.

    Args:
        dataset_df (pd.DataFrame): Input data with 'healthCode' and label columns ('_value').
        demographic_df (pd.DataFrame): Demographic data with 'healthCode'.
        info_df (pd.DataFrame): Info data with 'healthCode', 'birthdate', 'createdOn'.
        test_size (float): Proportion for the test split.
        n_clusters (int): Number of clusters for K-Means.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: (train_df, test_df)
    """
    print("Starting stratified split using clustering...")

    # --- 1. Data Preparation (Similar to advanced split) ---
    # Deduplicate demographic data
    print("Deduplicating demographic data...")
    demographic_df['nan_count'] = demographic_df.isna().sum(axis=1)
    demographic_df = demographic_df.sort_values('nan_count').drop_duplicates(subset=['healthCode'], keep='first')
    demographic_df = demographic_df.drop(columns=['nan_count'])
    print(f"Deduplicated demographic data: {len(demographic_df)} healthCodes")

    # Identify all unique healthCodes in the dataset
    all_dataset_hcs = set(dataset_df['healthCode'].unique())
    all_demo_hcs = set(demographic_df['healthCode'].unique())
    
    # Identify healthCodes missing demographic data
    missing_demo_hcs = all_dataset_hcs - all_demo_hcs
    print(f"Found {len(missing_demo_hcs)} healthCodes in dataset that lack demographic data")
    
    # Continue with normal processing for healthCodes with demographic data
    has_demo_hcs = all_dataset_hcs & all_demo_hcs
    print(f"Processing {len(has_demo_hcs)} healthCodes with demographic data for clustering")
    
    # Filter dataset to only include healthCodes with demographic data for clustering
    dataset_with_demo = dataset_df[dataset_df['healthCode'].isin(has_demo_hcs)].copy()

    # Calculate age
    info_df = info_df.copy()
    info_df['birthdate'] = pd.to_datetime(info_df['birthdate'], errors='coerce', utc=True)
    info_df['createdOn'] = pd.to_datetime(info_df['createdOn'], errors='coerce', utc=True)
    info_df['age'] = (info_df['createdOn'] - info_df['birthdate']).dt.days / 365.25
    info_df['age'] = info_df['age'].round().astype('Int64')

    # Merge demographic and info
    merged_demo_info = demographic_df.merge(
        info_df[['healthCode', 'age']],
        on="healthCode",
        how="left"
    )[['healthCode', 'BiologicalSex', 'age']]
    print(f"Merged demo/info data: {len(merged_demo_info)} healthCodes")

    # Calculate average label values for healthCodes with demographic data
    label_cols = [col for col in dataset_df.columns if col.endswith('_value')]
    print(f"Calculating average values for {len(label_cols)} labels...")
    avg_label_values = dataset_with_demo.groupby('healthCode')[label_cols].mean(numeric_only=True).reset_index()
    print(f"Average label values calculated for {len(avg_label_values)} healthCodes")

    # Combine all features per healthCode
    features_df = merged_demo_info.merge(avg_label_values, on='healthCode', how='inner')
    print(f"Combined feature matrix shape for clustering: {features_df.shape}")

    # Store healthCodes for later use
    healthCodes = features_df['healthCode'].copy()
    features_for_clustering = features_df.drop(columns=['healthCode'])

    # --- 2. Preprocessing for Clustering ---
    print("Preprocessing features for clustering...")
    numerical_features = ['age'] + label_cols
    categorical_features = ['BiologicalSex']

    # Identify columns that actually exist in the dataframe
    existing_numerical = [f for f in numerical_features if f in features_for_clustering.columns]
    existing_categorical = [f for f in categorical_features if f in features_for_clustering.columns]

    # --- Fill missing categorical values BEFORE encoding ---
    for col in existing_categorical:
        if features_for_clustering[col].isnull().any():
            print(f"Filling missing values in categorical column '{col}' with 'Missing'")
            features_for_clustering[col] = features_for_clustering[col].fillna('Missing')
            # Ensure the column is treated as object/string type after filling
            features_for_clustering[col] = features_for_clustering[col].astype(object)

    # Create preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Impute missing numerical with median
        ('scaler', StandardScaler()) # Scale numerical features
    ])

    categorical_pipeline = Pipeline([
        # ('imputer', SimpleImputer(strategy='most_frequent')), # No longer needed
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # One-hot encode sex (including 'Missing')
    ])

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, existing_numerical),
            ('cat', categorical_pipeline, existing_categorical)
        ],
        remainder='passthrough'
    )

    # Apply preprocessing
    processed_features = preprocessor.fit_transform(features_for_clustering)
    print(f"Processed feature matrix shape: {processed_features.shape}")

    # Handle potential all-NaN columns after imputation (if a column was all NaNs initially)
    # SimpleImputer might leave NaNs if fit on all-NaN data; replace these with 0
    processed_features = np.nan_to_num(processed_features)

    # --- 3. Clustering ---
    print(f"Applying K-Means clustering with {n_clusters} clusters...")
    if processed_features.shape[0] < n_clusters:
        print(f"Warning: Number of samples ({processed_features.shape[0]}) is less than n_clusters ({n_clusters}). Reducing n_clusters.")
        n_clusters = processed_features.shape[0]

    if n_clusters <= 1:
        print("Warning: Not enough samples or clusters to perform meaningful clustering. Assigning all to cluster 0.")
        cluster_labels = np.zeros(processed_features.shape[0], dtype=int)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        try:
            cluster_labels = kmeans.fit_predict(processed_features)
            print("Clustering complete.")
        except Exception as e:
             print(f"Error during clustering: {e}. Assigning all to cluster 0.")
             cluster_labels = np.zeros(processed_features.shape[0], dtype=int)

    # Create a DataFrame for stratification
    stratify_df = pd.DataFrame({'healthCode': healthCodes, 'cluster': cluster_labels})

    # --- 4. Stratified Split ---
    print("Performing stratified split based on cluster labels...")

    # Check if stratification is possible
    unique_clusters = stratify_df['cluster'].nunique()
    cluster_counts = stratify_df['cluster'].value_counts()
    min_samples_per_cluster = cluster_counts.min()

    if unique_clusters <= 1 or min_samples_per_cluster < 2:
        print("Warning: Cannot stratify by cluster (too few clusters or samples per cluster). Performing regular split on healthCodes.")
        train_hc, test_hc = train_test_split(
            healthCodes,
            test_size=test_size,
            random_state=random_state
        )
    else:
        train_hc, test_hc = train_test_split(
            stratify_df['healthCode'],
            test_size=test_size,
            stratify=stratify_df['cluster'],
            random_state=random_state
        )

    print(f"Split into {len(train_hc)} train and {len(test_hc)} test healthCodes.")

    # --- 5. Create Final DataFrames ---
    if missing_demo_hcs:
        # Handle healthCodes missing demographic data separately
        print(f"Splitting {len(missing_demo_hcs)} healthCodes without demographic data...")
        
        # Convert to list for random selection
        missing_demo_hcs_list = list(missing_demo_hcs)
        
        # Randomly split these healthCodes with the same test_size
        train_missing_hc, test_missing_hc = train_test_split(
            missing_demo_hcs_list,
            test_size=test_size,
            random_state=random_state
        )
        
        print(f"Split into {len(train_missing_hc)} train and {len(test_missing_hc)} test healthCodes without demographic data")
        
        # Combine with healthCodes from clustering split
        train_hc = list(train_hc) + train_missing_hc
        test_hc = list(test_hc) + test_missing_hc
        
        print(f"Final split (including all healthCodes): {len(train_hc)} train, {len(test_hc)} test")

    # Create final train and test DataFrames from all included healthCodes
    train_df = dataset_df[dataset_df['healthCode'].isin(train_hc)].copy()
    test_df = dataset_df[dataset_df['healthCode'].isin(test_hc)].copy()

    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Test DataFrame shape: {test_df.shape}")

    # Verification
    common_hc = set(train_df['healthCode'].unique()) & set(test_df['healthCode'].unique())
    print(f"Common healthCodes between train/test: {len(common_hc)}")
    if len(common_hc) > 0:
        print("WARNING: Overlap detected in healthCodes between train and test sets!")
    
    # Verify all dataset healthCodes are assigned
    final_train_hcs = set(train_df['healthCode'].unique())
    final_test_hcs = set(test_df['healthCode'].unique()) 
    all_split_hcs = final_train_hcs | final_test_hcs
    if all_split_hcs != all_dataset_hcs:
        missing_hcs = all_dataset_hcs - all_split_hcs
        print(f"WARNING: {len(missing_hcs)} healthCodes from the dataset were not assigned to either train or test")

    print("Stratified split by clustering complete.")
    return train_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stratify and split health data by healthCode based on label completeness.')
    parser.add_argument('--input_parquet', required=True,
                        help='Path to the input denormalized Parquet file.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save the output train and test Parquet files.')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split (default: 0.2).')
    parser.add_argument('--num_bins', type=int, default=5,
                        help='Number of bins for stratifying label counts (default: 5).')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility (default: 42).')
    parser.add_argument('--demographic_parquet', 
                        help='Path to the demographic data Parquet file for advanced stratification.')
    parser.add_argument('--info_parquet', 
                        help='Path to the info data Parquet file for advanced stratification.')
    parser.add_argument('--use_advanced_split', action='store_true',
                        help='Use advanced stratification with demographic features.')

    args = parser.parse_args()

    # Expand user paths and create output directory
    input_path = Path(os.path.expanduser(args.input_parquet)).resolve()
    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    train_output_path = output_dir / "train_dataset.parquet"
    test_output_path = output_dir / "test_dataset.parquet"

    # --- Execution ---
    print(f"Loading data from: {input_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df_full = pd.read_parquet(input_path)
    print(f"Loaded dataframe shape: {df_full.shape}")

    # Determine which split method to use
    if args.use_advanced_split and args.demographic_parquet and args.info_parquet:
        # Load demographic and info data
        demographic_path = Path(os.path.expanduser(args.demographic_parquet)).resolve()
        info_path = Path(os.path.expanduser(args.info_parquet)).resolve()
        
        print(f"Loading demographic data from: {demographic_path}")
        if not demographic_path.is_file():
            raise FileNotFoundError(f"Demographic file not found: {demographic_path}")
        demographic_df = pd.read_parquet(demographic_path)
        
        print(f"Loading info data from: {info_path}")
        if not info_path.is_file():
            raise FileNotFoundError(f"Info file not found: {info_path}")
        info_df = pd.read_parquet(info_path)
        
        # Perform advanced stratified split
        print("Using advanced stratified split with demographic data...")
        train_df, test_df = stratified_split_advanced(
            dataset_df=df_full,
            demographic_df=demographic_df,
            info_df=info_df,
            test_size=args.test_size,
            random_state=args.random_state
        )
    else:
        # Use the basic stratified split
        if args.use_advanced_split:
            print("Warning: Advanced split requested but demographic or info data not provided.")
            print("Falling back to basic stratified split...")
        else:
            print("Using basic stratified split by label completeness...")
            
        train_df, test_df = stratify_and_split_by_nan_labels(
            df_full,
            test_size=args.test_size,
            num_bins=args.num_bins,
            random_state=args.random_state
        )

    # Save the results
    print(f"Saving train dataset to: {train_output_path}")
    train_df.to_parquet(train_output_path, index=False)

    print(f"Saving test dataset to: {test_output_path}")
    test_df.to_parquet(test_output_path, index=False)

    print("Script finished successfully.") 

    """Example usage:
    # Basic stratified split:
    python src/train_test_splitter.py \
    --input_parquet ~/Downloads/global_records.parquet \
    --output_dir ~/Downloads/ \
    --test_size 0.2 \
    --num_bins 5
    
    # Advanced stratified split with demographic data:
    python src/train_test_splitter.py \
    --input_parquet ~/Downloads/global_records.parquet \
    --demographic_parquet ~/Downloads/demographic_data.parquet \
    --info_parquet ~/Downloads/info_data.parquet \
    --output_dir ~/Downloads/ \
    --test_size 0.2 \
    --use_advanced_split
    """
