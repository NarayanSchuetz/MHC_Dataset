#!/bin/bash

# Configuration - modify these paths as needed
# Dataset generation parameters
SHERLOCK_DATASET_PATH="/scratch/users/schuetzn/data/mhc_dataset"
OUTPUT_PATH="/scratch/users/schuetzn/data/mhc_dataset_out"
OTHER_DATA_BASE_PATH="/home/users/schuetzn"

BATCH_SIZE=100
NUM_PROCESSES=4

# Data quality parameters
MIN_CHANNEL_COVERAGE=7.638888888888889  # 25th percentile of data coverage
MIN_CHANNELS_WITH_DATA=4
WINDOW_SIZE=7
MIN_REQUIRED_DAYS=5

# Label dataset parameters
LABEL_CSV="${OTHER_DATA_BASE_PATH}/combined_mhc_data.csv"
LABELLED_JSON_OUTPUT="${OUTPUT_PATH}/labelled_dataset.json"
LABELLED_PARQUET_OUTPUT="${OUTPUT_PATH}/labelled_dataset.parquet"

# Train-test split parameters
INPUT_PARQUET="${LABELLED_PARQUET_OUTPUT}"
SPLIT_OUTPUT_DIR="${OUTPUT_PATH}/splits"
TEST_SIZE=0.2
VALIDATION_SIZE=0.1
RANDOM_STATE=42
SPLIT_METHOD="cluster"  # Options: basic, advanced, cluster
N_CLUSTERS=8
# Set to "true" to only include participants who opted to share with all qualified researchers
SHARING_SUBSET="true"
# Only needed for advanced or cluster split methods
DEMOGRAPHIC_PARQUET="${OTHER_DATA_BASE_PATH}/mhc_demographics.parquet"
INFO_PARQUET="${OTHER_DATA_BASE_PATH}/mhc_full_participant_info.parquet"

# Print configuration
echo "Creating dataset with the following configuration:"
echo "- Dataset path: $SHERLOCK_DATASET_PATH"
echo "- Output path: $OUTPUT_PATH"
echo "- Batch size: $BATCH_SIZE"
echo "- Number of processes: $NUM_PROCESSES"
echo "- Min channel coverage: $MIN_CHANNEL_COVERAGE"
echo "- Min channels with data: $MIN_CHANNELS_WITH_DATA"
echo "- Window size: $WINDOW_SIZE"
echo "- Min required days: $MIN_REQUIRED_DAYS"
echo "- Test size: $TEST_SIZE"
echo "- Validation size: $VALIDATION_SIZE"
echo ""

# Make sure output directory exists
mkdir -p "$OUTPUT_PATH"

# Execute the Python script with the specified parameters
echo "Step 1: Generating dataset windows..."
python3 src/generate_windows.py \
  --sherlock_path "$SHERLOCK_DATASET_PATH" \
  --output_path "$OUTPUT_PATH" \
  --batch_size "$BATCH_SIZE" \
  --num_processes "$NUM_PROCESSES" \
  --min_channel_coverage "$MIN_CHANNEL_COVERAGE" \
  --min_channels_with_data "$MIN_CHANNELS_WITH_DATA" \
  --window_size "$WINDOW_SIZE" \
  --min_required_days "$MIN_REQUIRED_DAYS"

# Check if the script ran successfully
if [ $? -ne 0 ]; then
  echo "Error: Dataset window creation failed."
  exit 1
fi

echo "Step 2: Creating labelled dataset..."
python3 src/labelled_dataset.py \
  --interval_csv "${OUTPUT_PATH}/valid_7day_windows.csv" \
  --label_csv "$LABEL_CSV" \
  --json_output "$LABELLED_JSON_OUTPUT" \
  --parquet_output "$LABELLED_PARQUET_OUTPUT"

# Check if the script ran successfully
if [ $? -ne 0 ]; then
  echo "Error: Labelled dataset creation failed."
  exit 1
fi

echo "Step 3: Creating train-test splits..."
# Create splits directory
mkdir -p "$SPLIT_OUTPUT_DIR"

# Build the command based on the split method
SPLIT_CMD="python3 src/train_test_splitter.py \
  --input_parquet \"$INPUT_PARQUET\" \
  --output_dir \"$SPLIT_OUTPUT_DIR\" \
  --test_size $TEST_SIZE \
  --random_state $RANDOM_STATE \
  --split_method $SPLIT_METHOD"

# Add method-specific arguments
if [ "$SPLIT_METHOD" = "basic" ]; then
  SPLIT_CMD="$SPLIT_CMD --num_bins $NUM_BINS"
elif [ "$SPLIT_METHOD" = "advanced" ] || [ "$SPLIT_METHOD" = "cluster" ]; then
  SPLIT_CMD="$SPLIT_CMD --demographic_parquet \"$DEMOGRAPHIC_PARQUET\" --info_parquet \"$INFO_PARQUET\""
  
  if [ "$SPLIT_METHOD" = "cluster" ]; then
    SPLIT_CMD="$SPLIT_CMD --n_clusters $N_CLUSTERS"
  fi
fi

# Add sharing_subset flag if true
if [ "$SHARING_SUBSET" = "true" ]; then
  SPLIT_CMD="$SPLIT_CMD --sharing_subset"
fi

# Execute the split command
eval $SPLIT_CMD

# Check if the script ran successfully
if [ $? -ne 0 ]; then
  echo "Error: Train-test splitting failed."
  exit 1
fi

echo "Step 4: Creating train-validation splits..."
# Calculate adjusted validation size as a fraction of the train set
# If VALIDATION_SIZE=0.1 (10% of total) and TEST_SIZE=0.2 (20% of total),
# adjusted validation size should be 0.1/0.8 = 0.125 (12.5% of train set)
ADJUSTED_VAL_SIZE=$(echo "scale=4; $VALIDATION_SIZE / (1 - $TEST_SIZE)" | bc)

# Build the command for train-validation split
VAL_SPLIT_CMD="python3 src/train_test_splitter.py \
  --input_parquet \"$SPLIT_OUTPUT_DIR/train_dataset.parquet\" \
  --output_dir \"$SPLIT_OUTPUT_DIR/temp\" \
  --test_size $ADJUSTED_VAL_SIZE \
  --random_state $RANDOM_STATE \
  --split_method $SPLIT_METHOD"

# Add method-specific arguments
if [ "$SPLIT_METHOD" = "basic" ]; then
  VAL_SPLIT_CMD="$VAL_SPLIT_CMD --num_bins $NUM_BINS"
elif [ "$SPLIT_METHOD" = "advanced" ] || [ "$SPLIT_METHOD" = "cluster" ]; then
  VAL_SPLIT_CMD="$VAL_SPLIT_CMD --demographic_parquet \"$DEMOGRAPHIC_PARQUET\" --info_parquet \"$INFO_PARQUET\""
  
  if [ "$SPLIT_METHOD" = "cluster" ]; then
    VAL_SPLIT_CMD="$VAL_SPLIT_CMD --n_clusters $N_CLUSTERS"
  fi
fi

# Create temporary directory for train-val split
mkdir -p "$SPLIT_OUTPUT_DIR/temp"

# Execute the train-validation split command
eval $VAL_SPLIT_CMD

# Check if the script ran successfully
if [ $? -ne 0 ]; then
  echo "Error: Train-validation splitting failed."
  exit 1
fi

# Move the resulting files to their final locations
mv "$SPLIT_OUTPUT_DIR/temp/test_dataset.parquet" "$SPLIT_OUTPUT_DIR/validation_dataset.parquet"
mv "$SPLIT_OUTPUT_DIR/temp/train_dataset.parquet" "$SPLIT_OUTPUT_DIR/train_dataset.parquet"
# The original "$SPLIT_OUTPUT_DIR/train_dataset.parquet" (which was train+val combined) has now been overwritten by the final train set.
# The "$SPLIT_OUTPUT_DIR/test_dataset.parquet" (final test set from Step 3) remains untouched and correctly named.
rm -rf "$SPLIT_OUTPUT_DIR/temp"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
  echo "Dataset creation, labelling, and splitting completed successfully!"
  echo "Output files are available at: $SPLIT_OUTPUT_DIR"
  echo "The data was split into train, validation, and test sets with the following proportions:"
  test_pct=$(echo "scale=1; $TEST_SIZE * 100" | bc)
  val_pct=$(echo "scale=1; $VALIDATION_SIZE * 100" | bc)
  train_size=$(echo "scale=1; 1 - $TEST_SIZE - $VALIDATION_SIZE" | bc)
  train_pct=$(echo "scale=1; $train_size * 100" | bc)
  echo "- Test set: ${TEST_SIZE} (${test_pct}% of the total dataset)"
  echo "- Validation set: ${VALIDATION_SIZE} (${val_pct}% of the total dataset)"
  echo "- Train set: ${train_size} (${train_pct}% of the total dataset)"
  echo ""
  echo "Files:"
  echo "- test_dataset.parquet: Test set (${test_pct}% of data)"
  echo "- validation_dataset.parquet: Validation set (${val_pct}% of data)"
  echo "- train_dataset.parquet: Train set (${train_pct}% of data)"
else
  echo "Error: Data splitting failed."
  exit 1
fi

