#!/bin/bash

# Configuration - modify these paths as needed
# Dataset generation parameters
SHERLOCK_DATASET_PATH="/path/to/your/mhc_dataset"
OUTPUT_PATH="/path/to/your/output"
BATCH_SIZE=100
NUM_PROCESSES=16

# Data quality parameters
MIN_CHANNEL_COVERAGE=7.638888888888889  # 25th percentile of data coverage
MIN_CHANNELS_WITH_DATA=4
WINDOW_SIZE=7
MIN_REQUIRED_DAYS=5

# Train-test split parameters
INPUT_PARQUET="${OUTPUT_PATH}/valid_7day_windows.csv"
SPLIT_OUTPUT_DIR="${OUTPUT_PATH}/splits"
TEST_SIZE=0.3
RANDOM_STATE=42
SPLIT_METHOD="cluster"  # Options: basic, advanced, cluster
N_CLUSTERS=8
# Set to "true" to only include participants who opted to share with all qualified researchers
SHARING_SUBSET="true"
# Only needed for advanced or cluster split methods
DEMOGRAPHIC_PARQUET="${OUTPUT_PATH}/demographic_data.parquet"
INFO_PARQUET="${OUTPUT_PATH}/info_data.parquet"

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
echo ""

# Make sure output directory exists
mkdir -p "$OUTPUT_PATH"

# Execute the Python script with the specified parameters
echo "Step 1: Generating dataset windows..."
python src/generate_windows.py \
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

echo "Step 2: Creating train-test splits..."
# Create splits directory
mkdir -p "$SPLIT_OUTPUT_DIR"

# Build the command based on the split method
SPLIT_CMD="python src/train_test_splitter.py \
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
if [ $? -eq 0 ]; then
  echo "Dataset creation and splitting completed successfully!"
  echo "Output files are available at: $SPLIT_OUTPUT_DIR"
else
  echo "Error: Train-test splitting failed."
  exit 1
fi

