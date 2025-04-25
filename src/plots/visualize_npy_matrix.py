import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from typing import Optional, Tuple, List, Union

from constants import HKQuantityType, MotionActivityType, HKWorkoutType

# Define the colormap for visualization
DEFAULT_CMAP = plt.cm.viridis

# Define feature labels in the NPY matrix
FEATURE_LABELS = [
    # HealthKit Quantity Types
    "StepCount",
    "ActiveEnergyBurned",
    "DistanceWalkingRunning",
    "DistanceCycling",
    "AppleStandTime",
    "HeartRate",
    # Workout Types
    "HKWorkoutActivityTypeWalking",
    "HKWorkoutActivityTypeCycling",
    "HKWorkoutActivityTypeRunning",
    "HKWorkoutActivityTypeOther",
    "HKWorkoutActivityTypeMixedMetabolicCardioTraining",
    "HKWorkoutActivityTypeTraditionalStrengthTraining",
    "HKWorkoutActivityTypeElliptical",
    "HKWorkoutActivityTypeHighIntensityIntervalTraining",
    "HKWorkoutActivityTypeFunctionalStrengthTraining",
    "HKWorkoutActivityTypeYoga",
    # Sleep Analysis
    "Sleep Asleep",
    "Sleep In Bed",
    # Motion Types
    "stationary",
    "walking",
    "running",
    "automotive",
    "cycling",
    "not available"
]

def visualize_npy_file(
    npy_file_path: str,
    output_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (15, 20),
    dpi: int = 100,
    feature_subset: Optional[List[int]] = None,
    title: Optional[str] = None
) -> None:
    """
    Visualizes a NPY matrix file created by the create.py module as individual time series plots.
    
    Parameters:
    -----------
    npy_file_path : str
        Path to the NPY file to visualize
    output_path : Optional[str]
        Path to save the visualization. If None, the plot is not saved.
    show_plot : bool
        Whether to display the plot
    figsize : Tuple[int, int]
        Figure size in inches (width, height)
    dpi : int
        Resolution of the figure
    feature_subset : Optional[List[int]]
        Indices of features to visualize. If None, all features are visualized.
    title : Optional[str]
        Title for the plot. If None, uses the filename.
    
    Returns:
    --------
    None
    """
    # Load the NPY file
    try:
        data_matrix = np.load(npy_file_path)
    except Exception as e:
        print(f"Error loading NPY file: {e}")
        return
    
    # Extract the data channel (channel index 1)
    if data_matrix.shape[0] == 2:  # If the array has mask and data channels
        data = data_matrix[1]  # Data channel
        mask = data_matrix[0]  # Mask channel
    else:
        data = data_matrix
        mask = np.ones_like(data)
        print("Warning: Data doesn't have the expected shape with mask and data channels.")
    
    # Apply feature subset filter if provided
    if feature_subset is not None:
        data = data[feature_subset]
        mask = mask[feature_subset]
        labels = [FEATURE_LABELS[i] for i in feature_subset]
    else:
        labels = FEATURE_LABELS[:data.shape[0]]  # Use only as many labels as we have features
    
    # Create time array (minutes)
    time = np.arange(data.shape[1])
    
    # Create figure with subplots
    n_features = len(labels)
    fig, axes = plt.subplots(n_features, 1, figsize=figsize, dpi=dpi, sharex=True)
    if n_features == 1:
        axes = [axes]  # Ensure axes is always a list
    
    # Add more space on the left for labels
    plt.subplots_adjust(left=0.2)
    
    # Plot each feature
    for i, (ax, label) in enumerate(zip(axes, labels)):
        # Special handling for heart rate
        if "HeartRate" in label:
            # Plot heart rate as markers only
            ax.plot(time, data[i], 'o', markersize=2, label='Data')
        else:
            # Plot other features as lines
            ax.plot(time, data[i], label='Data')
        
        # Plot the mask if it's not all ones
        if not np.all(mask[i] == 1):
            ax.plot(time, mask[i], label='Mask', alpha=0.5)
        
        # Keep y-ticks but hide the feature label as y-axis label
        ax.set_ylabel('')  # Clear default axis label
        
        # Add feature label as text on the left
        ax.text(-0.22, 0.5, label, va='center', ha='left', transform=ax.transAxes)
        
        # Add legend if mask is plotted
        if not np.all(mask[i] == 1):
            ax.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Set x-axis label and ticks for the last subplot
    axes[-1].set_xlabel('Time (minutes)')
    
    # Convert minutes to hours for x-axis ticks
    hours = np.arange(0, 24, 2)
    hour_ticks = np.array(hours) * 60
    axes[-1].set_xticks(hour_ticks)
    axes[-1].set_xticklabels([f"{h:02d}:00" for h in hours])
    
    # Add title
    if title is None:
        title = os.path.basename(npy_file_path)
    fig.suptitle(title, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def visualize_npy_directory(
    directory_path: str,
    output_directory: Optional[str] = None,
    max_files: int = 10,
    **kwargs
) -> None:
    """
    Visualizes all NPY files in a directory.
    
    Parameters:
    -----------
    directory_path : str
        Path to the directory containing NPY files
    output_directory : Optional[str]
        Directory to save visualizations. If None, visualizations are not saved.
    max_files : int
        Maximum number of files to visualize
    **kwargs
        Additional arguments to pass to visualize_npy_file
    
    Returns:
    --------
    None
    """
    # Find all NPY files in the directory
    npy_files = [f for f in os.listdir(directory_path) if f.endswith('.npy')]
    npy_files.sort()  # Sort files by name
    
    # Limit to max_files
    if len(npy_files) > max_files:
        print(f"Found {len(npy_files)} NPY files, visualizing first {max_files}.")
        npy_files = npy_files[:max_files]
    
    # Create output directory if needed
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Process each file
    for npy_file in npy_files:
        file_path = os.path.join(directory_path, npy_file)
        
        # Determine output path if saving
        output_path = None
        if output_directory:
            base_name = os.path.splitext(npy_file)[0]
            output_path = os.path.join(output_directory, f"{base_name}_visualization.png")
        
        # Visualize the file
        visualize_npy_file(
            npy_file_path=file_path,
            output_path=output_path,
            title=npy_file,
            **kwargs
        )


if __name__ == "__main__":
    # Example usage
    # visualize_npy_file("/path/to/data/2022-01-01.npy")
    
    # Example with feature subset (healthkit steps, sleep, and stationary motion)
    # visualize_npy_file(
    #     "/path/to/data/2022-01-01.npy",
    #     feature_subset=[0, 7, 8, 10],  # StepCount, Sleep Asleep, Sleep In Bed, stationary
    #     output_path="/path/to/output/visualization.png"
    # )
    
    # Example to visualize a specific NPY file
    visualize_npy_file(
        "/var/folders/s5/9tx92s6n1r1g20vf5q0j7gxc0000gp/T/tmp7tynf79f/2017-08-03.npy",
        output_path="/Users/narayanschuetz/tmp_data/verify_data/2017-08-03_visualization.png",
        show_plot=True
    )

