from enum import Enum
from typing import List
from src.constants import HKQuantityType, MotionActivityType, HKWorkoutType


def get_daily_matrix_column_names() -> List[str]:
    """
    Get the column names (rows) of the daily matrix in order.
    
    Returns:
        List[str]: List of column names in the order they appear in the daily matrix
    """
    column_names = []
    
    # Add HealthKit quantity types
    column_names.extend([qt.value for qt in HKQuantityType])
    
    # Add Motion activity types
    column_names.extend([mt.value for mt in MotionActivityType])
    
    # Add Sleep states
    column_names.extend([
        "HKCategoryValueSleepAnalysisAsleep",
        "HKCategoryValueSleepAnalysisInBed"
    ])
    
    # Add Workout types
    column_names.extend([wt.value for wt in HKWorkoutType])
    
    return column_names

if __name__ == "__main__":
    print(get_daily_matrix_column_names())
