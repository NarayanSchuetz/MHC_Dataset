import json
import pandas as pd
import enum
import abc
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
from collections import defaultdict


class HKID(enum.Enum):
    HKQuantityTypeIdentifierStepCount = "HKQuantityTypeIdentifierStepCount"
    HKQuantityTypeIdentifierActiveEnergyBurned = "HKQuantityTypeIdentifierActiveEnergyBurned"
    HKQuantityTypeIdentifierDistanceWalkingRunning = "HKQuantityTypeIdentifierDistanceWalkingRunning"
    HKQuantityTypeIdentifierDistanceCycling = "HKQuantityTypeIdentifierDistanceCycling"
    HKQuantityTypeIdentifierAppleStandTime = "HKQuantityTypeIdentifierAppleStandTime"
    HKQuantityTypeIdentifierHeartRate = "HKQuantityTypeIdentifierHeartRate"
    HKCategoryTypeIdentifierSleepAnalysis = "HKCategoryTypeIdentifierSleepAnalysis"
    HKWorkoutTypeIdentifier = "HKWorkoutTypeIdentifier"
    MotionCollector = "MotionCollector"

class CalculatRobustStatsForHealthkit:
    def calculate(self, df, filepath=None):
        sums = df.groupby("device").value.sum(numeric_only=True)
        means = df.groupby("device").value.mean(numeric_only=True)
        medians = df.groupby("device").value.median(numeric_only=True)
        q25s = self._calculate_quantiles(df, 0.25)
        q75s = self._calculate_quantiles(df, 0.75)
        iqrs = q75s - q25s
        result = {
            "sums": sums.to_dict(),
            "means": means.to_dict(),
            "medians": medians.to_dict(),
            "q25s": q25s.to_dict(),
            "q75s": q75s.to_dict(),
            "iqrs": iqrs.to_dict()
        }
        if filepath is not None:
            with open(filepath + ".json", "w") as f:
                json.dump(result, f)
        return result
        
    def _calculate_quantiles(self, df, q):
        return df.groupby("device").value.quantile(q, numeric_only=True)

class CalculateHourlyMeanForHealthkit:
    def calculate(self, df, filepath=None):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DateTime index.")
        
        hourly_median = df.resample('H').value.mean()
        full_range = pd.date_range(start=df.index.min().normalize(), periods=24, freq='H')
        hourly_median = hourly_median.reindex(full_range, fill_value=pd.NA)

        if filepath is not None:
            hourly_median.to_frame().to_parquet(filepath + ".parquet")

        return hourly_median

class CalculateSleepDuration:
    def calculate(self, df, filepath=None):
        total_seconds_in_bed = df[df["category value"] == "HKCategoryValueSleepAnalysisInBed"].groupby("device").value.sum()
        total_seconds_asleep = df[df["category value"] == "HKCategoryValueSleepAnalysisAsleep"].groupby("device").value.sum()
        
        result = {
            "seconds_in_bed": total_seconds_in_bed.to_dict(),
            "total_seconds_asleep": total_seconds_asleep.to_dict()
        }

        if filepath is not None:
            with open(filepath + ".json", "w") as f:
                json.dump(result, f)

        return result

class CalculateWorkoutSeconds:
    def calculate(self, df, filepath=None):
        df = df.copy()
        df["delta"] = (df["endTime"] - df["startTime"]).dt.total_seconds()
        result_df = df[["workoutType", "delta"]]

        # Summing deltas for each workout type
        result_df = result_df.groupby("workoutType", as_index=False).sum()

        # Creating a DataFrame for all workout categories with delta 0
        all_workouts_df = pd.DataFrame(HkWorkout2Plot.WORKOUT_CATEGORIES, columns=["workoutType"])
        all_workouts_df["delta"] = 0

        # Merging with the result_df to ensure all workout types are present
        final_df = pd.merge(all_workouts_df, result_df, on="workoutType", how="left", suffixes=("", "_actual"))
        final_df["delta"] = final_df["delta_actual"].fillna(0)
        final_df = final_df[["workoutType", "delta"]]

        if filepath is not None:
            final_df.to_parquet(filepath + ".parquet")

        return final_df

class CalculateMotionDistribution:
    def calculate(self, df, filepath=None):
        df = df.copy()
        df["delta"] = (df["endTime"] - df["startTime"]).dt.total_seconds()

        # Summing deltas for each activity
        sums = df.groupby("activity", as_index=False)["delta"].sum()

        # Creating a DataFrame for all activities with delta 0
        all_activities_df = pd.DataFrame(Motion2Plot.ACTIVITIES, columns=["activity"])
        all_activities_df["delta"] = 0

        # Merging with the sums to ensure all activities are present
        final_df = pd.merge(all_activities_df, sums, on="activity", how="left", suffixes=("", "_actual"))
        final_df["delta"] = final_df["delta_actual"].fillna(0)
        final_df = final_df[["activity", "delta"]]

        # Ensuring the order of activities is as specified in ACTIVITIES
        final_df = final_df.set_index("activity").reindex(Motion2Plot.ACTIVITIES).reset_index()
        final_df.index = final_df.activity

        result = final_df.delta.to_dict()

        if filepath is not None:
            with open(filepath + ".json", "w") as f:
                json.dump(result, f)

        return result

class HealthKitCalculatorFactory:
    @staticmethod
    def get_calculator(hk_id):
        if hk_id == HKID.HKQuantityTypeIdentifierStepCount:
            return CalculatRobustStatsForHealthkit()
        elif hk_id == HKID.HKQuantityTypeIdentifierActiveEnergyBurned:
            return CalculatRobustStatsForHealthkit()
        elif hk_id == HKID.HKQuantityTypeIdentifierDistanceWalkingRunning:
            return CalculatRobustStatsForHealthkit()
        elif hk_id == HKID.HKQuantityTypeIdentifierDistanceCycling:
            return CalculatRobustStatsForHealthkit()
        elif hk_id == HKID.HKQuantityTypeIdentifierAppleStandTime:
            return CalculatRobustStatsForHealthkit()
        elif hk_id == HKID.HKQuantityTypeIdentifierHeartRate:
            return CalculateHourlyMeanForHealthkit()
        elif hk_id == HKID.HKCategoryTypeIdentifierSleepAnalysis:
            return CalculateSleepDuration()
        elif hk_id == HKID.HKWorkoutTypeIdentifier:
            return CalculateWorkoutSeconds()
        elif hk_id == HKID.MotionCollector:
            return CalculateMotionDistribution()
        else:
            raise ValueError(f"Unsupported HKID: {hk_id}")

class TsPreprocessors(abc.ABC):
    @abc.abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class Identity(TsPreprocessors):
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

class HkDedupAppVersion(TsPreprocessors):
    def __init__(self, strategy: str = "dominant"):
        self._strategy = strategy

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._strategy == "dominant":
            return self._dedup_dominant_app_version(df)
        else:
            raise ValueError(f"Unknown strategy '{self._strategy}'")

    def _dedup_dominant_app_version(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.appVersion.isna().all():
            print("Warning: All appVersion values are NaN. Returning the original DataFrame.")
            return df  # Return the DataFrame unchanged
        
        dominant_app_version = df.appVersion.value_counts().index[0]
        return df[df.appVersion == dominant_app_version]


class BaseTs2Plot(abc.ABC):
    HK_QUANTITY_TYPE_IDENTIFIER = None
    _FACTORY = HealthKitCalculatorFactory
    _YLIM = None
    _COLORMAP = None
    _XLABEL = ""
    _YLABEL = ""
    _TITLE = ""

    def __init__(self, preprocessors: Tuple = tuple(), alpha=0.8, color=True, labels=False, legend=True, root_dir=".", resample_by_minute=False):
        self._preprocessors = preprocessors
        self.alpha = alpha
        self.color = color
        self._labels = labels
        self._legend = legend
        self._root_dir = root_dir
        self._resample_by_minute = resample_by_minute

        if self.HK_QUANTITY_TYPE_IDENTIFIER is None:
            raise NotImplementedError("HK_QUANTITY_TYPE_IDENTIFIER must be defined in any subclass")
        if self._YLIM is None:
            raise NotImplementedError("_YLIM must be defined in any subclass")
        if self._COLORMAP is None:
            raise NotImplementedError("_COLORMAP must be defined in any subclass")

    def _main(self, df: pd.DataFrame, ax: plt.axes) -> None:
        if df.empty:
            return

        df = df.copy()
        for preprocessor in self._preprocessors:
            df = preprocessor.preprocess(df)

        # Resample data by minute if requested
        if self._resample_by_minute and not df.empty:
            df = self._resample_to_minute_intervals(df)

        #self._calculate_and_save_reconstruction_targets(df, os.path.join(self._root_dir, self.HK_QUANTITY_TYPE_IDENTIFIER.value))

        return self.extract(df, ax)
    
    def _resample_to_minute_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample a DataFrame to minute intervals using mean aggregation.
        This is applied after timezone normalization.
        """
        # Make a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        try:
            print(f"Resampling data with shape {result_df.shape} to minute intervals")
            
            # Convert to datetime index if not already
            if not isinstance(result_df.index, pd.DatetimeIndex):
                if 'startTime' in result_df.columns:
                    result_df.index = pd.DatetimeIndex(result_df['startTime'])
                    print("Converted 'startTime' column to datetime index for resampling")
                else:
                    print("Warning: Cannot resample DataFrame without a datetime index or startTime column")
                    return df
            
            # Check if required columns are present
            required_cols = ['startTime', 'device']
            missing_cols = [col for col in required_cols if col not in result_df.columns]
            if missing_cols:
                print(f"Warning: Missing required columns for resampling: {missing_cols}")
                return df
            
            # Group by device first to maintain device separation
            device_groups = []
            for device_name, device_group in result_df.groupby('device'):
                # Resample each device group separately
                print(f"Resampling device group: {device_name} with {len(device_group)} records")
                
                # Round startTime to the nearest minute to ensure proper grouping
                device_group['startTime_minute'] = device_group['startTime'].dt.floor('min')
                
                # Aggregate by minute
                minute_groups = []
                for minute, minute_group in device_group.groupby('startTime_minute'):
                    # Create a single aggregated record for this minute
                    agg_record = {
                        'device': device_name,
                        'startTime': minute,
                        'endTime': minute + pd.Timedelta(minutes=1)
                    }
                    
                    # Add the value column if it exists
                    if 'value' in minute_group.columns:
                        agg_record['value'] = minute_group['value'].mean()
                    
                    # Copy other columns from the first record in the group
                    for col in minute_group.columns:
                        if col not in agg_record and col not in ['startTime', 'endTime', 'startTime_minute']:
                            # Use mean for numeric columns, first for others
                            if pd.api.types.is_numeric_dtype(minute_group[col]):
                                agg_record[col] = minute_group[col].mean()
                            else:
                                agg_record[col] = minute_group[col].iloc[0]
                    
                    minute_groups.append(agg_record)
                
                # Convert minute groups to DataFrame
                if minute_groups:
                    device_minute_df = pd.DataFrame(minute_groups)
                    device_groups.append(device_minute_df)
                    print(f"Created {len(device_minute_df)} minute-level records for device {device_name}")
                else:
                    print(f"No minute-level records created for device {device_name}")
            
            # Combine all device groups
            if device_groups:
                resampled = pd.concat(device_groups, ignore_index=True)
                # Set index to startTime
                resampled.index = pd.DatetimeIndex(resampled['startTime'])
                print(f"Final resampled data has shape {resampled.shape}")
                return resampled
            else:
                print("No resampled data created")
                return df
            
        except Exception as e:
            print(f"Error in resampling: {e}")
            import traceback
            traceback.print_exc()
            return df  # Return original if resampling fails

    @abc.abstractmethod
    def extract(self, df: pd.DataFrame, ax: plt.axes) -> None:
        pass

    def _calculate_and_save_reconstruction_targets(self, df, filepath):
        calculator = self._FACTORY.get_calculator(self.HK_QUANTITY_TYPE_IDENTIFIER)
        calculator.calculate(df=df, filepath=filepath)

    def _add_labels(self, ax: plt.axes):
        ax.set_title(self._TITLE)
        ax.set_ylabel(self._YLABEL)
        ax.set_xlabel(self._XLABEL)

    __call__ = _main

class HkBarChart(BaseTs2Plot):
    HK_QUANTITY_TYPE_IDENTIFIER = None
    _YLIM = None
    _COLORMAP = None
    _XLABEL = ""
    _YLABEL = ""
    _TITLE = ""

    def extract(self, df: pd.DataFrame, ax: plt.axes) -> None:
        # Get the top 5 devices
        devices = df.device.value_counts()[:5]
        devices = devices.index

        # Sort the devices alphabetically
        devices = sorted(devices)

        # Prepare colors for each device
        device_colors = {device: self._COLORMAP(i) for i, device in enumerate(devices)}

        handles = []
        for device in devices:
            sub = df[df.device == device]
            device_patches = []
            color = device_colors[device]
            for i, (_, row) in enumerate(sub.iterrows()):
                start_time, end_time = row.startTime, row.endTime
                duration = (end_time - start_time).total_seconds()
                if duration == 0:
                    continue
                start_second = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
                height = row["value"] / duration
                height = np.clip(height, *self._YLIM)
                if self.color:
                    patch = ax.broken_barh([(start_second, duration)], (0, height), facecolors=color, alpha=self.alpha)
                else:
                    patch = ax.broken_barh([(start_second, duration)], (0, height), alpha=self.alpha)
                device_patches.append(patch)
            if device_patches:
                handles.append(device_patches[-1])

        ax.set_ylim(*self._YLIM)
        ax.set_xlim(0, 24*60*60)
        if self._legend and len(handles) > 0:
            ax.legend(handles, devices, loc='upper right', title="Devices")
        if self._labels:
            self._add_labels(ax)

class HkSteps2Plot(HkBarChart):
    HK_QUANTITY_TYPE_IDENTIFIER = HKID.HKQuantityTypeIdentifierStepCount
    _YLIM = (0, 5)  # 0-5 Steps per second
    _COLORMAP = ListedColormap(['#1A237E', '#304FFE','#3949AB', '#7986CB', '#8C9EFF'])
    _XLABEL = "Time (seconds from midnight)"
    _YLABEL = "Steps per second (steps/s)"
    _TITLE = "Step Count"

class HkActiveEnergyBurned2Plot(HkBarChart):
    HK_QUANTITY_TYPE_IDENTIFIER = HKID.HKQuantityTypeIdentifierActiveEnergyBurned
    _YLIM = (0, 250)
    _COLORMAP = ListedColormap(["#311B92", '#7C4DFF', '#673AB7', '#B39DDB', '#6200EA'])
    _XLABEL = "Time (seconds from midnight)"
    _YLABEL = "Calories per second (cal/s)"
    _TITLE = "Active Energy Burned"

class HkDistanceWalkingRunning2Plot(HkBarChart):
    HK_QUANTITY_TYPE_IDENTIFIER = HKID.HKQuantityTypeIdentifierDistanceWalkingRunning
    _YLIM = (0, 3)
    _COLORMAP = ListedColormap(["#004D40", "#00BFA5", "#009688", "#64FFDA", "#80CBC4"])
    _XLABEL = "Time (seconds from midnight)"
    _YLABEL = "Distance per second (m/s)"
    _TITLE = "Distance Walking/Running"

class HkDistanceCycling2Plot(HkBarChart):
    HK_QUANTITY_TYPE_IDENTIFIER = HKID.HKQuantityTypeIdentifierDistanceCycling
    _YLIM = (0, 10)
    _COLORMAP = ListedColormap(["#BF360C", "#FF3D00", "#FF5722", "#FF8A65", "#FF9E80"])
    _XLABEL = "Time (seconds from midnight)"
    _YLABEL = "Distance per second (m/s)"
    _TITLE = "Distance Cycling"

class HkAppleStandTime2Plot(HkBarChart):
    HK_QUANTITY_TYPE_IDENTIFIER = HKID.HKQuantityTypeIdentifierAppleStandTime
    _YLIM = (0, 1)
    _COLORMAP = ListedColormap(["#3E2723", "#5D4037", "#8D6E63", "#A1887F", "#D7CCC8"])
    _XLABEL = "Time (seconds from midnight)"
    _YLABEL = "Stand per second (stand/s)"
    _TITLE = "Stand Time"

    def extract(self, df, ax):
        df["value"] = df["value"] * 60  # convert minutes to seconds
        super().extract(df, ax)

class HkHeartRate2Plot(BaseTs2Plot):
    HK_QUANTITY_TYPE_IDENTIFIER = HKID.HKQuantityTypeIdentifierHeartRate
    _YLIM = (40, 200)
    _COLORMAP = ListedColormap(["#B71C1C", "#FF1744", "#F44336", "#EF9A9A", "#FF5252"])
    _XLABEL = "Time (seconds from midnight)"
    _YLABEL = "Heart Rate (bpm)"
    _TITLE = "Heart Rate"
    _MARKER_SIZE = 0.1

    def extract(self, df: pd.DataFrame, ax: plt.axes) -> None:
        devices = df.device.value_counts()[:5]
        devices = devices.index
        devices = sorted(devices)
        device_colors = {device: self._COLORMAP(i) for i, device in enumerate(devices)}

        for device in devices:
            sub = df[df.device == device].copy()
            color = device_colors[device]
            sub['value'] = sub['value'] * 60 
            sub['time_since_midnight'] = sub['startTime'].dt.hour * 3600 + sub['startTime'].dt.minute * 60 + sub['startTime'].dt.second
            
            # Use line plot for resampled data (more appropriate for minute resolution)
            if self._resample_by_minute:
                # Sort by time to ensure proper line connection
                sub = sub.sort_values('time_since_midnight')
                ax.plot(sub['time_since_midnight'], sub['value'], color=color, alpha=self.alpha, label=device, linewidth=1)
            else:
                # Use scatter plot for original high-resolution data
                ax.scatter(sub['time_since_midnight'], sub['value'], color=color, alpha=self.alpha, label=device, s=self._MARKER_SIZE)
        
        ax.set_ylim(*self._YLIM)
        ax.set_xlim(0, 24*60*60)
        if self._legend and len(devices) > 0:
            ax.legend(loc='upper right', title="Devices")
        if self._labels:
            self._add_labels(ax)

class HkSleep2Plot(BaseTs2Plot):
    HK_QUANTITY_TYPE_IDENTIFIER = HKID.HKCategoryTypeIdentifierSleepAnalysis
    _YLIM = (0, 5)
    _COLORMAP = ListedColormap(['#880E4F', '#F50057','#EC407A', '#F48FB1', '#FF4081'])
    _XLABEL = "Time (seconds from midnight)"
    _YLABEL = "Devices"
    _TITLE = "Sleep Analysis"

    def extract(self, df: pd.DataFrame, ax: plt.axes) -> None:
        if df.empty:
            print("Empty DataFrame for Sleep plot")
            return
            
        devices = df.device.value_counts()[:5]
        devices = devices.index
        y_ticks = list(range(5))
        y_labels = list(devices) + [''] * (5 - len(devices))

        color_mapping = {
            "HKCategoryValueSleepAnalysisAsleep": 1.0,
            "HKCategoryValueSleepAnalysisInBed": 0.5
        }

        for i, device in enumerate(devices):
            sub = df[df.device == device]
            color = self._COLORMAP(i)
            for _, row in sub.iterrows():
                try:
                    start_time = row.startTime
                    # In case value isn't present, use fixed duration
                    if hasattr(row, 'value') and not pd.isna(row.value):
                        end_time = start_time + pd.Timedelta(seconds=row.value)
                    else:
                        # For resampled data, use 1-minute duration
                        end_time = row.endTime if hasattr(row, 'endTime') else start_time + pd.Timedelta(minutes=1)
                        
                    start_second = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
                    duration = (end_time - start_time).total_seconds()
                    if duration == 0:
                        continue
                        
                    # Get alpha (transparency) value based on sleep category
                    if hasattr(row, 'category value') and not pd.isna(row['category value']):
                        alpha = color_mapping.get(row['category value'], 0.5)
                    else:
                        alpha = 0.5
                        
                    ax.broken_barh([(start_second, duration)], (i - 0.4, 0.8), facecolors=color, alpha=alpha)
                except Exception as e:
                    print(f"Error plotting sleep data: {e}")
                    continue
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(-0.5, 4.5)
        ax.set_xlim(0, 24*60*60)
        if self._labels:
            self._add_labels(ax)

class HkWorkout2Plot(BaseTs2Plot):
    HK_QUANTITY_TYPE_IDENTIFIER = HKID.HKWorkoutTypeIdentifier
    _YLIM = (0, 10)
    _COLORMAP = ListedColormap(['#827717', '#AFB42B', '#CDDC39', '#E6EE9C', '#AEEA00', '#C6FF00', '#EEFF41', '#F4FF81', '#F0F4C3', '#FFEE58'])
    _XLABEL = "Time (seconds from midnight)"
    _YLABEL = "Workouts"
    _TITLE = "Workout Analysis"

    WORKOUT_CATEGORIES = [
        "HKWorkoutActivityTypeWalking",
        "HKWorkoutActivityTypeCycling",
        "HKWorkoutActivityTypeRunning",
        "HKWorkoutActivityTypeOther",
        "HKWorkoutActivityTypeMixedMetabolicCardioTraining",
        "HKWorkoutActivityTypeTraditionalStrengthTraining",
        "HKWorkoutActivityTypeElliptical",
        "HKWorkoutActivityTypeHighIntensityIntervalTraining",
        "HKWorkoutActivityTypeFunctionalStrengthTraining",
        "HKWorkoutActivityTypeYoga"
    ]

    def extract(self, df: pd.DataFrame, ax: plt.axes) -> None:
        if df.empty:
            print("Empty DataFrame for Workout plot")
            return
            
        # Default to 'Other' if workoutType is missing or not in predefined categories
        if 'workoutType' in df.columns:
            df['workoutType'] = df['workoutType'].apply(
                lambda x: x if x in self.WORKOUT_CATEGORIES else 'HKWorkoutActivityTypeOther')
        else:
            print("Warning: workoutType column missing in workout data")
            df['workoutType'] = 'HKWorkoutActivityTypeOther'
            
        workout_types = self.WORKOUT_CATEGORIES
        y_ticks = list(range(10))
        y_labels = [w[21:] for w in workout_types]

        for i, workout in enumerate(workout_types):
            sub = df[df.workoutType == workout]
            if sub.empty:
                continue
                
            color = self._COLORMAP(i)
            for _, row in sub.iterrows():
                try:
                    start_time = row.startTime
                    end_time = row.endTime if hasattr(row, 'endTime') else start_time + pd.Timedelta(minutes=1)
                    
                    start_second = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
                    duration = (end_time - start_time).total_seconds()
                    if duration == 0:
                        continue
                    ax.broken_barh([(start_second, duration)], (i - 0.4, 0.8), facecolors=color, alpha=self.alpha)
                except Exception as e:
                    print(f"Error plotting workout data: {e}")
                    continue
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(-0.5, 9.5)
        ax.set_xlim(0, 24*60*60)
        if self._labels:
            self._add_labels(ax)

class Motion2Plot(BaseTs2Plot):
    HK_QUANTITY_TYPE_IDENTIFIER = HKID.MotionCollector
    _YLIM = (-0.5, 5.5)
    _XLIM = (0, 24 * 60 * 60)
    _COLORMAP = ListedColormap(['#1B5E20', '#388E3C', '#00C853', '#66BB6A', '#00E676', '#69F0AE'])
    _XLABEL = "Time (seconds from midnight)"
    _YLABEL = "Activities"
    _TITLE = "Activity Analysis"
    alpha = 0.7

    ACTIVITIES = [
        "stationary",
        "automotive",
        "walking",
        "not available",
        "running",
        "cycling"
    ]

    def extract(self, df: pd.DataFrame, ax: plt.axes) -> None:
        if df.empty:
            print("Empty DataFrame for Motion plot")
            return
            
        # Default to 'not available' if activity is missing or not in predefined list
        if 'activity' in df.columns:
            df['activity'] = df['activity'].apply(lambda x: x if x in self.ACTIVITIES else 'not available')
        else:
            print("Warning: activity column missing in motion data")
            df['activity'] = 'not available'
            
        y_ticks = list(range(len(self.ACTIVITIES)))
        y_labels = self.ACTIVITIES

        for i, activity in enumerate(self.ACTIVITIES):
            sub = df[df.activity == activity]
            if sub.empty:
                continue
                
            color = self._COLORMAP(i)
            for _, row in sub.iterrows():
                try:
                    start_time = pd.to_datetime(row.startTime)
                    end_time = pd.to_datetime(row.endTime) if hasattr(row, 'endTime') else start_time + pd.Timedelta(minutes=1)
                    
                    start_second = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
                    duration = (end_time - start_time).total_seconds()
                    if duration == 0:
                        continue
                    ax.broken_barh([(start_second, duration)], (i - 0.4, 0.8), facecolors=color, alpha=self.alpha)
                except Exception as e:
                    print(f"Error plotting motion data: {e}")
                    continue
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(self._YLIM)
        ax.set_xlim(self._XLIM)
        if self._labels:
            ax.set_title(self._TITLE)
            self._add_labels(ax)

    def _add_labels(self, ax: plt.axes) -> None:
        for bar in ax.patches:
            try:
                if bar.get_width() > 60:  # Only label bars wider than 1 minute
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f'{int(bar.get_width() / 60)} min',
                        ha='center',
                        va='center',
                        color='white',
                        fontsize=8
                    )
            except Exception as e:
                print(f"Error adding label to bar: {e}")
                continue
            

def load_data(base_path: str, user_id: str, file_type: str) -> pd.DataFrame:
    """
    Load HealthKit data from parquet files. If the file is not found, return an empty DataFrame.
    """
    file_paths = {
        'healthkit': f"{base_path}/healthkit/private/{user_id}.parquet",
        'workout': f"{base_path}/healthkit_workout/private/healthkit_workout.parquet",
        'sleep': f"{base_path}/healthkit_sleep/private/healthkit_sleep.parquet",
        'motion': f"{base_path}/motion_collector_preprocessed/{user_id}.parquet"
    }

    file_path = file_paths.get(file_type)
    if file_path is None:
        raise ValueError(f"Invalid file_type '{file_type}'. Must be one of {list(file_paths.keys())}")

    try:
        df = pd.read_parquet(file_path)
        if file_type in ['workout', 'sleep']:
            df = df[df.healthCode == user_id]
        df.index = df.startTime
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"Error loading file at {file_path}: {e}")
        return pd.DataFrame(columns=['source', 'healthCode'], index=pd.DatetimeIndex([]))


def find_iphones_and_apple_watches(df_healthkit: pd.DataFrame) -> Tuple[set, set]:
    """
    Find Apple Watch and iPhone devices in HealthKit data.
    """
    apple_watches = set(df_healthkit[
        (df_healthkit.type == "HKQuantityTypeIdentifierHeartRate") & 
        (df_healthkit.sourceIdentifier.str.contains("com.apple.health."))
    ].source)
    
    iphones = set(df_healthkit[
        (df_healthkit.type != "HKQuantityTypeIdentifierHeartRate") & 
        (df_healthkit.sourceIdentifier.str.contains("com.apple.health."))
    ].source)
    
    return apple_watches, iphones

def create_plots(
    df_healthkit: pd.DataFrame, 
    df_sleep: pd.DataFrame, 
    df_workout: pd.DataFrame, 
    df_motion: pd.DataFrame,
    output_size: Tuple[int, int] = (320, 260),
    dpi: int = 100, 
    filepath: str = None, 
    legend: bool = False, 
    labels: bool = False, 
    show_output: bool = True,
    resample_by_minute: bool = False
) -> None:
    """
    Create a 3x3 grid of HealthKit plots.
    """
    figsize = (output_size[0] / dpi, output_size[1] / dpi)
    fig, ax = plt.subplots(3, 3, figsize=figsize, dpi=dpi,
                           gridspec_kw={'wspace': 0, 'hspace': 0})
    ax_flat = ax.flatten()

    plotters = [
        HkSteps2Plot(alpha=0.6, color=True, preprocessors=(HkDedupAppVersion(),), labels=labels, legend=legend, resample_by_minute=resample_by_minute),
        HkActiveEnergyBurned2Plot(alpha=0.6, color=True, preprocessors=(HkDedupAppVersion(),), labels=labels, legend=legend, resample_by_minute=resample_by_minute),
        HkDistanceWalkingRunning2Plot(alpha=0.6, color=True, preprocessors=(HkDedupAppVersion(),), labels=labels, legend=legend, resample_by_minute=resample_by_minute),
        HkDistanceCycling2Plot(alpha=0.6, color=True, preprocessors=(HkDedupAppVersion(),), labels=labels, legend=legend, resample_by_minute=resample_by_minute),
        HkAppleStandTime2Plot(alpha=0.6, color=True, preprocessors=(HkDedupAppVersion(),), labels=labels, legend=legend, resample_by_minute=resample_by_minute),
        HkHeartRate2Plot(alpha=1, color=True, preprocessors=(HkDedupAppVersion(),), labels=labels, legend=legend, resample_by_minute=resample_by_minute),
        HkSleep2Plot(alpha=None, color=True, preprocessors=(HkDedupAppVersion(),), labels=labels, legend=legend, resample_by_minute=resample_by_minute),
        HkWorkout2Plot(alpha=0.6, color=True, preprocessors=(HkDedupAppVersion(),), labels=labels, legend=legend, resample_by_minute=resample_by_minute),
        Motion2Plot(alpha=0.6, color=True, labels=labels, legend=legend, resample_by_minute=resample_by_minute)
    ]

    for i, plotter in enumerate(plotters):
        if i < 6:
            sub_data = df_healthkit[df_healthkit.type == plotter.HK_QUANTITY_TYPE_IDENTIFIER.value]
            plotter(sub_data, ax_flat[i])
        elif i == 6:
            plotter(df_sleep, ax_flat[i])
        elif i == 7:
            plotter(df_workout, ax_flat[i])
        else:
            plotter(df_motion, ax_flat[i])

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    for axis in ax_flat:
        axis.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False,
                         right=False, labelright=False)

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    else:
        if not show_output:
            raise ValueError("A filepath must be provided if show_output is disabled.")
    
    if show_output:
        plt.show()
    else:
        plt.close(fig)  # Ensure the figure is closed if not shown

def convert_to_local_time(df, offset_column="startTime_timezone_offset"):
    """
    Converts the timezone of a DataFrame to local time based on the timezone offset.
    Adjusts both the DataFrame index and the startTime/endTime columns.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    df = df.copy()
    
    # Apply offset to the index
    df.index = df.index + pd.to_timedelta(df[offset_column], unit='m')
    
    # Also apply offset to the actual time columns
    if 'startTime' in df.columns:
        df['startTime'] = df['startTime'] + pd.to_timedelta(df[offset_column], unit='m')
    
    if 'endTime' in df.columns:
        # If endTime has its own offset column, use that; otherwise use startTime's offset
        if 'endTime_timezone_offset' in df.columns:
            df['endTime'] = df['endTime'] + pd.to_timedelta(df['endTime_timezone_offset'], unit='m')
        else:
            df['endTime'] = df['endTime'] + pd.to_timedelta(df[offset_column], unit='m')
    
    return df

if __name__ == "__main__":
    pass
