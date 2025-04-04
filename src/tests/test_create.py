import os
import glob
import numpy as np
import pandas as pd
import pytest

from create import (
    _generate_healthkit_minute_level_daily_data, 
    _generate_sleep_minute_level_daily_data, 
    _generate_motion_minute_level_daily_data, 
    _generate_workout_minute_level_daily_data,
    _convert_healthkit_units
)
from constants import HKQuantityType, MotionActivityType, HKWorkoutType


def load_test_dataframe():
    """
    Attempt to load a test HealthKit dataframe from a CSV file in the
    directory specified by the HEALTHKIT_TEST_DATA_DIR environment variable.
    If a CSV file is not found, generate a dummy dataframe with minimal test data.
    """
    test_data_dir = os.getenv("HEALTHKIT_TEST_DATA_DIR")
    if test_data_dir and os.path.isdir(test_data_dir):
        # Look for CSV files in the test data directory.
        csv_files = glob.glob(os.path.join(test_data_dir, "*.csv"))
        if csv_files:
            # Load the first CSV file found.
            df = pd.read_parquet(csv_files[0])
            return df

    # If no CSV file is available, generate a simple dummy dataframe for one day.
    # For example, January 1, 2022.
    start = pd.Timestamp("2022-01-01 00:00:00")
    end = start + pd.Timedelta(hours=1)  # record lasting 1 hour
    data = {
        "startTime": [start],
        "endTime": [end],
        "startTime_timezone_offset": [0],
        "endTime_timezone_offset": [0],
        "value": [3600],  # so that average value per second becomes 1.0
        # Assign one of the HealthKit category types (using the first as an example)
        "type": [list(HKQuantityType)[0].value]
    }
    df_dummy = pd.DataFrame(data)
    df_dummy.index = df_dummy["startTime"]
    return df_dummy


class TestConvertHealthKitUnits:

    def test_empty_dataframe(self):
        """Test that an empty DataFrame is returned unchanged."""
        df_empty = pd.DataFrame(columns=['value', 'unit'])
        df_converted = _convert_healthkit_units(df_empty)
        pd.testing.assert_frame_equal(df_empty, df_converted)

    def test_no_unit_column(self):
        """Test that a DataFrame without a 'unit' column is returned unchanged."""
        df_no_unit = pd.DataFrame({'value': [100]})
        df_converted = _convert_healthkit_units(df_no_unit)
        pd.testing.assert_frame_equal(df_no_unit, df_converted)

    def test_count_per_second_no_conversion(self):
        """Test that count/s is NOT converted (as the code is commented out)."""
        df_input = pd.DataFrame({'value': [1.0], 'unit': ['count/s']})
        df_expected = pd.DataFrame({'value': [1.0], 'unit': ['count/s']}) # Expect no change
        df_converted = _convert_healthkit_units(df_input)
        pd.testing.assert_frame_equal(df_expected, df_converted)

    def test_calories_no_conversion(self):
        """Test that cal is NOT converted to Cal (kcal) (as the code is commented out)."""
        df_input = pd.DataFrame({'value': [5000.0], 'unit': ['cal']})
        df_expected = pd.DataFrame({'value': [5000.0], 'unit': ['cal']}) # Expect no change
        df_converted = _convert_healthkit_units(df_input)
        pd.testing.assert_frame_equal(df_expected, df_converted, check_exact=False, rtol=1e-5)

    def test_kilocalories_no_conversion(self):
        """Test that kcal is NOT converted to Cal (kcal) (as the code is commented out)."""
        df_input = pd.DataFrame({'value': [5.0], 'unit': ['kcal']})
        df_expected = pd.DataFrame({'value': [5.0], 'unit': ['kcal']}) # Expect no change
        df_converted = _convert_healthkit_units(df_input)
        pd.testing.assert_frame_equal(df_expected, df_converted)

    def test_feet_to_meters(self):
        """Test conversion from feet (ft) to meters (m)."""
        df_input = pd.DataFrame({'value': [10.0], 'unit': ['ft']})
        df_expected = pd.DataFrame({'value': [10.0 * 0.3048], 'unit': ['m']})
        df_converted = _convert_healthkit_units(df_input)
        pd.testing.assert_frame_equal(df_expected, df_converted, check_exact=False, rtol=1e-5)

    def test_inches_to_meters(self):
        """Test conversion from inches (in) to meters (m)."""
        df_input = pd.DataFrame({'value': [12.0], 'unit': ['in']})
        df_expected = pd.DataFrame({'value': [12.0 * 0.0254], 'unit': ['m']})
        df_converted = _convert_healthkit_units(df_input)
        pd.testing.assert_frame_equal(df_expected, df_converted, check_exact=False, rtol=1e-5)

    def test_cm_to_meters(self):
        """Test conversion from centimeters (cm) to meters (m)."""
        df_input = pd.DataFrame({'value': [175.0], 'unit': ['cm']})
        df_expected = pd.DataFrame({'value': [1.75], 'unit': ['m']})
        df_converted = _convert_healthkit_units(df_input)
        pd.testing.assert_frame_equal(df_expected, df_converted, check_exact=False, rtol=1e-5)

    def test_km_to_meters(self):
        """Test conversion from kilometers (km) to meters (m)."""
        df_input = pd.DataFrame({'value': [5.0], 'unit': ['km']})
        df_expected = pd.DataFrame({'value': [5000.0], 'unit': ['m']})
        df_converted = _convert_healthkit_units(df_input)
        pd.testing.assert_frame_equal(df_expected, df_converted, check_exact=False, rtol=1e-5)

    def test_miles_to_meters(self):
        """Test conversion from miles (mi) to meters (m)."""
        df_input = pd.DataFrame({'value': [1.0], 'unit': ['mi']})
        df_expected = pd.DataFrame({'value': [1609.34], 'unit': ['m']})
        df_converted = _convert_healthkit_units(df_input)
        pd.testing.assert_frame_equal(df_expected, df_converted, check_exact=False, rtol=1e-5)

    def test_pounds_to_kilograms(self):
        """Test conversion from pounds (lb) to kilograms (kg)."""
        df_input = pd.DataFrame({'value': [150.0], 'unit': ['lb']})
        df_expected = pd.DataFrame({'value': [150.0 * 0.45359237], 'unit': ['kg']})
        df_converted = _convert_healthkit_units(df_input)
        pd.testing.assert_frame_equal(df_expected, df_converted, check_exact=False, rtol=1e-5)

    def test_grams_to_kilograms(self):
        """Test conversion from grams (g) to kilograms (kg)."""
        df_input = pd.DataFrame({'value': [750.0], 'unit': ['g']})
        df_expected = pd.DataFrame({'value': [0.75], 'unit': ['kg']})
        df_converted = _convert_healthkit_units(df_input)
        pd.testing.assert_frame_equal(df_expected, df_converted, check_exact=False, rtol=1e-5)

    def test_mph_to_mps(self):
        """Test conversion from miles per hour (mph) to meters per second (m/s)."""
        df_input = pd.DataFrame({'value': [60.0], 'unit': ['mph']})
        df_expected = pd.DataFrame({'value': [60.0 * 0.44704], 'unit': ['m/s']})
        df_converted = _convert_healthkit_units(df_input)
        pd.testing.assert_frame_equal(df_expected, df_converted, check_exact=False, rtol=1e-5)

    def test_floz_to_liters(self):
        """Test conversion from fluid ounces (fl_oz) to liters (L)."""
        df_input = pd.DataFrame({'value': [33.814], 'unit': ['fl_oz']}) # Approx 1 Liter
        df_expected = pd.DataFrame({'value': [33.814 * 0.0295735], 'unit': ['L']})
        df_converted = _convert_healthkit_units(df_input)
        pd.testing.assert_frame_equal(df_expected, df_converted, check_exact=False, rtol=1e-5)

    def test_mixed_units(self):
        """Test a DataFrame with multiple different units, some converted, some not."""
        df_input = pd.DataFrame({
            'value': [1.0, 5000.0, 10.0, 150.0, 60.0],
            'unit': ['count/s', 'cal', 'ft', 'lb', 'mph']
        })
        # count/s and cal are NOT converted anymore
        df_expected = pd.DataFrame({
            'value': [1.0, 5000.0, 10.0 * 0.3048, 150.0 * 0.45359237, 60.0 * 0.44704],
            'unit': ['count/s', 'cal', 'm', 'kg', 'm/s']
        })
        df_converted = _convert_healthkit_units(df_input)
        pd.testing.assert_frame_equal(df_expected, df_converted, check_exact=False, rtol=1e-5)

    def test_unconverted_units(self):
        """Test that units not specified for conversion remain unchanged."""
        df_input = pd.DataFrame({
            'value': [100.0, 70.0, 12.0, 5.0], # Added kcal
            'unit': ['bpm', 'm', 'kg', 'kcal'] # bpm is not converted, m and kg are already target units, kcal is not converted
        })
        df_expected = df_input.copy() # Expect it to be identical
        df_converted = _convert_healthkit_units(df_input)
        pd.testing.assert_frame_equal(df_expected, df_converted)


class TestGenerateSleepMinuteLevelDailyData:

    def test_asleep_interval(self):
        """
        Test a single sleep interval that is marked as asleep.
        This interval is from 02:00 to 02:30 and should set the asleep array
        from minute 120 (2*60) to 150 (2*60+30) to 1.
        """
        df = pd.DataFrame({
            "startTime": [pd.Timestamp("2023-10-09 02:00:00")],
            "endTime":   [pd.Timestamp("2023-10-09 02:30:00")],
            "category value":      ["HKCategoryValueSleepAnalysisAsleep"]
        })
        date = pd.Timestamp("2023-10-09")
        result = _generate_sleep_minute_level_daily_data(df, date)

        expected_asleep = np.zeros(1440, dtype=np.float32)
        expected_inbed = np.zeros(1440, dtype=np.float32)
        expected_asleep[120:150] = 1.0

        np.testing.assert_array_equal(result[0], expected_asleep)
        np.testing.assert_array_equal(result[1], expected_inbed)
        assert result.shape == (2, 1440)

    def test_inbed_interval(self):
        """
        Test a single sleep interval marked as in-bed.
        The interval is from 23:30 to 23:55, so the in-bed array should be
        marked from minute index 1410 (23*60+30) to 1435.
        """
        df = pd.DataFrame({
            "startTime": [pd.Timestamp("2023-10-09 23:30:00")],
            "endTime":   [pd.Timestamp("2023-10-09 23:55:00")],
            "category value":      ["HKCategoryValueSleepAnalysisInBed"]
        })
        date = pd.Timestamp("2023-10-09")
        result = _generate_sleep_minute_level_daily_data(df, date)

        expected_asleep = np.zeros(1440, dtype=np.float32)
        expected_inbed = np.zeros(1440, dtype=np.float32)
        # 23:30 is minute 1410 and 23:55 is minute 1435.
        expected_inbed[1410:1435] = 1.0

        np.testing.assert_array_equal(result[0], expected_asleep)
        np.testing.assert_array_equal(result[1], expected_inbed)

    def test_overlapping_asleep_intervals(self):
        """
        Test two overlapping asleep intervals.
        The first interval is from 01:00 to 01:30 and the second from 01:20 to 01:50.
        The resulting asleep array should be marked from minute 60 (01:00)
        to minute 110 (01:50).
        """
        df = pd.DataFrame({
            "startTime": [pd.Timestamp("2023-10-09 01:00:00"),
                          pd.Timestamp("2023-10-09 01:20:00")],
            "endTime":   [pd.Timestamp("2023-10-09 01:30:00"),
                          pd.Timestamp("2023-10-09 01:50:00")],
            "category value":      ["HKCategoryValueSleepAnalysisAsleep",
                          "HKCategoryValueSleepAnalysisAsleep"]
        })
        date = pd.Timestamp("2023-10-09")
        result = _generate_sleep_minute_level_daily_data(df, date)

        expected_asleep = np.zeros(1440, dtype=np.float32)
        # First interval: 01:00 (minute 60) to 01:30 (minute 90)
        # Second interval: 01:20 (minute 80) to 01:50 (minute 110)
        # The OR operation yields minutes 60-110 as 1.
        expected_asleep[60:110] = 1.0
        expected_inbed = np.zeros(1440, dtype=np.float32)

        np.testing.assert_array_equal(result[0], expected_asleep)
        np.testing.assert_array_equal(result[1], expected_inbed)

    def test_missing_times_skip(self):
        """
        Test that a record with missing startTime or endTime is ignored.
        In this case the output arrays should remain all zeros.
        """
        df = pd.DataFrame({
            "startTime": [pd.Timestamp("2023-10-09 03:00:00")],
            "endTime":   [pd.NaT],
            "category value":      ["HKCategoryValueSleepAnalysisAsleep"]
        })
        date = pd.Timestamp("2023-10-09")
        result = _generate_sleep_minute_level_daily_data(df, date)
        expected = np.zeros((2, 1440), dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_mixed_types(self):
        """
        Test a mix of asleep and in-bed records in one day.
        The first record is asleep from 01:00 to 01:30, and the second is in-bed
        from 01:15 to 01:45. The output arrays should reflect the respective intervals.
        """
        df = pd.DataFrame({
            "startTime": [pd.Timestamp("2023-10-09 01:00:00"),
                          pd.Timestamp("2023-10-09 01:15:00")],
            "endTime":   [pd.Timestamp("2023-10-09 01:30:00"),
                          pd.Timestamp("2023-10-09 01:45:00")],
            "category value":      ["HKCategoryValueSleepAnalysisAsleep",
                          "HKCategoryValueSleepAnalysisInBed"]
        })
        date = pd.Timestamp("2023-10-09")
        result = _generate_sleep_minute_level_daily_data(df, date)

        expected_asleep = np.zeros(1440, dtype=np.float32)
        expected_inbed = np.zeros(1440, dtype=np.float32)
        # For asleep: from 01:00 to 01:30 -> minutes 60 to 90.
        expected_asleep[60:90] = 1.0
        # For in-bed: from 01:15 to 01:45 -> minutes 75 to 105.
        expected_inbed[75:105] = 1.0

        np.testing.assert_array_equal(result[0], expected_asleep)
        np.testing.assert_array_equal(result[1], expected_inbed)

    def test_empty_dataframe_returns_nan_arrays(self):
        """
        Test that an empty DataFrame returns arrays filled with NaN.
        """
        df = pd.DataFrame(columns=["startTime", "endTime", "category value"])
        date = pd.Timestamp("2023-10-09")
        result = _generate_sleep_minute_level_daily_data(df, date)
        expected = np.full((2, 1440), np.nan, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


class TestGenerateMotionMinuteLevelDailyData:
    
    def test_empty_dataframe_returns_nan(self):
        """
        Test that an empty DataFrame returns a (n_types, 1440) array filled with NaNs.
        """
        df = pd.DataFrame(columns=["startTime", "endTime", "activity", "confidence"])
        date = pd.Timestamp("2023-10-09")
        result = _generate_motion_minute_level_daily_data(df, date)
        expected = np.full((len(MotionActivityType), 1440), np.nan, dtype=np.float32)
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_single_event_stationary(self):
        """
        Test a single stationary event.
        The event starts exactly at 00:10 and ends at 00:20, so minutes 10-19 (inclusive) 
        in the "stationary" row (assumed row index corresponding to the enum) should have the constant confidence value.
        """
        date = pd.Timestamp("2023-10-09")
        stationary_start = date + pd.Timedelta(minutes=10)
        stationary_end = date + pd.Timedelta(minutes=20)
        df = pd.DataFrame({
            "startTime": [stationary_start],
            "endTime": [stationary_end],
            "activity": ["stationary"],
            "confidence": [5.0]
        })

        result = _generate_motion_minute_level_daily_data(df, date)
        expected = np.full((len(MotionActivityType), 1440), np.nan, dtype=np.float32)
        # Assuming "stationary" is one of the enum types. Here we set the expected confidence for the corresponding row.
        # Since the event covers minute indices 10 to 19 (inclusive), set these values.
        # (Note: Array slice uses Python's [start:stop] convention where stop is exclusive.)
        stationary_index = list(MotionActivityType).index(MotionActivityType.STATIONARY)
        expected[stationary_index, 10:20] = 5.0
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_non_aligned_event_walking(self):
        """
        Test a walking event that does not align exactly with minute boundaries.
        The event runs from 00:00:30 to 00:01:30.
        For the "walking" row, minute 0 (seconds 0-59) will have partial event coverage (seconds 30-59) 
        and minute 1 (seconds 60-119) will have the remainder; in both cases,
        the average should equal the constant confidence value.
        """
        date = pd.Timestamp("2023-10-09")
        start = date + pd.Timedelta(seconds=30)  # 00:00:30
        end = date + pd.Timedelta(seconds=90)    # 00:01:30
        df = pd.DataFrame({
            "startTime": [start],
            "endTime": [end],
            "activity": ["walking"],
            "confidence": [10.0]
        })

        result = _generate_motion_minute_level_daily_data(df, date)
        expected = np.full((len(MotionActivityType), 1440), np.nan, dtype=np.float32)
        walking_index = list(MotionActivityType).index(MotionActivityType.WALKING)
        # The event covers seconds 30-59 in minute 0 and seconds 60-89 in minute 1.
        expected[walking_index, 0] = 10.0
        expected[walking_index, 1] = 10.0
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_overlapping_events_automotive(self):
        """
        Test overlapping events for the automotive motion type.
        Two overlapping events (with confidence values 20 and 30) over the same time interval 
        yield an average value of 25 for all minutes fully covered by the events.
        """
        date = pd.Timestamp("2023-10-09")
        start = date + pd.Timedelta(minutes=30)
        end = date + pd.Timedelta(minutes=40)
        df = pd.DataFrame({
            "startTime": [start, start],
            "endTime": [end, end],
            "activity": ["automotive", "automotive"],
            "confidence": [20.0, 30.0]
        })

        result = _generate_motion_minute_level_daily_data(df, date)
        expected = np.full((len(MotionActivityType), 1440), np.nan, dtype=np.float32)
        automotive_index = list(MotionActivityType).index(MotionActivityType.AUTOMOTIVE)
        # The event covers minutes 30 to 39 (each minute fully covered) with an average of 25.
        expected[automotive_index, 30:40] = 25.0
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_multiple_types(self):
        """
        Test a DataFrame with events for multiple motion types.
        - A stationary event from 01:00 to 01:10 with confidence 15.0.
        - A running event from 02:00 to 02:15 with confidence 8.0.
        Verify that the respective rows in the final array are correctly populated.
        """
        date = pd.Timestamp("2023-10-09")
        stationary_start = date + pd.Timedelta(hours=1)  # 01:00:00
        stationary_end = date + pd.Timedelta(hours=1, minutes=10)  # 01:10:00
        running_start = date + pd.Timedelta(hours=2)  # 02:00:00
        running_end = date + pd.Timedelta(hours=2, minutes=15)  # 02:15:00
        df = pd.DataFrame({
            "startTime": [stationary_start, running_start],
            "endTime": [stationary_end, running_end],
            "activity": ["stationary", "running"],
            "confidence": [15.0, 8.0]
        })

        result = _generate_motion_minute_level_daily_data(df, date)
        expected = np.full((len(MotionActivityType), 1440), np.nan, dtype=np.float32)
        stationary_index = list(MotionActivityType).index(MotionActivityType.STATIONARY)
        running_index = list(MotionActivityType).index(MotionActivityType.RUNNING)
        # For the stationary event: from 1:00 to 1:10 corresponds to minutes 60 to 69.
        expected[stationary_index, 60:70] = 15.0
        # For the running event: from 2:00 to 2:15 corresponds to minutes 120 to 134.
        expected[running_index, 120:135] = 8.0
        np.testing.assert_allclose(result, expected, equal_nan=True)


class TestGenerateWorkoutMinuteLevelDailyData:

    def test_empty_dataframe_returns_nan_arrays(self):
        """
        If the input DataFrame is empty (but with the required columns),
        the function should return an array of shape 
        (len(HKWorkoutType), 1440) filled with np.nan.
        """
        df = pd.DataFrame(columns=["startTime", "endTime", "workoutType", "recordId"])
        date = pd.Timestamp("2023-10-09")
        result = _generate_workout_minute_level_daily_data(df, date)
        expected = np.full((len(HKWorkoutType), 1440), np.nan, dtype=np.float32)
        assert result.shape == (len(HKWorkoutType), 1440)
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_single_workout_event(self):
        """
        Test a single workout event for a specific workout type.
        For example, a Running workout from 02:00:00 to 02:30:00 should mark the 
        corresponding row (for HKWorkoutActivityTypeRunning) from minute index 120 to 150 as active.
        Other workout type rows should be np.nan.
        """
        date = pd.Timestamp("2023-10-09")
        start = date + pd.Timedelta(hours=2)            # 02:00:00
        end = date + pd.Timedelta(hours=2, minutes=30)    # 02:30:00
        df = pd.DataFrame({
            "startTime": [start],
            "endTime": [end],
            "workoutType": [HKWorkoutType.HKWorkoutActivityTypeRunning.value],
            "recordId": ["r1"]
        })
        result = _generate_workout_minute_level_daily_data(df, date)
        
        # For Running: minutes from 120 (02:00) up to 150 (02:30). (End index is exclusive.)
        expected_running = np.zeros(1440, dtype=np.float32)
        expected_running[120:150] = 1.0
        
        # Build expected output for each workout type row.
        expected = []
        for workout in HKWorkoutType:
            if workout == HKWorkoutType.HKWorkoutActivityTypeRunning:
                expected.append(expected_running)
            else:
                expected.append(np.full(1440, np.nan, dtype=np.float32))
        expected = np.array(expected)
        
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_non_aligned_workout_event(self):
        """
        Test a workout event that is not neatly aligned with minute boundaries.
        For example, a Walking event from 00:00:30 to 00:01:10.
        The start_time (00:00:30) leads to a start minute index of 0,
        and the end_time (00:01:10) yields end minute = ceil(70/60) = 2.
        Thus, minutes 0 and 1 should be flagged.
        """
        date = pd.Timestamp("2023-10-09")
        start = date + pd.Timedelta(seconds=30)   # 00:00:30
        end = date + pd.Timedelta(seconds=90)       # 00:01:30
        df = pd.DataFrame({
            "startTime": [start],
            "endTime": [end],
            "workoutType": [HKWorkoutType.HKWorkoutActivityTypeWalking.value],
            "recordId": ["r2"]
        })
        result = _generate_workout_minute_level_daily_data(df, date)
        
        expected_walking = np.zeros(1440, dtype=np.float32)
        expected_walking[0:2] = 1.0
        
        expected = []
        for workout in HKWorkoutType:
            if workout == HKWorkoutType.HKWorkoutActivityTypeWalking:
                expected.append(expected_walking)
            else:
                expected.append(np.full(1440, np.nan, dtype=np.float32))
        expected = np.array(expected)
        
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_overlapping_workout_events_same_record(self):
        """
        Test that overlapping events from the same workout (i.e. same recordId)
        are merged correctly.
        For example, two Cycling records (with recordId "r3"):
          - One from 03:00:00 to 03:15:00.
          - Another from 03:10:00 to 03:30:00.
        Their union should mark minutes from 03:00 (180) to 03:30 (210) as active.
        """
        date = pd.Timestamp("2023-10-09")
        start1 = date + pd.Timedelta(hours=3)               # 03:00:00
        end1 = date + pd.Timedelta(hours=3, minutes=15)       # 03:15:00
        start2 = date + pd.Timedelta(hours=3, minutes=10)     # 03:10:00
        end2 = date + pd.Timedelta(hours=3, minutes=30)       # 03:30:00
        
        df = pd.DataFrame({
            "startTime": [start1, start2],
            "endTime": [end1, end2],
            "workoutType": [HKWorkoutType.HKWorkoutActivityTypeCycling.value,
                            HKWorkoutType.HKWorkoutActivityTypeCycling.value],
            "recordId": ["r3", "r3"]
        })
        result = _generate_workout_minute_level_daily_data(df, date)
        
        expected_cycling = np.zeros(1440, dtype=np.float32)
        # Union: 03:00:00 (180 min) to 03:30:00 (210 min)
        expected_cycling[180:210] = 1.0
        
        expected = []
        for workout in HKWorkoutType:
            if workout == HKWorkoutType.HKWorkoutActivityTypeCycling:
                expected.append(expected_cycling)
            else:
                expected.append(np.full(1440, np.nan, dtype=np.float32))
        expected = np.array(expected)
        
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_event_outside_day(self):
        """
        Test a workout event that is completely outside the target day.
        If a record exists but its times do not overlap with the given day,
        then the function processes the record (i.e. df is non-empty) and the 
        resulting indicator stays at 0. For that workout type, the output should be a
        0-filled array rather than np.nan.
        """
        date = pd.Timestamp("2023-10-09")
        # Create an event for TraditionalStrengthTraining on 2023-10-10.
        start = pd.Timestamp("2023-10-10 00:10:00")
        end = pd.Timestamp("2023-10-10 00:20:00")
        df = pd.DataFrame({
            "startTime": [start],
            "endTime": [end],
            "workoutType": [HKWorkoutType.HKWorkoutActivityTypeTraditionalStrengthTraining.value],
            "recordId": ["r4"]
        })
        result = _generate_workout_minute_level_daily_data(df, date)
        
        expected_traditional = np.zeros(1440, dtype=np.float32)
        expected = []
        for workout in HKWorkoutType:
            if workout == HKWorkoutType.HKWorkoutActivityTypeTraditionalStrengthTraining:
                expected.append(expected_traditional)
            else:
                expected.append(np.full(1440, np.nan, dtype=np.float32))
        expected = np.array(expected)
        
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_multiple_workout_types(self):
        """
        Test multiple workout events for different workout types in one day.
        For example, one Running event from 02:00:00 to 02:10:00 and one Yoga event from 20:00:00 to 20:30:00.
        The corresponding rows should be marked with the active minutes while rows for other types remain np.nan.
        """
        date = pd.Timestamp("2023-10-09")
        running_start = date + pd.Timedelta(hours=2)            # 02:00:00
        running_end = date + pd.Timedelta(hours=2, minutes=10)    # 02:10:00
        yoga_start = date + pd.Timedelta(hours=20)               # 20:00:00
        yoga_end = date + pd.Timedelta(hours=20, minutes=30)       # 20:30:00
        
        df = pd.DataFrame({
            "startTime": [running_start, yoga_start],
            "endTime": [running_end, yoga_end],
            "workoutType": [HKWorkoutType.HKWorkoutActivityTypeRunning.value,
                            HKWorkoutType.HKWorkoutActivityTypeYoga.value],
            "recordId": ["r5", "r6"]
        })
        result = _generate_workout_minute_level_daily_data(df, date)
        
        expected_running = np.zeros(1440, dtype=np.float32)
        expected_running[120:130] = 1.0  # 02:00 (120) to 02:10 (130)
        
        expected_yoga = np.zeros(1440, dtype=np.float32)
        expected_yoga[1200:1230] = 1.0   # 20:00 (1200) to 20:30 (1230)
        
        expected = []
        for workout in HKWorkoutType:
            if workout == HKWorkoutType.HKWorkoutActivityTypeRunning:
                expected.append(expected_running)
            elif workout == HKWorkoutType.HKWorkoutActivityTypeYoga:
                expected.append(expected_yoga)
            else:
                expected.append(np.full(1440, np.nan, dtype=np.float32))
        expected = np.array(expected)
        
        np.testing.assert_allclose(result, expected, equal_nan=True)


