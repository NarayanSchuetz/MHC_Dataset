import enum

class FileType(enum.Enum):
    """
    File types.
    """
    HEALTHKIT = "healthkit"
    WORKOUT = "workout"
    SLEEP = "sleep"
    MOTION = "motion"


class HKQuantityType(enum.Enum):
    """
    The types of data that will be retrieved from our HealthKit files.
    """
    HKQuantityTypeIdentifierStepCount = "HKQuantityTypeIdentifierStepCount"
    HKQuantityTypeIdentifierActiveEnergyBurned = "HKQuantityTypeIdentifierActiveEnergyBurned"
    HKQuantityTypeIdentifierDistanceWalkingRunning = "HKQuantityTypeIdentifierDistanceWalkingRunning"
    HKQuantityTypeIdentifierDistanceCycling = "HKQuantityTypeIdentifierDistanceCycling"
    HKQuantityTypeIdentifierAppleStandTime = "HKQuantityTypeIdentifierAppleStandTime"
    HKQuantityTypeIdentifierHeartRate = "HKQuantityTypeIdentifierHeartRate"


class HKWorkoutType(enum.Enum):
    """
    The types of workouts that can be recorded in HealthKit.
    """
    HKWorkoutActivityTypeWalking = "HKWorkoutActivityTypeWalking"
    HKWorkoutActivityTypeCycling = "HKWorkoutActivityTypeCycling"
    HKWorkoutActivityTypeRunning = "HKWorkoutActivityTypeRunning"
    HKWorkoutActivityTypeOther = "HKWorkoutActivityTypeOther"
    HKWorkoutActivityTypeMixedMetabolicCardioTraining = "HKWorkoutActivityTypeMixedMetabolicCardioTraining"
    HKWorkoutActivityTypeTraditionalStrengthTraining = "HKWorkoutActivityTypeTraditionalStrengthTraining"
    HKWorkoutActivityTypeElliptical = "HKWorkoutActivityTypeElliptical"
    HKWorkoutActivityTypeHighIntensityIntervalTraining = "HKWorkoutActivityTypeHighIntensityIntervalTraining"
    HKWorkoutActivityTypeFunctionalStrengthTraining = "HKWorkoutActivityTypeFunctionalStrengthTraining"
    HKWorkoutActivityTypeYoga = "HKWorkoutActivityTypeYoga"


class MotionActivityType(enum.Enum):
    """
    The types of motion activities that can be recorded in HealthKit.
    """
    STATIONARY = "stationary"
    WALKING = "walking"
    RUNNING = "running"
    AUTOMOTIVE = "automotive"
    CYCLING = "cycling"
    UNKNOWN = "not available"
