import math
import time
import numpy as np
from dataclasses import dataclass
from filterpy.kalman import KalmanFilter
from object_detector import ObjectType
from object_pose_estimator import ObjectWithPosition
from scipy.optimize import linear_sum_assignment



@dataclass(slots=True)
class ObjectStatus:
    object_id: int
    object_type: ObjectType


@dataclass(slots=True)
class KalmanFilterStatus:
    trustiness: float = 0.0  # How much we trust in this object tracking (0 ~ 1)
    updated_in_the_last_cycle: bool = False
    cycles_without_update: int = 0
    cycles_updates: int = 0  # Number of cycles this object has been updated in sequence
    last_update_timestamp: int = 0
    cycles: int = 1  # Number of cycle this object exist

    def update_trustiness(self, trustiness_coefficient: float):
        MAX_DELTA = 0.07
        trustiness_rate = max(-1, min(1, trustiness_coefficient))
        if trustiness_rate < 0:
            delta = MAX_DELTA * trustiness_rate * math.exp(-1.5 * self.trustiness)
        else:
            delta = MAX_DELTA * trustiness_rate * math.exp(-1.5 * (1.0 - self.trustiness))
        self.trustiness = max(min(self.trustiness + delta, 1.0), 0.0)


@dataclass(slots=True)
class TrackedObject:
    objectStatus: ObjectStatus
    kalmanFilter: KalmanFilter
    kalmanFilterStatus: KalmanFilterStatus


class ObjectTracker:
    INVALID_OBJECT_DISTANCE_THRESHOLD: float = 0.05

    def __init__(self):
        self.tracked_objects: list[TrackedObject] = []
        self.last_timestamp: int = 0
        self.object_id_counter: int = 0

    def track(
        self,
        input_candidates: list[tuple[int, list[ObjectWithPosition]]],
    ) -> list[TrackedObject]:
        # 0. Reset the status of the filters
        for tracked_object in self.tracked_objects:
            tracked_object.kalmanFilterStatus.updated_in_the_last_cycle = False

        for timestamp, sensor_data in input_candidates:
            if (timestamp - self.last_timestamp > 500000000) and len(self.tracked_objects) > 0:
                # If the time difference is too big, we need to reset the filter
                self.tracked_objects = []
            self._run_filter_for_sensor(timestamp, sensor_data)
            self.last_timestamp = timestamp

        # Update trustiness for objects that were not updated in the last cycle
        to_delete = []
        for index, tracked_object in enumerate(self.tracked_objects):
            if not tracked_object.kalmanFilterStatus.updated_in_the_last_cycle:
                tracked_object.kalmanFilterStatus.update_trustiness(-1)
                tracked_object.kalmanFilterStatus.cycles_without_update += 1
                tracked_object.kalmanFilterStatus.cycles_updates -= 1

            if (
                tracked_object.kalmanFilterStatus.trustiness < 0.15
                and tracked_object.kalmanFilterStatus.cycles_updates <= -3
            ):
                to_delete.append(index)

        # Clear the untrustworthy objects
        to_delete = sorted(to_delete, reverse=True)
        for index in to_delete:
            self.tracked_objects.pop(index)

        return self.tracked_objects

    def _run_filter_for_sensor(
        self, timestamp, measurements: list[ObjectWithPosition]
    ) -> None:
        # 0. Check if we have any tracked object
        len_tracked_objects: int = len(self.tracked_objects)
        if len_tracked_objects == 0:
            # If we don't have any tracked object, we need to create one for each measurement
            for measurement in measurements:
                if measurement.position[0] < self.INVALID_OBJECT_DISTANCE_THRESHOLD:
                    # Invalid measurement, skip it
                    continue

                self._create_new_tracked_object(
                    measurement,
                    timestamp=timestamp,
                    trustiness=0.15
                )
            return

        # 1. Update ball status predictions (We assume that all measurement from the same sensor has the same timestamp)
        self._update_kalman_predictions(timestamp)

        ## Association Step:
        # 2.1 We need to construct a matrix table with the association between the measurements and the tracked objects 
        # to use the Hungarian algorithm to find the best association
        len_measurements: int = len(measurements)
        matrix_size: int = max(len_tracked_objects, len_measurements) # Tracked Objects ( Row ) x Measurements ( Column )
        cost_matrix: np.ndarray = np.full((matrix_size, matrix_size), 1e9)

        # 2.2 For each measurement we compute the cost of association with the tracked objects
        for index, measurement in enumerate(measurements):
            if measurement.position[0] < self.INVALID_OBJECT_DISTANCE_THRESHOLD:
                # Invalid measurement, keep the cost at infinity
                continue

            association_result: np.ndarray = self._make_association(
                measurement.object_type, measurement.position, measurement.variance
            )
            cost_matrix[:len_tracked_objects, index] = association_result.flatten()

        # 2.3 Apply the Hungarian algorithm to find the best association
        objects_association_indexes, measurement_association_indexes = linear_sum_assignment(cost_matrix)

        ## Apply measurement to the Kalman Filter
        for index in range(matrix_size):
            measurement_association_index = measurement_association_indexes[index]
            if measurement_association_index >= len_measurements:
                continue
            
            # 3.1 Get the association for this measurement
            associated_object_index = objects_association_indexes[index]
            if cost_matrix[associated_object_index, measurement_association_index] == 1e9:
                # This measurement is not associated with any tracked object
                # So we create a new tracked object
                if measurements[measurement_association_index].position[0] >= self.INVALID_OBJECT_DISTANCE_THRESHOLD:
                    self._create_new_tracked_object(
                        measurements[measurement_association_index],
                        timestamp=timestamp,
                        trustiness=0.15,
                    )
                continue
          
            # 3.2 Apply the measurement into the Kalman filter if associated
            self._apply_measurement_to_kalman_filter(
                    association_index=associated_object_index,
                    association_weight=cost_matrix[associated_object_index, measurement_association_index],
                    position=measurements[measurement_association_index].position,
                    variance=measurements[measurement_association_index].variance,
                    timestamp=timestamp,
                )
                

    def _update_kalman_predictions(self, timestamp_ns: int) -> None:
        # Using the same model for every tracked object, maybe in the future it has a specific model
        # for each tracker so we can use other information to have a better prediction
        # Such as to be close to a robot ou hit the floor.

        matrix_f_base = np.eye(9)
        ACCELERATION_VARIANCE = 0.01
        matrix_q_base = ACCELERATION_VARIANCE * np.block([
            [np.diag([0.25, 0.25, 0.25]), np.diag([0.50, 0.50, 0.50]), np.diag([0.50, 0.50, 0.50])],
            [np.diag([0.50, 0.50, 0.50]), np.diag([1.00, 1.00, 1.00]), np.diag([1.00, 1.00, 1.00])],
            [np.diag([0.50, 0.50, 0.50]), np.diag([1.00, 1.00, 1.00]), np.diag([1000, 1000, 1000])],
        ])

        for tracked_object in self.tracked_objects:
            delta_time_ns = (
                timestamp_ns - tracked_object.kalmanFilterStatus.last_update_timestamp
            )
            delta_time_s = delta_time_ns / 1e9

            use_gravity = tracked_object.kalmanFilter.x[2] > 0.2
            matrix_f = matrix_f_base.copy()
            self._update_matrix_f(matrix_f, use_gravity, delta_time_s)
            matrix_q = matrix_q_base.copy()
            self._update_matrix_q(matrix_q, tracked_object.kalmanFilter.x[3:6], delta_time_s)

            tracked_object.kalmanFilter.predict(F=matrix_f, Q=matrix_q)
            if tracked_object.kalmanFilter.x[2] < 0:
                tracked_object.kalmanFilter.x[2] = 0
                tracked_object.kalmanFilter.x[5] = -tracked_object.kalmanFilter.x[5] * 0.5
                tracked_object.kalmanFilter.x[8] = abs(tracked_object.kalmanFilter.x[8]) * 0.5
            tracked_object.kalmanFilterStatus.last_update_timestamp = timestamp_ns


    def _make_association(
        self, measurement_type: ObjectType, mean: np.ndarray, covariance: np.ndarray
    ) -> list[float]:
        STILL_DISTANCE_THRESHOLD = 1.0
        association_costs = np.full((len(self.tracked_objects), 1), 1e9)
        for index, tracked_object in enumerate(self.tracked_objects):
            if measurement_type != tracked_object.objectStatus.object_type:
                continue

            pos_diff = mean[:3] - tracked_object.kalmanFilter.x[:3]
            distance_diff = np.linalg.norm(pos_diff)
            if distance_diff > STILL_DISTANCE_THRESHOLD:
                continue
            
            mahlanobis_distance = np.dot(np.dot(pos_diff, np.linalg.inv(tracked_object.kalmanFilter.P[0:3, 0:3])), pos_diff.transpose())
            low_trustiness_punishment = (2.0 - tracked_object.kalmanFilterStatus.trustiness) ** 2
            association_costs[index, 0] = mahlanobis_distance * low_trustiness_punishment

        return association_costs

    def _apply_measurement_to_kalman_filter(
        self,
        association_index: int,
        association_weight: float,
        position: np.ndarray,
        variance: np.ndarray,
        timestamp: int,
    ) -> None:

        covariance = np.diag(variance[0])
        self.tracked_objects[association_index].kalmanFilter.update(
            z=position, R=covariance
        )

        trustiness_coefficient = 1.0
        if variance[0][0] > 0.25:
            trustiness_coefficient *= 0.65

        self.tracked_objects[association_index].kalmanFilterStatus.update_trustiness(
            trustiness_coefficient
        )

        self.tracked_objects[
            association_index
        ].kalmanFilterStatus.updated_in_the_last_cycle = True

        self.tracked_objects[
            association_index
        ].kalmanFilterStatus.cycles_without_update = 0

        if self.tracked_objects[association_index].kalmanFilterStatus.cycles_updates < 5:
            self.tracked_objects[association_index].kalmanFilterStatus.cycles_updates += 1

        self.tracked_objects[
            association_index
        ].kalmanFilterStatus.last_update_timestamp = timestamp

    def _create_new_tracked_object(
        self,
        object: ObjectWithPosition,
        timestamp: int,
        trustiness: int | None = None,
    ) -> None:
        self.object_id_counter += 1
        new_object_id = self.object_id_counter
        # 1.0 Create Kalman Filter object
        kalmanFilter = KalmanFilter(dim_x=9, dim_z=3)

        # 1.1 Create first measurements
        # 1.1.1 Add first position measurement
        kalmanFilter.x = np.array(
            [
                object.position[0],
                object.position[1],
                object.position[2],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        # 1.1.1 Add first covariances
        position_covariance = np.diag(object.variance[0])
        velocity_covariance = np.diag(100 * object.variance[0])
        acceleration_covariance = np.diag(10000 * object.variance[0])
        kalmanFilter.P = np.block(
            [
                [position_covariance, np.zeros((3, 3)), np.zeros((3, 3))],
                [np.zeros((3, 3)), velocity_covariance, np.zeros((3, 3))],
                [np.zeros((3, 3)), np.zeros((3, 3)), acceleration_covariance],
            ]
        )

        # 1.2 Add H Model
        kalmanFilter.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
            ]
        )

        # 2.1 Create Kalman Filter Status
        if trustiness is None:
            trustiness = self._compute_trustiness_from_covariance(covariance=position_covariance)

        kalman_filters_status = KalmanFilterStatus(
            trustiness=trustiness,
            last_update_timestamp=timestamp,
            updated_in_the_last_cycle=True,
        )

        # 3. Create the TrackedObject
        self.tracked_objects.append(
            TrackedObject(
                objectStatus=ObjectStatus(
                    object_id=new_object_id,
                    object_type=object.object_type,
                ),
                kalmanFilter=kalmanFilter,
                kalmanFilterStatus=kalman_filters_status,
            )
        )

    def _compute_trustiness_from_covariance(self, covariance: np.ndarray) -> float:
        MAX_TRUSTNESS = 0.5
        covariance_norm_diag = np.linalg.norm(np.diag(covariance))

        if covariance_norm_diag < 0.01:  # Zero
            return MAX_TRUSTNESS

        return MAX_TRUSTNESS * min((174 / covariance_norm_diag), 1.0)

    def _update_matrix_f(self, matrix_f: np.ndarray, use_gravity: bool, dt_s: float):
        matrix_f[0, 3] = dt_s
        matrix_f[1, 4] = dt_s
        matrix_f[2, 5] = dt_s

        matrix_f[3, 6] = dt_s
        matrix_f[4, 7] = dt_s
        matrix_f[5, 8] = dt_s

        matrix_f[6, 0] = dt_s**2 / 2
        matrix_f[7, 1] = dt_s**2 / 2
        matrix_f[8, 2] = dt_s**2 / 2

        if use_gravity:
            matrix_f[8, 5] = -9.81 * dt_s
    
    def _update_matrix_q(self, matrix_q: np.ndarray, velocity: np.ndarray, dt_s: float):
        dt2_s: float = dt_s ** 2
        dt3_s: float = dt_s ** 3
        dt4_s: float = dt_s ** 4

        speed = np.linalg.norm(velocity)
        if speed > 0.3:
            direction = velocity / speed
            ddT = np.outer(direction, direction)
            I = np.eye(3)
            matrix_q[0:3, 0:3] = (2.0 * dt4_s) * ddT + (0.5 * dt4_s) * (I - ddT)
        else:
            matrix_q[0, 0] *= dt4_s
            matrix_q[1, 1] *= dt4_s
            matrix_q[2, 2] *= dt4_s

        matrix_q[3, 3] *= dt4_s
        matrix_q[4, 4] *= dt4_s
        matrix_q[5, 5] *= dt4_s

        matrix_q[0, 3] *= dt3_s
        matrix_q[1, 4] *= dt3_s
        matrix_q[2, 5] *= dt3_s
        matrix_q[3, 0] *= dt3_s
        matrix_q[4, 1] *= dt3_s
        matrix_q[5, 2] *= dt3_s

        matrix_q[0, 6] *= dt2_s
        matrix_q[1, 7] *= dt2_s
        matrix_q[2, 8] *= dt2_s
        matrix_q[6, 0] *= dt2_s
        matrix_q[7, 1] *= dt2_s
        matrix_q[8, 2] *= dt2_s


if __name__ == "__main__":
    tracker = ObjectTracker()
    start_perf_counter = time.perf_counter()
    first_detection: list[ObjectWithPosition] = [
        ObjectWithPosition(
            ObjectType.BALL, np.array([[1, 2, 3]]), np.array([[0.1, 0.1, 0.1]])
        ),
    ]

    start_time = time.time_ns()
    tracker.track([(start_time, first_detection)])

    second_detection: list[ObjectWithPosition] = [
        ObjectWithPosition(
            ObjectType.BALL, np.array([[1.2, 2.2, 3.3]]), np.array([[0.1, 0.1, 0.1]])
        ),
    ]

    second_time = start_time + int(1 * 1e9)
    result = tracker.track([(second_time, second_detection)])

    third_detection: list[ObjectWithPosition] = [
        ObjectWithPosition(
            ObjectType.BALL,
            np.array([[1.43, 2.4, 3.6]]),
            np.array([[0.1, 0.1, 0.1]]),
        ),
    ]

    third_time = second_time + int(1 * 1e9)
    tracker.track([(third_time, third_detection)])

    forth_detection: list[ObjectWithPosition] = [
        ObjectWithPosition(
            ObjectType.BALL,
            np.array([[1.43, 2.8, 3.6]]),
            np.array([[0.1, 0.1, 0.1]]),
        ),
    ]

    forth_time = third_time + int(1 * 1e9)
    result = tracker.track([(forth_time, forth_detection)])
    print(f"Time: {1000 * (time.perf_counter() - start_perf_counter)} ms")
    print(result)
