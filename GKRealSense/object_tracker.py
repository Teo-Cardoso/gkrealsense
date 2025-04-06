import time
import numpy as np
from dataclasses import dataclass
from filterpy.kalman import KalmanFilter
from object_detector import ObjectType
from object_pose_estimator import ObjectWithPosition


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
        trustiness_rate = trustiness_coefficient
        self.trustiness = max(min(self.trustiness + trustiness_rate, 1.0), 0.0)


@dataclass(slots=True)
class TrackedObject:
    objectStatus: ObjectStatus
    kalmanFilter: KalmanFilter
    kalmanFilterStatus: KalmanFilterStatus


class ObjectTracker:
    def __init__(self):
        self.tracked_objects: list[TrackedObject] = []
        self.last_timestamp: int = 0
        self.initialized: bool = False
        self.object_id_counter: int = 0

    def track(
        self,
        input_candidates: list[tuple[int, list[ObjectWithPosition]]],
    ) -> list[TrackedObject]:
        if not self.initialized:
            self._initialize_filter(input_candidates)
        else:
            # 0. Reset the status of the filters
            for tracked_object in self.tracked_objects:
                tracked_object.kalmanFilterStatus.updated_in_the_last_cycle = False

            for timestamp, sensor_data in input_candidates:
                self._run_filter_for_sensor(timestamp, sensor_data)

        # Update trustiness for objects that were not updated in the last cycle
        to_delete = []
        for index, tracked_object in enumerate(self.tracked_objects):
            if not tracked_object.kalmanFilterStatus.updated_in_the_last_cycle:
                tracked_object.kalmanFilterStatus.update_trustiness(-0.05)
                tracked_object.kalmanFilterStatus.cycles_without_update += 1
                tracked_object.kalmanFilterStatus.cycles_updates -= 1

            if (
                tracked_object.kalmanFilterStatus.trustiness < 0.15
                and tracked_object.kalmanFilterStatus.cycles_updates <= -3
            ):
                to_delete.append(index)

        # Clear the untrustworthy objects
        # TO FIX: This is not the best way to delete objects from the list
        for index in to_delete:
            self.tracked_objects.pop(index)

        return self.tracked_objects

    def _run_filter_for_sensor(
        self, timestamp, measurements: list[ObjectWithPosition]
    ) -> None:
        # 1. Update ball status predictions (We assume that all measurement from the same sensor has the same timestamp)
        self._update_kalman_predictions(timestamp)

        # 2. For each measurement
        for measurement in measurements:
            if measurement.position[0] == 0:
                continue

            # 2.1 Make the associations between measurement and tracked balls
            (
                association_result,
                association_weight,
                association_index,
            ) = self._make_association(
                measurement.object_type, measurement.position, measurement.variance
            )

            # 2.2 Apply the measurement into the kalman filter if associated
            if association_result:
                self._apply_measurement_to_kalman_filter(
                    association_index=association_index,
                    association_weight=association_weight,
                    position=measurement.position,
                    variance=measurement.variance,
                    timestamp=timestamp,
                )
            # 2.3 Create a new tracking object if not associated
            else:
                self._create_new_tracking_object(
                    measurement,
                    timestamp=timestamp,
                )

    def _update_kalman_predictions(self, timestamp_ns: int) -> None:
        # Using the same model for every tracked object, maybe in the future it has a specific model
        # for each tracker so we can use other information to have a better prediction
        # Such as to be close to a robot ou hit the floor.

        matrix_f = np.eye(9)
        ACCELERATION_VARIANCE = 0.1
        matrix_q_base = ACCELERATION_VARIANCE * np.block([
            [np.diag([0.25, 0.25, 0.25]), np.diag([0.25, 0.25, 0.25]), np.diag([0.25, 0.25, 0.25])],
            [np.diag([0.25, 0.25, 0.25]), np.eye(3), np.eye(3)],
            [np.diag([0.25, 0.25, 0.25]), np.eye(3), np.eye(3)]
        ])
        

        for tracked_object in self.tracked_objects:
            delta_time_ns = (
                timestamp_ns - tracked_object.kalmanFilterStatus.last_update_timestamp
            )
            delta_time_s = delta_time_ns / 1e9

            use_gravity = tracked_object.kalmanFilter.x[2] > 0.2
            self._update_matrix_f(matrix_f, use_gravity, delta_time_s)
            self._update_matrix_q(matrix_q_base, delta_time_s)
            matrix_q = matrix_q_base

            tracked_object.kalmanFilter.predict(F=matrix_f, Q=matrix_q)
            if tracked_object.kalmanFilter.x[2] < 0:
                tracked_object.kalmanFilter.x[2] = 0
                tracked_object.kalmanFilter.x[5] = -tracked_object.kalmanFilter.x[5] * 0.5
                tracked_object.kalmanFilter.x[8] = abs(tracked_object.kalmanFilter.x[8]) * 0.5
            tracked_object.kalmanFilterStatus.last_update_timestamp = timestamp_ns

    def _initialize_filter(
        self, sensors_inputs: list[tuple[int, list[ObjectWithPosition]]]
    ) -> None:
        self.initialized = True
        for timestamp, sensor_measurements in sensors_inputs:
            for measurement in sensor_measurements:
                self._create_new_tracking_object(
                    measurement,
                    timestamp,
                    trustiness=0.15,
                )

    def _make_association(
        self, measurement_type: ObjectType, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[bool, float, int]:
        # Perform the association to found out each object is be measurand by the measurement.
        # Now it is done only a simple distance check, but this can be improved in the future.
        # Add new features, as color, size, velocity and others.
        # We are not using the covariance now, only the mean
        MOVING_DISTANCE_THRESHOLD = 0.6
        STILL_DISTANCE_THRESHOLD = 0.4
        association_result: tuple[bool, float, int] = (False, 0.0, 0)
        for index, tracked_object in enumerate(self.tracked_objects):
            if measurement_type != tracked_object.objectStatus.object_type:
                continue
            
            pos_diff = mean[:3] - tracked_object.kalmanFilter.x[:3]
            distance_diff = np.linalg.norm(pos_diff)
            velocity = tracked_object.kalmanFilter.x[3:6]
            speed = np.linalg.norm(velocity)
            if speed > 0.2:
                direction = velocity / speed
                threshold = MOVING_DISTANCE_THRESHOLD * abs(direction)
                
                check_x = pos_diff[0] > max(threshold[0], STILL_DISTANCE_THRESHOLD)
                check_y = pos_diff[1] > max(threshold[1], STILL_DISTANCE_THRESHOLD)
                check_z = pos_diff[2] > max(threshold[2], STILL_DISTANCE_THRESHOLD)
                if check_x or check_y or check_z:
                    continue
                
            else:
                # Se o objeto está parado, usa distância normal
                check_x = pos_diff[0] > 3.0 * STILL_DISTANCE_THRESHOLD
                check_y = pos_diff[1] > STILL_DISTANCE_THRESHOLD
                check_z = pos_diff[2] > 0.75 * STILL_DISTANCE_THRESHOLD
                if check_x or check_y or check_z:
                    continue

            if not association_result[0] or association_result[1] > distance_diff:
                # Update the association result
                association_result = (True, distance_diff, index)

        return association_result

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

        # association_weight  # This need to be better planned
        trustiness_coefficient = 0.1

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

    def _create_new_tracking_object(
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
    
    def _update_matrix_q(self, matrix_q: np.ndarray, dt_s: float):
        dt2_s: float = dt_s ** 2
        dt3_s: float = dt_s ** 3
        dt4_s: float = dt_s ** 4

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
