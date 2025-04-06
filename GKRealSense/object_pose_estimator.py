"""
Object Pose Estimator Module
This module provides functionality to estimate the pose of detected objects using RealSense depth data.
It converts the list of DetectedObject into a list of ObjectWithPosition
"""

from dataclasses import dataclass
from realsense_handler import rs
from object_detector import DetectedObject, ObjectType
import numpy as np


@dataclass(slots=True)
class ObjectWithPosition:
    """Dataclass to store object with position"""

    object_type: ObjectType
    position: np.ndarray = np.zeros((1, 3))
    variance: np.ndarray = np.zeros((1, 3))


class ObjectPoseEstimator:
    def __init__(
        self,
        depth_intrinsics: rs.intrinsics,
        color_intrinsics: rs.intrinsics,
        ir_intrinsics: rs.intrinsics,
        transform_camera_to_robot: np.ndarray = np.eye(4),
    ):
        self.depth_intrinsics: rs.intrinsics = depth_intrinsics
        self.color_intrinsics: rs.intrinsics = color_intrinsics
        self.ir_intrinsics: rs.intrinsics = ir_intrinsics
        self.transform_camera_to_robot: np.ndarray = transform_camera_to_robot

    def _compute_variance(self, position: np.ndarray) -> np.ndarray:
        # TODO: Implement the computation of the variance
        return np.array([[0.25, 0.25, 0.25]])

    def _map_detected_object_to_object_with_position(
        self, camera_to_world: np.ndarray, depth_frame: rs.depth_frame, detected_object: DetectedObject
    ) -> ObjectWithPosition:
        # Improvement point: get the average from the pixels in the neighbourhood of the center point
        x_point = int((detected_object.box[0] + detected_object.box[2]) / 2)
        y_point = int((detected_object.box[1] + detected_object.box[3]) / 2)
        z_distance: float = depth_frame.get_distance(x_point, y_point)
        x_distance, y_distance, _ = rs.rs2_deproject_pixel_to_point(
            self.depth_intrinsics,
            [x_point, y_point],
            z_distance,
        )

        position = np.array([[x_distance, y_distance, z_distance, 1]]).transpose()
        position = np.dot(camera_to_world, position)
        position = position[:3].flatten()
        return ObjectWithPosition(
            detected_object.type,
            position,
            self._compute_variance(position),
        )

    def estimate_position(
        self, robot_to_world: np.ndarray, depth_frame: rs.depth_frame, detected_objects: list[DetectedObject]
    ) -> list[ObjectWithPosition]:
        """Estimate position of detected objects"""
        camera_to_world: np.ndarray = np.dot(robot_to_world, self.transform_camera_to_robot)

        detected_objects_with_position: list[ObjectWithPosition] = [
            self._map_detected_object_to_object_with_position(
                camera_to_world, depth_frame, detected_object
            )
            for detected_object in detected_objects
        ]

        return detected_objects_with_position
