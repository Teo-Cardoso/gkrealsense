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
        """Compute variance of the position"""
        return np.array([[min(0.1 * position[0], 0.45), 0.25, 0.25]])

    def _map_detected_object_to_object_with_position(
        self, camera_to_world: np.ndarray, depth_frame: rs.depth_frame, detected_object: DetectedObject
    ) -> ObjectWithPosition:
        # Improvement point: get the average from the pixels in the neighbourhood of the center point
        x_point = int((detected_object.box[0] + detected_object.box[2]) / 2)
        y_point = int((detected_object.box[1] + detected_object.box[3]) / 2)
        neighbourhood: int = 2

        distances: list[list[float]] = [[], [], []]
        for dx in range(-neighbourhood, neighbourhood + 1):
            for dy in range(-neighbourhood, neighbourhood + 1):
                
                x_point_neigh = int(x_point + dx)
                y_point_neigh = int(y_point + dy)
                if (
                    x_point_neigh < 0
                    or x_point_neigh >= self.depth_intrinsics.width
                    or y_point_neigh < 0
                    or y_point_neigh >= self.depth_intrinsics.height
                ):
                    continue

                z_distance: float = depth_frame.get_distance(x_point_neigh, y_point_neigh)
                if z_distance < 0:
                    continue
                
                x_distance, y_distance, _ = rs.rs2_deproject_pixel_to_point(
                    self.depth_intrinsics,
                    [x_point, y_point],
                    z_distance,
                )

                distances[0].append(x_distance)
                distances[1].append(y_distance)
                distances[2].append(z_distance)        

        if len(distances[0]) == 0:
            x_distance = 0.0
            y_distance = 0.0
            z_distance = 0.0
        else:
            x_distance = np.mean(distances[0])
            y_distance = np.mean(distances[1])
            z_distance = np.mean(distances[2])

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
