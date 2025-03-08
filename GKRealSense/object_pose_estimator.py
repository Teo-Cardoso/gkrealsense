from dataclasses import dataclass
from realsense_handler import rs
from object_detector import DetectedObject, ObjectType
import numpy as np


@dataclass(slot=True)
class ObjectWithPose:
    """Dataclass to store object with pose"""
    object_type: ObjectType
    pose: np.ndarray = np.zeros((1, 3))


class ObjectPoseEstimator:
    def __init__(self, depth_intrinsics: rs.intrinsics, color_intrinsics: rs.intrinsics, ir_intrinsics: rs.intrinsics):
        self.depth_intrinsics: rs.intrinsics = depth_intrinsics
        self.color_intrinsics: rs.intrinsics = color_intrinsics
        self.ir_intrinsics: rs.intrinsics = ir_intrinsics

    def _map_detected_object_to_object_with_pose(self, depth_frame: rs.depth_frame,  detected_object: DetectedObject) -> ObjectWithPose:
        # Improvement point: get the average from the pixels in the neighbourhood of the center point
        z_distance: float = depth_frame.get_distance(
            detected_object.box[0], detected_object.box[1])
        x_distance, y_distance = rs.rs2_deproject_pixel_to_point(
            self.depth_intrinsics, [detected_object.box[0], detected_object.box[1]], z_distance)
        return ObjectWithPose(detected_object.type, np.array([x_distance, y_distance, z_distance], shape=(1, 3)))

    def estimate_pose(self, depth_frame: rs.depth_frame, detected_objects: list[DetectedObject]) -> list[ObjectWithPose]:
        """Estimate pose of detected objects"""
        detected_objects_with_pose: list[ObjectWithPose] = [
            self._map_detected_object_to_object_with_pose(  depth_frame, detected_object) for detected_object in detected_objects]

        return detected_objects_with_pose
