from realsense_handler import FramesMix

import ultralytics
import numpy as np
from dataclasses import dataclass
from enum import Enum


class ObjectType(Enum):
    BALL = "ball"
    RED = "red"
    BLUE = "blue"
    ROBOT = "robot"
    PERSON = "person"


@dataclass(frozen=True, slots=True)
class DetectedObject:
    """Dataclass to store object information"""

    type: ObjectType
    confidence: float
    box: tuple[int, int, int, int]


class ObjectDetector:
    def __init__(self, color_model_name: str, ir_model_name: str):
        self.color_model = ultralytics.YOLO(color_model_name, task="detect")
        self.color_classes = [
            ObjectType.BALL,
            ObjectType.RED,
            ObjectType.BLUE,
            ObjectType.PERSON,
        ]

        self.ir_model = ultralytics.YOLO(ir_model_name, task="detect")
        self.ir_classes = [ObjectType.BALL, ObjectType.PERSON, ObjectType.ROBOT]

    def _get_class(self, using_color: bool, class_id: int) -> ObjectType:
        if using_color:
            return self.color_classes[min(class_id, 2)]
        else:
            return self.ir_classes[min(class_id, 2)]

    def _map_result_to_object(
        self, source_type: FramesMix, result: list[float]
    ) -> DetectedObject:
        confidence = result[4]
        type = self._get_class(source_type == FramesMix.DEPTH_COLOR, int(result[5]))
        box = tuple(map(lambda x: int(x), result[:4]))
        return DetectedObject(type, confidence, box)

    def detect(
        self, source_type: FramesMix, source: np.ndarray
    ) -> list[DetectedObject]:
        """Detect objects in the frame"""
        model: ultralytics.YOLO = (
            self.color_model if source_type == FramesMix.DEPTH_COLOR else self.ir_model
        )

        results: list[ultralytics.engine.results.Results] = model.predict(
            source=source,
            conf=0.5,
            verbose=False,
            device=0,
            half=True,
            int8=False,
            agnostic_nms=True,
            imgsz=640,
        )

        detected_objects: list[DetectedObject] = []
        result_size: int = len(results)
        for result_index in range(result_size):
            result_data: list[float] = results[result_index].boxes.data.tolist()

            if len(result_data) > 0:
                detected_objects.append(
                    self._map_result_to_object(source_type, result_data[0])
                )

        return detected_objects
