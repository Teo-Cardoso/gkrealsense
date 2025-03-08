from realsense_handler import RealSenseHandler, RealSenseConfig, FramesMix
from object_detector import ObjectDetector, DetectedObject
from object_pose_estimator import ObjectPoseEstimator, ObjectWithPose

import cv2
import numpy as np
import time


def main():
    realsense = RealSenseHandler(RealSenseConfig())
    obj_detector = ObjectDetector(color_model_name="", ir_model_name="")
    obj_pose_estimator = ObjectPoseEstimator(
        realsense.depth_intrinsics,
        color_intrinsics=realsense.color_intrinsics,
        ir_intrinsics=realsense.ir_intrinsics,
    )

    frame_count: int = 0
    start_time = time.perf_counter()
    while True:
        frames_mix, depth_frame, second_frame = realsense.get_frames()
        second_image = np.asanyarray(second_frame.get_data())
        if frames_mix == FramesMix.DEPTH_INFRARED:
            second_image = cv2.cvtColor(second_image, cv2.COLOR_GRAY2BGR)

        result: list[DetectedObject] = obj_detector.detect(frames_mix, second_image)
        result_pose: list[ObjectWithPose] = obj_pose_estimator.estimate_pose(
            depth_frame, result
        )

        for obj_index in range(len(result)):
            x1, y1, x2, y2 = result[obj_index].box
            cv2.circle(
                second_image,
                (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                5,
                (0, 255, 0),
                -1,
            )

            label = f"{str(result[obj_index].type)}:{result[obj_index].confidence:.2f}"
            cv2.putText(
                second_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            x, y, z = list(result_pose[obj_index].pose)[0]
            cv2.putText(
                second_image,
                f"{x:.2f}, {y:.2f}, {z:.2f}",
                (x1, y2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Depth", np.asanyarray(depth_frame.get_data()))
        if frames_mix == FramesMix.DEPTH_COLOR:
            cv2.imshow("Color", second_image)
        else:
            cv2.imshow("Infrared", second_image)

        frame_count += 1
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.perf_counter()
            print(f"FPS: {fps:.2f}")

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
