from realsense_handler import RealSenseHandler, RealSenseConfig, FramesMix
from object_detector import ObjectDetector, DetectedObject, ObjectType
from object_pose_estimator import ObjectPoseEstimator, ObjectWithPosition
from object_tracker import ObjectTracker

import cv2
import numpy as np
import time

def main():
    SAVE_DATA = False
    realsense = RealSenseHandler(RealSenseConfig())
    obj_detector = ObjectDetector(
        color_model_name="/workspaces/neno_ws/best_with_lines.pt",
        ir_model_name="/workspaces/neno_ws/best_300ep.pt",
    )
    obj_pose_estimator = ObjectPoseEstimator(
        realsense.depth_intrinsics,
        color_intrinsics=realsense.color_intrinsics,
        ir_intrinsics=realsense.color_intrinsics,
    )

    obj_tracker = ObjectTracker()

    frame_count: int = 0
    start_time = time.perf_counter()
    first_timestamp = start_time

    while True:
        start_time = time.perf_counter()
        timestamp, frames_mix, depth_frame, second_frame = realsense.get_frames()

        second_image = np.asanyarray(second_frame.get_data())
        if frames_mix == FramesMix.DEPTH_INFRARED:
            second_image = cv2.cvtColor(second_image, cv2.COLOR_GRAY2BGR)

        result: list[DetectedObject] = obj_detector.detect(frames_mix, second_image)
        print(f"Detected {len(result)} objects")
        result_pose: list[ObjectWithPosition] = obj_pose_estimator.estimate_position(
            depth_frame, result
        )

        track_result = obj_tracker.track([(timestamp, result_pose)])
        if SAVE_DATA:
            with open(f"track_results_{first_timestamp}.txt", "a") as file:
                file.write(f"timestamp: {timestamp}, objects: [")
                for track in track_result:
                    file.write(f" ({track.objectStatus.object_id}, {track.objectStatus.object_type}, {track.kalmanFilter.x}, {track.kalmanFilterStatus.trustiness}, {track.kalmanFilterStatus.updated_in_the_last_cycle})")
                    file.write(",")
                file.write("]\n")
                
        
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

            x, y, z = list(result_pose[obj_index].position)[0]
            cv2.putText(
                second_image,
                f"{x:.2f}, {y:.2f}, {z:.2f}",
                (x1, y2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        
        if SAVE_DATA:
            with open(f"image_data_{first_timestamp}.txt", "a") as img_file:
                img_file.write(f"timestamp: {timestamp}, image: [")
                img_file.write(str(second_image.tobytes()))
                img_file.write("]\n")

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

        key = cv2.waitKey(1)
        if key == ord('q'):   
            break


if __name__ == "__main__":
    main()