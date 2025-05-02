from realsense_handler import RealSenseHandler, RealSenseConfig, FramesMix
from object_detector import ObjectDetector, DetectedObject, ObjectType
from object_pose_estimator import ObjectPoseEstimator, ObjectWithPosition
from object_tracker import ObjectTracker
from ball_classifier import BallClassifier, BallClassifiedObject
import field_visualizer

import cv2
import numpy as np
import time

try:
    import threecameras_handler
except Exception as e:
    print("Error importing threecameras_handler:", e)

def main():
    SAVE_DATA = False
    VISUALIZE = True

    realsense = RealSenseHandler(RealSenseConfig())
    obj_detector = ObjectDetector(
        color_model_name="/home/robot3/Downloads/model4_480_640_half/noshirts11n21042025.engine",
        ir_model_name="/home/robot3/Downloads/model4_480_640_half/noshirts11n21042025.engine",
    )
    obj_pose_estimator = ObjectPoseEstimator(
        realsense.depth_intrinsics,
        color_intrinsics=realsense.color_intrinsics,
        ir_intrinsics=realsense.color_intrinsics,
        transform_camera_to_robot=np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0.535], [0, 0, 0, 1]]),
    )

    obj_tracker = ObjectTracker()

    ball_classifier = BallClassifier()

    if VISUALIZE:
        visualize_engine = field_visualizer.Engine()

    threecameras = threecameras_handler.ThreeCamerasHandler(angles_file="/home/robot3/MSL_2025/Detection-Tracking/Angulos-claud.txt", distances_file="/home/robot3/MSL_2025/Detection-Tracking/Distancias-claud.txt")

    frame_count: int = 0
    start_time = time.perf_counter()
    first_timestamp = time.time_ns()

    while True:
        timestamp, frames_mix, depth_frame, second_frame = realsense.get_frames()

        second_image = np.asanyarray(second_frame.get_data())
        if frames_mix == FramesMix.DEPTH_INFRARED:
            second_image = cv2.cvtColor(second_image, cv2.COLOR_GRAY2BGR)

        threecamera_timestamp, threecameras_frames = threecameras.get_images()

        image_sources = [second_image, *threecameras_frames]
        
        detect_loop_start = time.perf_counter()
        realsense_result, threecamera_result = obj_detector.detect(frames_mix, image_sources)
        print(f"[{frame_count}] Detect time: {1000 * (time.perf_counter() - detect_loop_start):.2f} ms")

        realsense_result_pose: list[ObjectWithPosition] = obj_pose_estimator.estimate_position(
            np.eye(4), depth_frame, realsense_result
        )
        threecamera_result_pose: list[list[ObjectWithPosition]] = threecameras.get_objects_position(np.eye(4), threecamera_result[0])

        measurements = [(timestamp, realsense_result_pose)]
        for cam_index in range(len(threecamera_result_pose)):
            if len(threecamera_result_pose[cam_index]) != 0:
                measurements.append((threecamera_timestamp, threecamera_result_pose[cam_index]))
        measurements.sort(key=lambda x: x[0])
        
        track_loop_start = time.perf_counter()
        track_result = obj_tracker.track(measurements)
        print(f"[{frame_count}] Track time: {1000 * (time.perf_counter() - track_loop_start):.2f} ms")

        classify_loop_start = time.perf_counter()
        ball_candidates, closest_ball, closest_ball_by_time = ball_classifier.classify(track_result)
        print(closest_ball_by_time)
        print(f"[{frame_count}] Ball classifier time: {1000 * (time.perf_counter() - classify_loop_start):.2f} ms")

        if VISUALIZE:
            visualize_engine.clear()
            visualize_engine.add_goalkeeper(field_visualizer.Goalkeeper(visualize_engine, field_visualizer.Point(0, 0)))
            for ball_id in ball_candidates:
                if ball_candidates[ball_id].properties.trustiness <= 0.2:
                    continue

                visualize_engine.add_ball(field_visualizer.Ball(visualize_engine, ball_candidates[ball_id]))
            
            if not visualize_engine.run():
                break

        if SAVE_DATA:
            with open(f"track_results_{first_timestamp}.txt", "a") as file:
                file.write(f"timestamp: {timestamp}, objects: [")
                for track in track_result:
                    if track.kalmanFilterStatus.cycles == 1:
                        x_prior , P_prior = track.kalmanFilter.x, track.kalmanFilter.P.tolist()
                    else:
                        x_prior, P_prior = track.kalmanFilter.x_prior, track.kalmanFilter.P_prior.tolist()
                    file.write(f" ({track.objectStatus.object_id}, {track.objectStatus.object_type}, {track.kalmanFilter.x}, {track.kalmanFilter.P.tolist()}, {x_prior}, {P_prior}, {track.kalmanFilterStatus.trustiness}, {track.kalmanFilterStatus.updated_in_the_last_cycle})")
                    file.write(",")
                file.write("]\n")
                
        # sources = [second_image, *threecameras_frames]
        # for obj_index in range(len(realsense_result)):
        #     x1, y1, x2, y2 = realsense_result[obj_index].box
        #     cv2.circle(
        #         second_image,
        #         (int((x1 + x2) / 2), int((y1 + y2) / 2)),
        #         5,
        #         (0, 255, 0),
        #         -1,
        #     )

        #     label = f"{str(realsense_result[obj_index].type)}:{realsense_result[obj_index].confidence:.2f}"
        #     cv2.putText(
        #         second_image,
        #         label,
        #         (x1, y1 - 10),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (0, 255, 0),
        #         2,
        #     )

        #     x, y, z = list(realsense_result_pose[obj_index].position)
        #     if x == None or y == None or z == None:
        #         x, y, z = 0, 0, 0

        #     cv2.putText(
        #         second_image,
        #         f"{x:.2f}, {y:.2f}, {z:.2f}",
        #         (x1, y2 + 10),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (0, 255, 0),
        #         2,
        #     )

        # for obj_index in range(len(threecamera_result[0])):
        #     x1, y1, x2, y2 = threecamera_result[0][obj_index].box
        #     cv2.circle(
        #         sources[threecamera_result[0][obj_index].source],
        #         (int((x1 + x2) / 2), int((y1 + y2) / 2)),
        #         5,
        #         (0, 255, 0),
        #         -1,
        #     )

        #     label = f"{str(threecamera_result[0][obj_index].type)}:{threecamera_result[0][obj_index].confidence:.2f}"
        #     cv2.putText(
        #         sources[threecamera_result[0][obj_index].source],
        #         label,
        #         (x1, y1 - 10),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (0, 255, 0),
        #         2,
        #     )

        #     x, y, z = list(threecamera_result_pose[obj_index].position)
        #     if x == None or y == None or z == None:
        #         x, y, z = 0, 0, 0

        #     cv2.putText(
        #         sources[threecamera_result[0][obj_index].source],
        #         f"{x:.2f}, {y:.2f}, {z:.2f}",
        #         (x1, y2 + 10),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (0, 255, 0),
        #         2,
        #     )
        
        # if SAVE_DATA:
        #     with open(f"image_data_{first_timestamp}.txt", "a") as img_file:
        #         img_file.write(f"timestamp: {timestamp}, image: [")
        #         img_file.write(str(second_image.tobytes()))
        #         img_file.write("]\n")

        # cv2.imshow("Depth", np.asanyarray(depth_frame.get_data()))
        # if frames_mix == FramesMix.DEPTH_COLOR:
        #     cv2.imshow("Color", second_image)
        # else:
        #     cv2.imshow("Infrared", second_image)
        # cv2.imshow("ThreeCamera1", sources[1])
        # cv2.imshow("ThreeCamera2", sources[2])
        # cv2.imshow("ThreeCamera3", sources[3])

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