from realsense_handler import RealSenseHandler, RealSenseConfig, FramesMix
from object_detector import ObjectDetector, DetectedObject, ObjectType
from object_pose_estimator import ObjectPoseEstimator, ObjectWithPosition
from object_tracker import ObjectTracker
from ball_classifier import BallClassifier, BallClassifiedObject
from behavior_control import ActionStatus, BehaviorControl, BahaviorControlAction
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
    VISUALIZE_FIELD = True
    VISUALIZE_CAMS = False

    realsense = RealSenseHandler(RealSenseConfig())
    obj_detector = ObjectDetector(
        color_model_name="/home/robot3/Downloads/model3/noshirts11n21042025.engine",
        ir_model_name="/home/robot3/Downloads/model3/noshirts11n21042025.engine",
    )
    obj_pose_estimator = ObjectPoseEstimator(
        realsense.depth_intrinsics,
        color_intrinsics=realsense.color_intrinsics,
        ir_intrinsics=realsense.color_intrinsics,
        transform_camera_to_robot=np.array([[0, 0, 1, 0.05], [1, 0, 0, -0.2], [0, -1, 0, 0.51], [0, 0, 0, 1]]),
    )

    obj_tracker = ObjectTracker()

    ball_classifier = BallClassifier()

    behavior_controller = BehaviorControl()

    if VISUALIZE_FIELD:
        visualize_engine = field_visualizer.Engine()

    threecameras = threecameras_handler.ThreeCamerasHandler(angles_file="/home/robot3/MSL_2025/Detection-Tracking/Angulos.txt", distances_file="/home/robot3/MSL_2025/Detection-Tracking/Distancias-claud.txt")

    frame_count: int = 0
    start_time = time.perf_counter()
    first_timestamp = time.time_ns()
    average_times = {
        "waiting_realsense": 0,
        "waiting_threecameras": 0,
        "detect": 0,
        "getposition": 0,
        "track": 0,
        "classify": 0,
        "behavior": 0,
    }

    robot_location = np.eye(4)
    robot_location[0, 3] = 0.0

    while True:
        waiting_realsense_start = time.perf_counter()
        timestamp, frames_mix, depth_frame, second_frame = realsense.get_frames()

        second_image = np.asanyarray(second_frame.get_data())
        if frames_mix == FramesMix.DEPTH_INFRARED:
            second_image = cv2.cvtColor(second_image, cv2.COLOR_GRAY2BGR)

        average_times["waiting_realsense"] += time.perf_counter() - waiting_realsense_start

        waiting_threecameras_start = time.perf_counter()
        threecamera_timestamp, threecameras_frames = threecameras.get_images()
        average_times["waiting_threecameras"] += time.perf_counter() - waiting_threecameras_start

        image_sources = [second_image, *threecameras_frames]
        
        detect_loop_start = time.perf_counter()
        realsense_result, threecamera_result = obj_detector.detect(frames_mix, image_sources)
        print(f"Realsense result: {realsense_result}")
        average_times["detect"] += time.perf_counter() - detect_loop_start

        getposition_loop_start = time.perf_counter()
        realsense_result_pose: list[ObjectWithPosition] = obj_pose_estimator.estimate_position(
            robot_location, depth_frame, realsense_result
        )
        threecamera_result_pose: list[ObjectWithPosition] = threecameras.get_objects_position(robot_location, threecamera_result[0])

        measurements = [(timestamp, realsense_result_pose), (threecamera_timestamp, threecamera_result_pose)]
        measurements.sort(key=lambda x: x[0])
        average_times["getposition"] += time.perf_counter() - getposition_loop_start

        track_loop_start = time.perf_counter()
        track_result = obj_tracker.track(measurements)
        average_times["track"] += time.perf_counter() - track_loop_start

        classify_loop_start = time.perf_counter()
        ball_candidates, closest_ball, closest_ball_by_time = ball_classifier.classify(track_result)
        average_times["classify"] += time.perf_counter() - classify_loop_start

        behavior_loop_start = time.perf_counter()
        target_ball_index = closest_ball_by_time[0] if (closest_ball_by_time[0] is not None) else None
        target_ball_index = closest_ball[0] if (target_ball_index is None) else target_ball_index
        
        target_ball = ball_candidates.get(target_ball_index, None)
        control_action: BahaviorControlAction = behavior_controller.control(closest_ball=target_ball)
        average_times["behavior"] += time.perf_counter() - behavior_loop_start

        if VISUALIZE_FIELD:
            visualize_engine.clear()
            visualize_engine.add_goalkeeper(field_visualizer.Goalkeeper(visualize_engine, field_visualizer.Point(robot_location[0, 3], robot_location[1, 3])))
            visualize_engine.add_goalkeeper(field_visualizer.Goalkeeper(visualize_engine, field_visualizer.Point(*control_action.target), (255, 0, 0)))

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
        
        if VISUALIZE_CAMS:
            sources = [second_image, *threecameras_frames]
            for obj_index in range(len(realsense_result)):
                x1, y1, x2, y2 = realsense_result[obj_index].box
                h = y2 - y1

                cv2.circle(
                    second_image,
                    (int((x1 + x2) / 2), int(y1 + 3 * h / 4)),
                    5,
                    (0, 255, 0),
                    -1,
                )

                label = f"{str(realsense_result[obj_index].type)}:{realsense_result[obj_index].confidence:.2f}"
                y_label = y1 - 10 if y2 < 460 else y1 - 30
                cv2.putText(
                    second_image,
                    label,
                    (x1, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                x, y, z = list(realsense_result_pose[obj_index].position)
                if x == None or y == None or z == None:
                    x, y, z = 0, 0, 0

                y_position = y1 + 10 if y2 < 460 else y1 - 10
                cv2.putText(
                    second_image,
                    f"{x:.2f}, {y:.2f}, {z:.2f}",
                    (x1, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            for obj_index in range(len(threecamera_result[0])):
                x1, y1, x2, y2 = threecamera_result[0][obj_index].box
                cv2.circle(
                    sources[threecamera_result[0][obj_index].source],
                    (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    5,
                    (0, 255, 0),
                    -1,
                )

                label = f"{str(threecamera_result[0][obj_index].type)}:{threecamera_result[0][obj_index].confidence:.2f}"
                cv2.putText(
                    sources[threecamera_result[0][obj_index].source],
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                x, y, z = list(threecamera_result_pose[obj_index].position)
                if x == None or y == None or z == None:
                    x, y, z = 0, 0, 0

                cv2.putText(
                    sources[threecamera_result[0][obj_index].source],
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

        if VISUALIZE_CAMS:
            cv2.imshow("Depth", np.asanyarray(depth_frame.get_data()))
            if frames_mix == FramesMix.DEPTH_COLOR:
                cv2.imshow("Color", second_image)
            else:
                cv2.imshow("Infrared", second_image)
            cv2.imshow("ThreeCamera1", sources[1])
            cv2.imshow("ThreeCamera2", sources[2])
            cv2.imshow("ThreeCamera3", sources[3])

        frame_count += 1
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            average_times = {k: v / frame_count for k, v in average_times.items()}
            frame_count = 0
            start_time = time.perf_counter()

            print(f"FPS: {fps:.2f}")
            print(f"Average waiting realsense time: {1000 * average_times['waiting_realsense']:.2f} ms")
            print(f"Average waiting threecameras time: {1000 * average_times['waiting_threecameras']:.2f} ms")
            print(f"Detect time: {1000 * average_times['detect']:.2f} ms")
            print(f"Get position time: {1000 * average_times['getposition']:.2f} ms")
            print(f"Track time: {1000 * average_times['track']:.2f} ms")
            print(f"Classify time: {1000 * average_times['classify']:.2f} ms")
            print(f"Behavior time: {1000 * average_times['behavior']:.2f} ms")
            print(f"Total time: {1000 * (average_times['waiting_realsense'] + average_times['waiting_threecameras'] + average_times['detect'] + average_times['getposition'] + average_times['track'] + average_times['classify'] + average_times['behavior']):.2f} ms")

            average_times = {
                "waiting_realsense": 0,
                "waiting_threecameras": 0,
                "detect": 0,
                "getposition": 0,
                "track": 0,
                "classify": 0,
                "behavior": 0,
            }

        if VISUALIZE_CAMS:
            key = cv2.waitKey(1)
            if key == ord('q'):   
                break


if __name__ == "__main__":
    main()