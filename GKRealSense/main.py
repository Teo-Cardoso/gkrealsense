from realsense_handler import RealSenseHandler, RealSenseConfig, FramesMix
from object_detector import ObjectDetector, DetectedObject, ObjectType
from object_pose_estimator import ObjectPoseEstimator, ObjectWithPosition
from object_tracker import ObjectTracker

import cv2
import numpy as np
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
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
    timestamps_vel = []
    velocity_x = []
    velocity_y = []
    velocity_z = []

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)  # 3 rows, 1 column
    axes[0].set_title("Velocity Components Over Time")
    axes[0].set_ylabel("Vx (m/s)")
    axes[1].set_ylabel("Vy (m/s)")
    axes[2].set_ylabel("Vz (m/s)")
    
    line_vx, = axes[0].plot([], [], 'r-', label="Vx")  # Red line
    line_vy, = axes[1].plot([], [], 'g-', label="Vy")  # Green line
    line_vz, = axes[2].plot([], [], 'b-', label="Vz")  # Blue line
    axes[2].set_xlabel("Time (s)")


    while True:
        start_time = time.perf_counter()
        timestamp, frames_mix, depth_frame, second_frame = realsense.get_frames()

        second_image = np.asanyarray(second_frame.get_data())
        if frames_mix == FramesMix.DEPTH_INFRARED:
            second_image = cv2.cvtColor(second_image, cv2.COLOR_GRAY2BGR)

        result: list[DetectedObject] = obj_detector.detect(frames_mix, second_image)
        result_pose: list[ObjectWithPosition] = obj_pose_estimator.estimate_position(
            depth_frame, result
        )

        track_result = obj_tracker.track([(timestamp, result_pose)])
        for track in track_result:
            if track.objectStatus.object_type == ObjectType.BLUE:
                timestamps_vel.append(timestamp - first_timestamp)
                velocity_x.append(track.kalmanFilter.x[0])
                velocity_y.append(track.kalmanFilter.x[1])
                velocity_z.append(track.kalmanFilter.x[2])
                break
        
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
            line_vx.set_data(timestamps_vel, velocity_x)
            line_vy.set_data(timestamps_vel, velocity_y)
            line_vz.set_data(timestamps_vel, velocity_z)

            for ax in axes:
                ax.relim()
                ax.autoscale_view()
    
            plt.draw()  # Redraw the plot
    
            break


if __name__ == "__main__":
    main()
    plt.savefig("img.png")
    plt.ioff()
