from realsense_handler import RealSenseHandler, RealSenseConfig, FramesMix
from object_detector import ObjectDetector, DetectedObject
import cv2
import numpy as np
import time

def main():
    realsense = RealSenseHandler(RealSenseConfig())
    obj_detector = ObjectDetector(color_model_name="", ir_model_name="")
    frame_count: int = 0
    start_time = time.perf_counter()
    while True:
        frames_mix, depth_frame, second_frame = realsense.get_frames()
        second_image = np.asanyarray(second_frame.get_data())
        if frames_mix == FramesMix.DEPTH_INFRARED:
            second_image = cv2.cvtColor(second_image, cv2.COLOR_GRAY2BGR)

        result: list[DetectedObject] = obj_detector.detect(frames_mix, second_image)
        
        for obj in result:
            x1, y1, x2, y2 = obj.box
            cv2.rectangle(second_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{str(obj.type)}:{obj.confidence:.2f}"
            cv2.putText(second_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Depth', np.asanyarray(depth_frame.get_data()))
        if frames_mix == FramesMix.DEPTH_COLOR:
            cv2.imshow('Color', second_image)
        else:
            cv2.imshow('Infrared',  second_image)
        
        
        frame_count += 1
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.perf_counter()
            print(f"FPS: {fps:.2f}")
        
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
