import cv2
import numpy as np
import time
from YOLOModel import ModelFactory


class DroneInference:
    def __init__(self, stream_url, model_type, model_path):
        self.stream_url = stream_url
        self.cap = None
        self.model = ModelFactory.create_model(model_type, model_path)
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def connect_stream(self):
        """Connect to the drone video stream"""
        self.cap = cv2.VideoCapture(self.stream_url)

        # Set buffer size to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Try to set a lower resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            print("Error: Could not open video stream")
            return False
        return True

    def run_inference(self):
        """Main inference loop"""
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Failed to grab frame, reconnecting...")
                self.cap.release()
                time.sleep(1)  # Wait before reconnecting
                if not self.connect_stream():
                    break
                continue

            # Calculate FPS
            self.frame_count += 1
            if self.frame_count >= 30:
                self.fps = self.frame_count / (time.time() - self.start_time)
                self.frame_count = 0
                self.start_time = time.time()

            # Run inference
            try:
                processed_image = self.model.predict(frame)
                if isinstance(processed_image, tuple):
                    annotated_frame, label = processed_image
                else:
                    annotated_frame = processed_image

                # Add FPS counter to the frame
                cv2.putText(
                    annotated_frame,
                    f"FPS: {self.fps:.1f}",
                    (1920, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Live Inference", annotated_frame)

            except Exception as e:
                print(f"Inference error: {e}")
                # Show the original frame if inference fails
                cv2.imshow("Live Inference", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    stream_url = "http://192.168.0.23/video_raw/tracking_front"

    # Initialize the drone inference
    # drone_inference = DroneInference(
    #     stream_url=stream_url,
    #     model_type="classification",
    #     model_path="microsoft/resnet-50",  # M:/Autonome Labs/Legion/Models/fire_smoke.pt",
    # )

    drone_inference = DroneInference(
        stream_url=stream_url,
        model_type="yolo",
        model_path="M:/Autonome Labs/Legion/Models/fire_smoke.pt",  # M:/Autonome Labs/Legion/Models/fire_smoke.pt",
    )

    # Connect to the stream
    if drone_inference.connect_stream():
        try:
            drone_inference.run_inference()
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            drone_inference.cleanup()
    else:
        print("Failed to connect to the drone stream")
