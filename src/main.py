import cv2
import numpy as np
import gradio as gr
from PIL import Image
import time
from typing import Union, Dict, Any
import json
from pathlib import Path
from fastrtc import Stream

# Import your model classes
from Utils import ImageUtils
from YOLOModel import (
    YOLOModel,
    YOLOV5Model,
    HuggingFaceClassificationModel,
    ImageCaptioningModel,
    ModelFactory,
)


class OptimizedModelManager:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.last_model_name = None
        self.fps_history = []
        self.frame_skip = 2  # Process every nth frame
        self.frame_counter = 0

    def load_model(self, model_type, model_name, **kwargs):
        try:
            if model_name in self.models:
                self.current_model = self.models[model_name]
                return True

            print(f"Loading model: {model_name}")
            model = ModelFactory.create_model(model_type, model_name, **kwargs)
            self.models[model_name] = model
            self.current_model = model
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, image, **kwargs):
        if self.current_model is None:
            return image

        # Skip frames to improve performance
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return image

        try:
            # Update model parameters if provided
            if hasattr(self.current_model, "conf") and "conf_threshold" in kwargs:
                self.current_model.conf = kwargs["conf_threshold"]
            if hasattr(self.current_model, "iou") and "iou_threshold" in kwargs:
                self.current_model.iou = kwargs["iou_threshold"]

            # Measure inference time
            start_time = time.time()
            result = self.current_model.predict(image)
            inference_time = time.time() - start_time

            # Calculate and display FPS
            fps = 1.0 / inference_time if inference_time > 0 else 0
            self.fps_history.append(fps)
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)

            avg_fps = (
                sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            )

            # Add FPS overlay to the image
            if isinstance(result, np.ndarray):
                cv2.putText(
                    result,
                    f"FPS: {avg_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            return result
        except Exception as e:
            print(f"Error during prediction: {e}")
            return image


# Initialize the optimized model manager
model_manager = OptimizedModelManager()

MODEL_CONFIGS = {
    "YOLOv8": {"type": "yolo", "name": "yolov8n.pt"},
    "YOLOv5": {"type": "yolov5", "name": "yolov5s.pt"},
    "fire": {"type": "yolo", "name": "M:/Autonome Labs/Legion/Models/fire_smoke.pt"},
    "Classification": {
        "type": "classification",
        "name": "microsoft/resnet-50",
    },
    "Captioning": {
        "type": "captioning",
        "name": "facebook/detr-resnet-50",
    },
}


# Preload models for faster switching
def preload_models():
    for model_name, config in MODEL_CONFIGS.items():
        model_manager.load_model(config["type"], config["name"])
    print("All models preloaded")


# Call this function at startup
preload_models()


# Inference function for the Stream
def stream_inference(
    frame, model_name="YOLOv8", conf_threshold=0.3, iou_threshold=0.45
):
    """
    Optimized inference function for real-time processing
    """
    # Set the current model if it's different from the last one
    if model_name != model_manager.last_model_name:
        if model_name in model_manager.models:
            model_manager.current_model = model_manager.models[model_name]
            model_manager.last_model_name = model_name
        else:
            model_config = MODEL_CONFIGS[model_name]
            model_manager.load_model(model_config["type"], model_config["name"])
            model_manager.last_model_name = model_name

    # Update thresholds if applicable
    kwargs = {"conf_threshold": conf_threshold, "iou_threshold": iou_threshold}

    # Run prediction
    annotated_frame = model_manager.predict(frame, **kwargs)

    # Ensure we're returning a numpy array
    if hasattr(annotated_frame, "convert"):
        annotated_frame = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)

    return annotated_frame


# Define the fastrtc Stream
stream = Stream(
    handler=stream_inference,
    modality="video",
    mode="send-receive",
    additional_inputs=[
        gr.Dropdown(
            choices=list(MODEL_CONFIGS.keys()), value="YOLOv8", label="Select Model"
        ),
        gr.Slider(0, 1, 0.05, value=0.3, label="Confidence Threshold"),
        gr.Slider(0, 1, 0.05, value=0.45, label="IoU Threshold"),
    ],
    # For local testing, rtc_configuration=None
    rtc_configuration=None,
    concurrency_limit=2,  # limit simultaneous streams
)

# # Add performance tuning options
# performance_options = gr.Accordion("Performance Options", open=False)
# with performance_options:
#     frame_skip_slider = gr.Slider(
#         minimum=1,
#         maximum=10,
#         step=1,
#         value=2,
#         label="Frame Skip (higher = faster but less accurate)",
#     )

#     def update_frame_skip(value):
#         model_manager.frame_skip = value
#         return f"Frame skip updated to {value}"

#     frame_skip_slider.change(update_frame_skip, inputs=frame_skip_slider)

# Launch the UI
if __name__ == "__main__":
    # Add performance options to the UI
    demo = gr.Blocks()
    with demo:
        gr.Markdown("# Real-time AI Model Inference")
        stream.ui.launch(server_port=7860)
