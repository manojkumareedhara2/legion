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
    OptimizedModelManager,
)


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


# # Preload models for faster switching
# def preload_models():
#     for model_name, config in MODEL_CONFIGS.items():
#         model_manager.load_model(config["type"], config["name"])
#     print("All models preloaded")


# # Call this function at startup
# preload_models()


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
