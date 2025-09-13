import cv2
import numpy as np
import gradio as gr
from PIL import Image
import time
from typing import Union, Dict, Any
import json
from pathlib import Path

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
        "name": "nlpconnect/vit-gpt2-image-captioning",
    },
}


# Preload models for faster switching
def preload_models():
    for model_name, config in MODEL_CONFIGS.items():
        print(f"Preloading {model_name}...")
        try:
            model_manager.load_model(config["type"], config["name"])
            print(f"Successfully preloaded {model_name}")
        except Exception as e:
            print(f"Failed to preload {model_name}: {e}")
    print("All models preloaded")


# Call this function at startup
preload_models()


# Inference function
def process_frame(
    frame, model_name="YOLOv8", conf_threshold=0.3, iou_threshold=0.45, frame_skip=2
):
    """
    Optimized inference function for real-time processing
    """
    if frame is None:
        return None, ""

    # Update frame skip
    model_manager.frame_skip = frame_skip

    # Set the current model if it's different from the last one
    if model_name != getattr(model_manager, "last_model_name", None):
        if model_name in model_manager.models:
            model_manager.current_model = model_manager.models[model_name]
            model_manager.last_model_name = model_name
        else:
            model_config = MODEL_CONFIGS[model_name]
            success = model_manager.load_model(
                model_config["type"], model_config["name"]
            )
            if success:
                model_manager.last_model_name = model_name
            else:
                return frame, ""

    # Update thresholds if applicable
    kwargs = {"conf_threshold": conf_threshold, "iou_threshold": iou_threshold}

    # Run prediction
    try:
        result = model_manager.predict(frame, **kwargs)

        # Handle different return types from different models
        if isinstance(result, tuple):
            # Classification/Captioning models return (image, text)
            annotated_frame, text_result = result
        else:
            # Detection models return just the image
            annotated_frame, text_result = result, ""

        # Ensure we're returning a numpy array
        if hasattr(annotated_frame, "convert"):
            annotated_frame = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)

        return annotated_frame, text_result
    except Exception as e:
        print(f"Error during inference: {e}")
        return frame, f"Error: {str(e)}"


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Real-time AI Model Inference")

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=list(MODEL_CONFIGS.keys()), value="YOLOv8", label="Select Model"
        )
        conf_slider = gr.Slider(0, 1, 0.05, value=0.3, label="Confidence Threshold")
        iou_slider = gr.Slider(0, 1, 0.05, value=0.45, label="IoU Threshold")
        frame_skip_slider = gr.Slider(
            minimum=1,
            maximum=10,
            step=1,
            value=2,
            label="Frame Skip (higher = faster but less accurate)",
        )

    with gr.Row():
        webcam_input = gr.Image(
            sources=["webcam"], streaming=True, label="Webcam Input"
        )
        output_image = gr.Image(label="Processed Output", streaming=True)

    # Text output for classification/captioning results
    text_output = gr.Textbox(label="Model Output", interactive=False)

    # Performance metrics
    fps_display = gr.Textbox(label="FPS", value="0.0")

    # Process frames in real-time
    def process_webcam(frame, model_name, conf_threshold, iou_threshold, frame_skip):
        if frame is None:
            return None, "", "0.0"

        start_time = time.time()
        processed_frame, text_result = process_frame(
            frame, model_name, conf_threshold, iou_threshold, frame_skip
        )
        end_time = time.time()

        # Calculate FPS
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        fps_text = f"FPS: {fps:.1f}"

        return processed_frame, text_result, fps_text

    # Set up the processing pipeline
    webcam_input.stream(
        process_webcam,
        inputs=[
            webcam_input,
            model_selector,
            conf_slider,
            iou_slider,
            frame_skip_slider,
        ],
        outputs=[output_image, text_output, fps_display],
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)
