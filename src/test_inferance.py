import cv2
import numpy as np
import gradio as gr
from PIL import Image
import time
from typing import Union, Dict, Any
import json
from pathlib import Path
import os

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

# Define model types and their corresponding default models
MODEL_TYPES = {
    "yolo": {
        "name": "YOLO (Ultralytics)",
        "models": {
            "YOLOv8n": "yolov8n.pt",
            "YOLOv8s": "yolov8s.pt",
            "YOLOv8m": "yolov8m.pt",
            "YOLOv8l": "yolov8l.pt",
            "YOLOv8x": "yolov8x.pt",
            "fire": "M:/Autonome Labs/Legion/Models/fire_smoke.pt",
        },
    },
    "yolov5": {
        "name": "YOLOv5",
        "models": {
            "YOLOv5s": "yolov5s.pt",
            "YOLOv5m": "yolov5m.pt",
            "YOLOv5l": "yolov5l.pt",
            "YOLOv5x": "yolov5x.pt",
        },
    },
    "classification": {
        "name": "Classification",
        "models": {
            "ResNet-50": "microsoft/resnet-50",
            "ViT": "google/vit-base-patch16-224",
        },
    },
    "captioning": {
        "name": "Image Captioning",
        "models": {"ViT-GPT2": "nlpconnect/vit-gpt2-image-captioning"},
    },
}

# Store custom models
CUSTOM_MODELS = {}

# Track current model state
current_model_type = "yolo"
current_model_name = "YOLOv8n"
current_model_path = "yolov8n.pt"


# Preload default models for faster switching
def preload_models():
    print("Preloading default models...")
    for model_type, type_info in MODEL_TYPES.items():
        for model_name, model_path in type_info["models"].items():
            # For fire model, check if path exists but try to load anyway
            if model_name == "fire":
                if not Path(model_path).exists():
                    print(f"Warning: Fire model path not found: {model_path}")
                    print("Trying to load anyway in case path is accessible...")

            print(f"Preloading {model_name}...")
            try:
                model_manager.load_model(model_type, model_path)
                print(f"Successfully preloaded {model_name}")
            except Exception as e:
                print(f"Failed to preload {model_name}: {e}")
                # For fire model, try a fallback path
                if model_name == "fire":
                    fallback_path = "./models/fire_smoke.pt"
                    print(f"Trying fallback path: {fallback_path}")
                    try:
                        model_manager.load_model(model_type, fallback_path)
                        MODEL_TYPES[model_type]["models"][model_name] = fallback_path
                        print(f"Successfully preloaded {model_name} from fallback path")
                    except Exception as e2:
                        print(f"Failed to load fire model from fallback: {e2}")
    print("Default models preloaded")


# Call this function at startup for defaults
preload_models()

# Video capture objects for stream handling
stream_capture = None
current_stream_url = ""
is_streaming = False  # Flag to control stream processing


def get_available_models(model_type):
    """Get available models for a given type"""
    if model_type not in MODEL_TYPES:
        return []

    # Get default models for this type
    models = list(MODEL_TYPES[model_type]["models"].keys())

    # Add custom models for this type if any
    custom_models = [
        name for name, mtype in CUSTOM_MODELS.items() if mtype == model_type
    ]
    models.extend(custom_models)

    return models


def get_model_path(model_type, model_name):
    """Get the path/identifier for a model"""
    # Check if it's a custom model
    if model_name in CUSTOM_MODELS:
        return CUSTOM_MODELS[model_name]

    # Check if it's a default model
    if model_type in MODEL_TYPES and model_name in MODEL_TYPES[model_type]["models"]:
        return MODEL_TYPES[model_type]["models"][model_name]

    return None


def get_frame_from_stream(stream_url):
    """Get a frame from a video stream URL"""
    global stream_capture, current_stream_url

    if not stream_url:
        return None

    # Initialize or reinitialize the capture if URL changed
    if stream_capture is None or stream_url != current_stream_url:
        if stream_capture is not None:
            stream_capture.release()

        try:
            stream_capture = cv2.VideoCapture(stream_url)
            current_stream_url = stream_url
            if not stream_capture.isOpened():
                print(f"Failed to open stream: {stream_url}")
                return None
        except Exception as e:
            print(f"Error opening stream: {e}")
            return None

    # Read a frame from the stream
    try:
        ret, frame = stream_capture.read()
        if ret:
            return frame
        else:
            # Try to reopen the stream if reading failed
            stream_capture.release()
            stream_capture = cv2.VideoCapture(stream_url)
            ret, frame = stream_capture.read()
            return frame if ret else None
    except Exception as e:
        print(f"Error reading from stream: {e}")
        return None


# Inference function with proper model switching
def process_frame(
    frame, model_type, model_name, conf_threshold=0.3, iou_threshold=0.45, frame_skip=2
):
    """
    Optimized inference function for real-time processing with proper model switching
    """
    global current_model_type, current_model_name, current_model_path

    if frame is None:
        return None, ""

    # Update frame skip
    model_manager.frame_skip = frame_skip

    # Get model path
    model_path = get_model_path(model_type, model_name)
    if not model_path:
        return frame, f"Model {model_name} not found"

    # Special handling for fire model path issues
    if model_name == "fire" and not Path(model_path).exists():
        fallback_path = "./models/fire_smoke.pt"
        print(f"Fire model path not found, trying fallback: {fallback_path}")
        if Path(fallback_path).exists():
            model_path = fallback_path
            MODEL_TYPES[model_type]["models"][model_name] = fallback_path
        else:
            return frame, f"Fire model not found at {model_path} or {fallback_path}"

    # Check if we need to switch models
    if (
        model_type != current_model_type
        or model_name != current_model_name
        or model_path != current_model_path
    ):
        print(f"Switching model from {current_model_name} to {model_name}")
        success = model_manager.load_model(model_type, model_path)
        if success:
            current_model_type = model_type
            current_model_name = model_name
            current_model_path = model_path
            print(f"Successfully switched to model: {model_name}")
        else:
            return frame, f"Failed to load {model_name}. Please check the model path."

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

        # Ensure we're returning a numpy array in the correct format
        if hasattr(annotated_frame, "convert"):
            # Convert PIL image to numpy array (BGR format for OpenCV)
            annotated_frame = np.array(annotated_frame)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        elif isinstance(annotated_frame, np.ndarray):
            # Ensure it's in BGR format
            if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
                # It's already a color image, assume it's BGR
                pass

        return annotated_frame, text_result
    except Exception as e:
        print(f"Error during inference: {e}")
        return frame, f"Error: {str(e)}"


# Function to stop processing and reset
def stop_processing():
    global stream_capture, current_stream_url, is_streaming
    if stream_capture is not None:
        stream_capture.release()
        stream_capture = None
    current_stream_url = ""
    is_streaming = False
    return None, "Stream stopped and resources released", "0.0"


# Function to load custom model
def load_custom_model(model_type, model_name, model_path):
    global CUSTOM_MODELS

    if not model_type or not model_name or not model_path:
        return "Please provide all fields", gr.update(
            choices=get_available_models(model_type)
        )

    try:
        success = model_manager.load_model(model_type, model_path)
        if success:
            # Add to custom models
            CUSTOM_MODELS[model_name] = model_path
            available_models = get_available_models(model_type)
            return f"Successfully loaded {model_name}", gr.update(
                choices=available_models, value=model_name
            )
        else:
            return "Failed to load model", gr.update(
                choices=get_available_models(model_type)
            )
    except Exception as e:
        return f"Error: {str(e)}", gr.update(choices=get_available_models(model_type))


# Update model selector when model type changes
def update_model_selector(model_type):
    available_models = get_available_models(model_type)
    if available_models:
        return gr.update(choices=available_models, value=available_models[0])
    return gr.update(choices=[], value=None)


# Function to handle model selection changes
def on_model_change(model_type, model_name):
    global current_model_type, current_model_name
    print(f"Model changed to: {model_type} - {model_name}")
    return f"Selected: {model_name}"


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Real-time AI Model Inference")
    gr.Markdown("Switch between different models and types seamlessly")

    with gr.Row():
        input_source = gr.Radio(
            choices=["Webcam", "Stream URL"], value="Webcam", label="Input Source"
        )

    # Model selection section
    with gr.Row():
        model_type_selector = gr.Dropdown(
            choices=list(MODEL_TYPES.keys()), value="yolo", label="Select Model Type"
        )
        model_selector = gr.Dropdown(
            choices=get_available_models("yolo"),
            value="YOLOv8n" if get_available_models("yolo") else None,
            label="Select Model",
        )
        model_status = gr.Textbox(
            label="Model Status", value="Ready", interactive=False
        )

    # Custom model loading
    with gr.Accordion("Load Custom Model", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                custom_model_name = gr.Textbox(
                    label="Custom Model Name", placeholder="Enter a name for your model"
                )
            with gr.Column(scale=2):
                custom_model_path = gr.Textbox(
                    label="Custom Model Path",
                    placeholder="Path to model file or HuggingFace identifier",
                )
            with gr.Column(scale=1):
                load_custom_btn = gr.Button("Load Custom Model")

    # Parameters
    with gr.Row():
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
        # Webcam input (visible by default)
        webcam_input = gr.Image(
            sources=["webcam"], streaming=True, label="Webcam Input", visible=True
        )

        # Stream URL input (hidden initially)
        stream_url_input = gr.Textbox(
            label="Stream URL",
            placeholder="Enter stream URL (e.g., rtsp://..., http://...)",
            visible=False,
        )
        stream_status = gr.Textbox(
            label="Stream Status", value="Not connected", visible=False
        )

    with gr.Row():
        output_image = gr.Image(label="Processed Output", streaming=True)

    # Text output for classification/captioning results
    text_output = gr.Textbox(label="Model Output", interactive=False)

    # Performance metrics
    fps_display = gr.Textbox(label="FPS", value="0.0")

    # Load model status
    load_status = gr.Textbox(
        label="Load Status", interactive=False, value="Default models loaded"
    )

    # Buttons for stream control
    with gr.Row():
        start_stream_btn = gr.Button("Start Stream Processing", visible=False)
        stop_stream_btn = gr.Button("Stop Processing", visible=False)

    # Update model selector when model type changes
    model_type_selector.change(
        fn=update_model_selector, inputs=model_type_selector, outputs=model_selector
    ).then(
        fn=on_model_change,
        inputs=[model_type_selector, model_selector],
        outputs=model_status,
    )

    # Update model status when model selection changes
    model_selector.change(
        fn=on_model_change,
        inputs=[model_type_selector, model_selector],
        outputs=model_status,
    )

    # Load custom model event
    load_custom_btn.click(
        fn=load_custom_model,
        inputs=[model_type_selector, custom_model_name, custom_model_path],
        outputs=[load_status, model_selector],
    ).then(
        fn=on_model_change,
        inputs=[model_type_selector, model_selector],
        outputs=model_status,
    )

    # Function to toggle input components and buttons based on selection
    def toggle_inputs(source):
        if source == "Webcam":
            return [
                gr.update(visible=True),  # webcam_input
                gr.update(visible=False),  # stream_url_input
                gr.update(visible=False),  # stream_status
                gr.update(visible=True),  # output_image
                gr.update(visible=False),  # start_stream_btn
                gr.update(visible=False),  # stop_stream_btn
            ]
        else:
            return [
                gr.update(visible=False),  # webcam_input
                gr.update(visible=True),  # stream_url_input
                gr.update(visible=True),  # stream_status
                gr.update(visible=True),  # output_image
                gr.update(visible=True),  # start_stream_btn
                gr.update(visible=False),  # stop_stream_btn (visible after start)
            ]

    input_source.change(
        fn=toggle_inputs,
        inputs=input_source,
        outputs=[
            webcam_input,
            stream_url_input,
            stream_status,
            output_image,
            start_stream_btn,
            stop_stream_btn,
        ],
    )

    # Process webcam frames
    def process_webcam(
        frame, model_type, model_name, conf_threshold, iou_threshold, frame_skip
    ):
        if frame is None:
            return None, "", "0.0"

        # Convert the frame from Gradio's RGB format to BGR for OpenCV
        if isinstance(frame, np.ndarray):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        start_time = time.time()
        processed_frame, text_result = process_frame(
            frame, model_type, model_name, conf_threshold, iou_threshold, frame_skip
        )
        end_time = time.time()

        # Convert back to RGB for Gradio display
        if processed_frame is not None and isinstance(processed_frame, np.ndarray):
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Calculate FPS
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        fps_text = f"FPS: {fps:.1f}"

        return processed_frame, text_result, fps_text

    # Process stream frames
    def process_stream(
        stream_url, model_type, model_name, conf_threshold, iou_threshold, frame_skip
    ):
        global is_streaming
        is_streaming = True

        # Update model status
        gr.Info(f"Starting stream processing with {model_name}")

        while is_streaming:
            if not stream_url:
                break

            # Get frame from stream
            frame = get_frame_from_stream(stream_url)
            if frame is None:
                break

            start_time = time.time()
            processed_frame, text_result = process_frame(
                frame, model_type, model_name, conf_threshold, iou_threshold, frame_skip
            )
            end_time = time.time()

            # Convert to RGB for Gradio display
            if processed_frame is not None and isinstance(processed_frame, np.ndarray):
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Calculate FPS
            fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            fps_text = f"FPS: {fps:.1f}"

            yield processed_frame, text_result, fps_text
            time.sleep(0.033)  # ~30 FPS

        # Ensure clean exit
        is_streaming = False

    # Function to show stop button after starting stream
    def show_stop_button():
        return gr.update(visible=True)

    # Start stream processing when button is clicked
    start_stream_btn.click(
        fn=show_stop_button,
        outputs=[stop_stream_btn],
    ).then(
        fn=process_stream,
        inputs=[
            stream_url_input,
            model_type_selector,
            model_selector,
            conf_slider,
            iou_slider,
            frame_skip_slider,
        ],
        outputs=[output_image, text_output, fps_display],
        show_progress="hidden",
    )

    # Stop processing when stop button is clicked
    stop_stream_btn.click(
        fn=stop_processing,
        outputs=[output_image, text_output, fps_display],
        show_progress="hidden",
    )

    # Set up the processing pipeline for webcam
    webcam_input.stream(
        process_webcam,
        inputs=[
            webcam_input,
            model_type_selector,
            model_selector,
            conf_slider,
            iou_slider,
            frame_skip_slider,
        ],
        outputs=[output_image, text_output, fps_display],
        show_progress="hidden",
    )

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("./models", exist_ok=True)

    # Add instructions for custom models
    print("\n" + "=" * 50)
    print("INSTRUCTIONS FOR CUSTOM MODELS:")
    print("1. Place custom model files in the './models/' directory")
    print("2. Use the 'Load Custom Model' section to add them")
    print("3. For fire detection, ensure the model is at './models/fire_smoke.pt'")
    print("=" * 50 + "\n")

    demo.launch(
        server_port=7860, share=False
    )  # Set share=False to avoid connection issues
