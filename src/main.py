import cv2
import numpy as np
import gradio as gr
from PIL import Image
import time
import os
import threading
from queue import Queue
from pathlib import Path

# Import fixed model classes
from model_manager import HighPerformanceModelManager, SmartModelFactory

# Initialize the optimized model manager
model_manager = HighPerformanceModelManager()

# Define optimized model types
OPTIMIZED_MODEL_TYPES = {
    "yolo": {
        "name": "YOLO (Ultralytics) - OPTIMIZED",
        "models": {
            "YOLOv8n (Fastest)": "yolov8n.pt",
            "YOLOv8s (Balanced)": "yolov8s.pt",
            "YOLOv8m (Better)": "yolov8m.pt",
        },
    },
    "yolov5": {
        "name": "YOLOv5 - OPTIMIZED",
        "models": {
            "YOLOv5s (Fastest)": "yolov5s.pt",
            "YOLOv5m (Balanced)": "yolov5m.pt",
        },
    },
    "classification": {
        "name": "Classification",
        "models": {
            "ResNet-50": "microsoft/resnet-50",
        },
    },
    "captioning": {
        "name": "Image Captioning",
        "models": {"ViT-GPT2": "nlpconnect/vit-gpt2-image-captioning"},
    },
}

# Add fire model if exists
FIRE_MODEL_PATH = "./models/fire_smoke.pt"
if os.path.exists(FIRE_MODEL_PATH):
    OPTIMIZED_MODEL_TYPES["yolo"]["models"]["Fire Detection"] = FIRE_MODEL_PATH

CUSTOM_MODELS = {}
current_model_type = "yolo"
current_model_name = "YOLOv8n (Fastest)"


# Stream management
class FixedStreamManager:
    def __init__(self):
        self.cap = None
        self.current_stream_url = ""
        self.is_streaming = False
        self.frame_queue = Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.read_thread = None

    def connect_stream(self, stream_url):
        """Connect to stream with better error handling"""
        self.disconnect()

        if not stream_url or not stream_url.strip():
            return False

        print(f"Attempting to connect to: {stream_url}")

        # Try different backends
        backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]

        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(stream_url, backend)
                if self.cap.isOpened():
                    # Set optimized parameters
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.current_stream_url = stream_url
                    print(f"✓ Stream connected with backend {backend}")
                    return True
            except Exception as e:
                print(f"✗ Backend {backend} failed: {e}")
                continue

        print(f"✗ Failed to connect to stream: {stream_url}")
        return False

    def start_stream_reader(self):
        """Start background frame reading"""
        if self.cap is None or not self.cap.isOpened():
            return False

        self.stop_event.clear()
        self.is_streaming = True
        self.read_thread = threading.Thread(target=self._stream_reader, daemon=True)
        self.read_thread.start()
        return True

    def _stream_reader(self):
        """Background thread to read frames"""
        while not self.stop_event.is_set() and self.is_streaming:
            if self.cap is None:
                break

            ret, frame = self.cap.read()

            if not ret:
                print("Failed to read frame from stream")
                time.sleep(0.1)
                continue

            # Keep only the latest frame
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass

            self.frame_queue.put(frame)
            time.sleep(0.01)

    def get_frame(self):
        """Get latest frame from stream"""
        try:
            return self.frame_queue.get_nowait()
        except:
            return None

    def disconnect(self):
        """Cleanup stream resources"""
        self.stop_event.set()
        self.is_streaming = False

        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.current_stream_url = ""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        print("✓ Stream disconnected")


# Initialize stream manager
stream_manager = FixedStreamManager()


def preload_fastest_models():
    """Preload only the fastest models for immediate use"""
    print("Preloading optimized models...")
    fast_models = [
        ("yolo", "yolov8n.pt", {"img_size": 320, "half_precision": True}),
    ]

    for model_type, model_path, kwargs in fast_models:
        print(f"Preloading {model_path} with optimizations...")
        try:
            success = model_manager.load_model(model_type, model_path, **kwargs)
            if success:
                print(f"✓ Successfully preloaded {model_path}")
            else:
                print(f"✗ Failed to preload {model_path}")
        except Exception as e:
            print(f"Error preloading {model_path}: {e}")


preload_fastest_models()


def get_available_models(model_type):
    if model_type not in OPTIMIZED_MODEL_TYPES:
        return []
    models = list(OPTIMIZED_MODEL_TYPES[model_type]["models"].keys())
    custom_models = [
        name for name, mtype in CUSTOM_MODELS.items() if mtype == model_type
    ]
    models.extend(custom_models)
    return models


def get_model_path(model_type, model_name):
    if model_name in CUSTOM_MODELS:
        return CUSTOM_MODELS[model_name]
    if (
        model_type in OPTIMIZED_MODEL_TYPES
        and model_name in OPTIMIZED_MODEL_TYPES[model_type]["models"]
    ):
        return OPTIMIZED_MODEL_TYPES[model_type]["models"][model_name]
    return None


def get_model_specific_params(
    model_type, conf_threshold, iou_threshold, img_size, frame_skip
):
    """Get model-specific parameters based on model type"""
    base_params = {
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "img_size": img_size,
        "frame_skip": frame_skip,
    }

    # Only detection models use conf/iou parameters
    if model_type not in ["yolo", "yolov5"]:
        # Remove detection-specific parameters for classification/captioning models
        base_params.pop("conf_threshold", None)
        base_params.pop("iou_threshold", None)
        base_params.pop("img_size", None)

    return base_params


def process_frame_optimized(
    frame,
    model_type,
    model_name,
    conf_threshold=0.3,
    iou_threshold=0.45,
    frame_skip=2,
    img_size=320,
):
    global current_model_type, current_model_name

    if frame is None:
        return None, ""

    model_manager.frame_skip = frame_skip

    model_path = get_model_path(model_type, model_name)
    if not model_path:
        return frame, f"Model {model_name} not found"

    if model_type != current_model_type or model_name != current_model_name:
        print(f"Switching to optimized model: {model_name}")

        # Smart parameter selection based on model type
        if model_type in ["yolo", "yolov5"]:
            # Detection models get all parameters
            optim_params = {
                "conf": conf_threshold,
                "iou": iou_threshold,
                "img_size": img_size,
                "half_precision": True,
            }
        else:
            # Classification/Captioning models only get device-related parameters
            optim_params = {"half_precision": True}

        success = model_manager.load_model(model_type, model_path, **optim_params)
        if success:
            current_model_type = model_type
            current_model_name = model_name
            print(f"✓ Successfully switched to: {model_name}")
        else:
            return frame, f"Failed to load {model_name}"

    try:
        # Only pass relevant parameters to prediction
        if model_type in ["yolo", "yolov5"]:
            kwargs = {
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "img_size": img_size,
            }
        else:
            kwargs = {}  # No detection parameters for classification/captioning

        result = model_manager.predict(frame, **kwargs)

        if isinstance(result, tuple):
            annotated_frame, text_result = result
        else:
            annotated_frame, text_result = result, ""

        return annotated_frame, text_result
    except Exception as e:
        print(f"Error during inference: {e}")
        return frame, f"Error: {str(e)}"


def process_webcam_optimized(
    frame, model_type, model_name, conf_threshold, iou_threshold, frame_skip, img_size
):
    if frame is None:
        return None, "", "0.0"

    # Convert from RGB to BGR for OpenCV
    if isinstance(frame, np.ndarray):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    start_time = time.time()
    processed_frame, text_result = process_frame_optimized(
        frame,
        model_type,
        model_name,
        conf_threshold,
        iou_threshold,
        frame_skip,
        img_size,
    )
    end_time = time.time()

    # Convert back to RGB for Gradio
    if processed_frame is not None and isinstance(processed_frame, np.ndarray):
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

    # Calculate performance metrics
    fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
    fps_text = f"FPS: {fps:.1f} | Model: {model_name}"

    return processed_frame, text_result, fps_text


def start_stream_processing(
    stream_url,
    model_type,
    model_name,
    conf_threshold,
    iou_threshold,
    frame_skip,
    img_size,
):
    """Start stream processing"""
    if not stream_url or not stream_url.strip():
        yield None, "Please enter a valid stream URL", "0.0"
        return

    print(f"Starting stream processing for: {stream_url}")

    # Connect to stream
    if not stream_manager.connect_stream(stream_url):
        error_frame = create_status_frame("Failed to connect to stream")
        yield error_frame, "Connection failed", "0.0"
        return

    if not stream_manager.start_stream_reader():
        error_frame = create_status_frame("Failed to start stream reader")
        yield error_frame, "Stream error", "0.0"
        return

    # Process frames continuously
    frame_count = 0
    fps_history = []

    while stream_manager.is_streaming:
        frame = stream_manager.get_frame()

        if frame is None:
            waiting_frame = create_status_frame("Waiting for stream frames...")
            yield waiting_frame, "Waiting for frames...", "0.0"
            time.sleep(0.5)
            continue

        # Process frame
        start_time = time.time()
        processed_frame, text_result = process_frame_optimized(
            frame,
            model_type,
            model_name,
            conf_threshold,
            iou_threshold,
            frame_skip,
            img_size,
        )
        processing_time = time.time() - start_time

        # Convert to RGB for display
        if processed_frame is not None and isinstance(processed_frame, np.ndarray):
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Calculate FPS
        fps = 1.0 / processing_time if processing_time > 0 else 0
        fps_history.append(fps)
        if len(fps_history) > 10:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

        fps_text = f"FPS: {avg_fps:.1f} | Stream: Active"

        frame_count += 1
        yield processed_frame, text_result, fps_text

        time.sleep(0.01)


def create_status_frame(message):
    """Create a status frame with message"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    return frame


def stop_stream_processing():
    """Stop stream processing"""
    stream_manager.disconnect()
    stopped_frame = create_status_frame("Stream stopped")
    return stopped_frame, "Stream stopped", "0.0"


def test_stream_url(stream_url):
    """Test if stream URL is valid"""
    if not stream_url or not stream_url.strip():
        return "Enter a stream URL to test"

    print(f"Testing stream URL: {stream_url}")

    # Test connection
    test_cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if test_cap.isOpened():
        ret, frame = test_cap.read()
        test_cap.release()
        if ret:
            return f"Stream connected successfully! Frame: {frame.shape}"
        else:
            return "Stream connected but no frames received"
    else:
        return "Failed to connect to stream"


# Create optimized Gradio interface
with gr.Blocks(title="High-Performance AI Inference", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # High-Performance Real-time AI Model Inference
    **Fixed version - No more parameter errors!**
    """)

    with gr.Row():
        input_source = gr.Radio(
            choices=["Webcam", "Stream URL"], value="Webcam", label="Input Source"
        )

    with gr.Row():
        model_type_selector = gr.Dropdown(
            choices=list(OPTIMIZED_MODEL_TYPES.keys()),
            value="yolo",
            label="Model Type",
        )
        model_selector = gr.Dropdown(
            choices=get_available_models("yolo"),
            value="YOLOv8n (Fastest)",
            label="Select Model",
        )

    # Stream URL section
    with gr.Row(visible=False) as stream_url_row:
        with gr.Column():
            stream_url_input = gr.Textbox(
                label="Stream URL",
                placeholder="Enter RTSP, HTTP, or file URL (e.g., rtsp://..., http://..., /path/to/video.mp4)",
                lines=1,
                scale=4,
            )
            stream_test_output = gr.Textbox(
                label="Connection Test",
                value="Enter URL and click Test Connection",
                interactive=False,
            )
        with gr.Column(scale=1):
            test_stream_btn = gr.Button("Test Connection", size="sm")
            start_stream_btn = gr.Button("Start Stream", variant="primary")
            stop_stream_btn = gr.Button("Stop Stream", variant="stop")

    # Performance tuning section (only show for detection models)
    with gr.Accordion("⚡ Detection Model Settings", open=True) as detection_settings:
        with gr.Row():
            conf_slider = gr.Slider(
                0.1, 0.9, value=0.3, step=0.05, label="Confidence Threshold"
            )
            iou_slider = gr.Slider(
                0.1, 0.9, value=0.45, step=0.05, label="IoU Threshold"
            )

        with gr.Row():
            frame_skip_slider = gr.Slider(
                minimum=1, maximum=10, value=3, step=1, label="Frame Skip"
            )
            img_size_slider = gr.Slider(
                minimum=160, maximum=640, value=320, step=32, label="Inference Size"
            )

    # Main display area
    with gr.Row():
        webcam_input = gr.Image(
            sources=["webcam"],
            streaming=True,
            label="Live Camera Input",
            height=400,
            visible=True,
        )

        output_image = gr.Image(label="Processed Output", streaming=True, height=400)

    # Results and metrics
    with gr.Row():
        text_output = gr.Textbox(
            label="Detection Results", interactive=False, max_lines=3
        )
        fps_display = gr.Textbox(
            label="Performance", value="Ready...", interactive=False
        )

    # Input source toggle function
    def toggle_inputs(source):
        if source == "Webcam":
            return [
                gr.update(visible=True),  # webcam_input
                gr.update(visible=False),  # stream_url_row
                gr.update(visible=True),  # output_image
            ]
        else:
            return [
                gr.update(visible=False),  # webcam_input
                gr.update(visible=True),  # stream_url_row
                gr.update(visible=True),  # output_image
            ]

    # Toggle detection settings based on model type
    def toggle_detection_settings(model_type):
        if model_type in ["yolo", "yolov5"]:
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)

    input_source.change(
        fn=toggle_inputs,
        inputs=input_source,
        outputs=[webcam_input, stream_url_row, output_image],
    )

    # Event handlers
    def update_model_selector(model_type):
        available_models = get_available_models(model_type)
        if available_models:
            return gr.update(choices=available_models, value=available_models[0])
        return gr.update(choices=[], value=None)

    model_type_selector.change(
        fn=update_model_selector, inputs=model_type_selector, outputs=model_selector
    ).then(
        fn=toggle_detection_settings,
        inputs=model_type_selector,
        outputs=detection_settings,
    )

    # Stream URL events
    test_stream_btn.click(
        fn=test_stream_url, inputs=stream_url_input, outputs=stream_test_output
    )

    start_stream_btn.click(
        fn=start_stream_processing,
        inputs=[
            stream_url_input,
            model_type_selector,
            model_selector,
            conf_slider,
            iou_slider,
            frame_skip_slider,
            img_size_slider,
        ],
        outputs=[output_image, text_output, fps_display],
        show_progress=True,
    )

    stop_stream_btn.click(
        fn=stop_stream_processing, outputs=[output_image, text_output, fps_display]
    )

    # Webcam processing pipeline
    webcam_input.stream(
        process_webcam_optimized,
        inputs=[
            webcam_input,
            model_type_selector,
            model_selector,
            conf_slider,
            iou_slider,
            frame_skip_slider,
            img_size_slider,
        ],
        outputs=[output_image, text_output, fps_display],
        show_progress="hidden",
    )

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)

    print(f"Inferance is started...")

    demo.launch(server_port=7860, share=False, show_error=True)
