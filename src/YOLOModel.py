import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import time
from abc import ABC, abstractmethod
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)


class ImageUtils:
    @staticmethod
    def convert_to_pil(image):
        if isinstance(image, np.ndarray):
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError("Unsupported input type. Use np.ndarray or PIL.Image.")

    @staticmethod
    def annotate_image(
        image, text, position=(10, 30), font_scale=1, color=(0, 255, 245), thickness=2
    ):
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        annotated_image = image.copy()
        cv2.putText(
            annotated_image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )
        return annotated_image


class BaseModel(ABC):
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Using device: {self.device}")

    @abstractmethod
    def predict(self, input_data):
        pass

    def preprocess_input(self, input_data):
        return ImageUtils.convert_to_pil(input_data)


class OptimizedYOLOModel(BaseModel):
    def __init__(
        self,
        model_path,
        conf=0.5,
        iou=0.45,
        img_size=320,
        device=None,
        half_precision=True,
    ):
        super().__init__(device)
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.half = half_precision and self.device == "cuda"

        # Move model to device and enable optimizations
        self.model.to(self.device)
        if self.half:
            self.model.model.half()  # Use half precision for faster inference
            print("Using half precision (FP16) for faster inference")

        # Warm up the model
        self._warm_up()

    def _warm_up(self):
        """Warm up the model with a dummy inference"""
        print("Warming up model...")
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.predict(dummy_input)
        print("Model warmup complete")

    def predict(self, input_data):
        try:
            results = self.model.predict(
                source=input_data,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.img_size,
                verbose=False,  # Disable verbose output for speed
                half=self.half,  # Use half precision if available
            )
            return results[0].plot()
        except Exception as e:
            print(f"YOLO inference error: {e}")
            return input_data


class OptimizedYOLOV5Model(BaseModel):
    def __init__(
        self,
        model_path,
        conf=0.1,
        iou=0.45,
        img_size=320,
        device=None,
        half_precision=True,
    ):
        super().__init__(device)
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=model_path, trust_repo=True
        )
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.half = half_precision and self.device == "cuda"

        # Model optimizations
        self.model.conf = conf
        self.model.iou = iou
        self.model.to(self.device)

        if self.half:
            self.model.model.half()
            print("Using half precision (FP16) for YOLOv5")

    def predict(self, input_data):
        try:
            results = self.model(input_data, size=self.img_size)
            return results.render()[0]
        except Exception as e:
            print(f"YOLOv5 inference error: {e}")
            return input_data


class OptimizedHuggingFaceClassificationModel(BaseModel):
    def __init__(self, model_name, device=None):
        super().__init__(device)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.id2label = self.model.config.id2label
        self.model.eval()  # Set to evaluation mode

    def predict(self, input_data, annotate=True):
        image = self.preprocess_input(input_data)

        with torch.no_grad():
            inputs = self.extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.device == "cuda":
                with torch.amp.autocast("cuda"):  # Use mixed precision
                    output = self.model(**inputs)
            else:
                output = self.model(**inputs)

            logits = output.logits
            predicted_idx = logits.argmax(-1).item()
            predicted_label = self.id2label[predicted_idx]

        if annotate:
            annotated_image = ImageUtils.annotate_image(
                image=image, text=predicted_label
            )
            return annotated_image, predicted_label
        else:
            return predicted_label


class OptimizedImageCaptioningModel(BaseModel):
    def __init__(self, model_name, device=None):
        super().__init__(device)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def predict(self, input_data, annotate=True, max_length=50):
        image = self.preprocess_input(input_data)

        with torch.no_grad():
            pixel_values = self.feature_extractor(
                images=image, return_tensors="pt"
            ).pixel_values.to(self.device)

            if self.device == "cuda":
                with torch.amp.autocast("cuda"):
                    output_ids = self.model.generate(
                        pixel_values, max_length=max_length
                    )
            else:
                output_ids = self.model.generate(pixel_values, max_length=max_length)

        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if annotate:
            annotated_image = ImageUtils.annotate_image(image=image, text=caption)
            return annotated_image, caption
        else:
            return caption


class OptimizedModelFactory:
    @staticmethod
    def create_model(model_type, model_path, **kwargs):
        if model_type == "yolo":
            return OptimizedYOLOModel(model_path=model_path, **kwargs)
        elif model_type == "yolov5":
            return OptimizedYOLOV5Model(model_path=model_path, **kwargs)
        elif model_type == "classification":
            return OptimizedHuggingFaceClassificationModel(
                model_name=model_path, **kwargs
            )
        elif model_type == "captioning":
            return OptimizedImageCaptioningModel(model_name=model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class HighPerformanceModelManager:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.fps_history = []
        self.frame_skip = 2
        self.frame_counter = 0
        self.last_inference_time = 0
        self.avg_fps = 0

    def load_model(self, model_type, model_name, **kwargs):
        try:
            model_key = f"{model_type}_{model_name}"

            if model_key in self.models:
                self.current_model = self.models[model_key]
                print(f"Model {model_name} already loaded, switching to it.")
                return True

            print(f"Loading optimized model: {model_name}")
            model = OptimizedModelFactory.create_model(model_type, model_name, **kwargs)
            self.models[model_key] = model
            self.current_model = model
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, image, **kwargs):
        if self.current_model is None:
            return image

        # Frame skipping for performance
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return image

        try:
            # Update model parameters
            if hasattr(self.current_model, "conf") and "conf_threshold" in kwargs:
                self.current_model.conf = kwargs["conf_threshold"]
            if hasattr(self.current_model, "iou") and "iou_threshold" in kwargs:
                self.current_model.iou = kwargs["iou_threshold"]
            if hasattr(self.current_model, "img_size") and "img_size" in kwargs:
                self.current_model.img_size = kwargs["img_size"]

            # Measure inference time
            start_time = time.time()
            result = self.current_model.predict(image)
            inference_time = time.time() - start_time
            self.last_inference_time = inference_time

            # Calculate FPS
            fps = 1.0 / inference_time if inference_time > 0 else 0
            self.fps_history.append(fps)
            if len(self.fps_history) > 5:  # Smaller window for more responsive FPS
                self.fps_history.pop(0)

            self.avg_fps = (
                sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            )

            # Add FPS overlay to the image
            if isinstance(result, np.ndarray):
                cv2.putText(
                    result,
                    f"FPS: {self.avg_fps:.1f} | Time: {inference_time * 1000:.1f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            return result
        except Exception as e:
            print(f"Error during prediction: {e}")
            return image
