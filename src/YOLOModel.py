import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import time
from Utils import ImageUtils
from abc import ABC, abstractmethod
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    AutoProcessor,
    AutoModelForCausalLM,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)


class BaseModel(ABC):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def predict(self, input_data):
        pass

    def preprocess_input(self, input_data):
        return ImageUtils.convert_to_pil(input_data)


class YOLOModel(BaseModel):
    def __init__(self, model_path, conf=0.5, iou=0.45, img_size=640, device=None):
        super().__init__(device)
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.img_size = img_size

    def predict(self, input_data):
        results = self.model.predict(
            source=input_data, conf=self.conf, iou=self.iou, imgsz=self.img_size
        )

        return results[0].plot()


class YOLOV5Model(BaseModel):
    def __init__(self, model_path, conf=0.1, iou=0.45, img_size=640, device=None):
        super().__init__(device)
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
        self.conf = conf
        self.iou = iou
        self.img_size = img_size

    def predict(self, input_data):
        results = self.model(input_data, size=self.img_size)
        return results.render()[0]


class HuggingFaceClassificationModel(BaseModel):
    def __init__(self, model_name, device=None):
        super().__init__(device)
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(
            self.device
        )
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.id2label = self.model.config.id2label

    def predict(self, input_data, annotate=True):
        image = self.preprocess_input(input_data)
        inputs = self.extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
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


class ImageCaptioningModel(BaseModel):
    def __init__(self, model_name, device=None):
        super().__init__(device)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(
            self.device
        )
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, input_data, annotate=True, max_length=50):
        image = self.preprocess_input(input_data)
        pixel_values = self.feature_extractor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)
        # attention_mask = torch.ones(pixel_values.shape[:-2], dtype=torch.long).to(
        #     self.device
        # )

        with torch.no_grad():
            output_ids = self.model.generate(pixel_values, max_length=max_length)
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if annotate:
            annotated_image = ImageUtils.annotate_image(image=image, text=caption)
            return annotated_image, caption
        else:
            return caption


class ModelFactory:
    @staticmethod
    def create_model(model_type, model_path, **kwargs):
        if model_type == "yolo":
            return YOLOModel(model_path=model_path, **kwargs)
        elif model_type == "yolov5":
            return YOLOV5Model(model_path=model_path, **kwargs)
        elif model_type == "classification":
            return HuggingFaceClassificationModel(model_name=model_path, **kwargs)
        elif model_type == "captioning":
            return ImageCaptioningModel(model_name=model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


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


