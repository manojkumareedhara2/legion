import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
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
    def __init__(self, model_path, conf=0.5, iou=0.45, img_size=640, device=None):
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


# class YOLOModel(BaseModel):
#     def __init__(self, model_path, conf=0.5, iou=0.45, img_size=640):
#         self.model = YOLO(model_path)
#         self.conf = conf
#         self.iou = iou
#         self.img_size = img_size

#     def predict(self, input_data):
#         results = self.model.predict(
#             source=input_data, conf=self.conf, iou=self.iou, imgsz=self.img_size
#         )
#         return results[0].plot()


# class YOLOV5Model(BaseModel):
#     def __init__(self, model_path, conf=0.25, iou=0.45, img_size=640):
#         self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
#         self.conf = conf
#         self.iou = iou
#         self.img_size = img_size

#     def predict(self, input_data):
#         results = self.model(input_data, size=self.img_size)
#         return results.render()[0]
