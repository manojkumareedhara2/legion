from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)
from PIL import Image
import numpy as np
import torch
from YOLOModel import BaseModel
import cv2


class HuggingFaceModel(BaseModel):
    def __init__(self, model_name, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(
            self.device
        )
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.id2label = self.model.config.id2label

    def predict(self, input_data):
        if isinstance(input_data, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB))
        elif isinstance(input_data, Image.Image):
            image = input_data
        else:
            raise ValueError("Unsupported input type. Use np.ndarray or PIL.Image.")

        inputs = self.extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model(**inputs)
            logits = output.logits
            predicted_idx = logits.argmax(-1).item()
            predoicted_label = self.id2label[predicted_idx]

        annotated_image = np.array(image)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.putText(
            annotated_image,
            predoicted_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 245),
            2,
        )
        return annotated_image


class ImageCaptioningModel(BaseModel):
    def __init__(self, model_name, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(
            self.device
        )
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, input_data):
        if isinstance(input_data, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB))
        elif isinstance(input_data, Image.Image):
            image = input_data
        else:
            raise ValueError("Unsupported input type. Use np.ndarray or PIL.Image.")

        # Correct argument name
        pixel_values = self.feature_extractor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values, max_length=50
            )  # pass the tensor directly
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        annotated_image = np.array(image)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.putText(
            annotated_image,
            caption,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 245),
            2,
        )
        return annotated_image
