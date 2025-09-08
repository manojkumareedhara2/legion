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


if __name__ == "__main__":
    video_source = 0
    model = ModelFactory.create_model(
        "yolo", "M:/Autonome Labs/Legion/Models/fire_smoke.pt"
    )

    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        processed_image = model.predict(frame)
        if isinstance(processed_image, tuple):
            annotated_frame, label = processed_image
        else:
            annotated_frame = processed_image

        cv2.imshow("Live Inferance", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
