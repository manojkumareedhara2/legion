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
    # stream_url = "http://192.168.0.23/video_raw/hires_small_color"
    video_source = 0  # "M:\Autonome Labs\Legion\data\yt1z.net - California wildfires Aerial view of Palisades fire (1080p).mp4"
    model = ModelFactory.create_model("yolov5", "M:/Autonome Labs/Legion/Models/rsp.pt")
    # add detection for hugging face #

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
