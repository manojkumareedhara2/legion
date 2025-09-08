import cv2,os
import numpy as np 
from PIL import Image 
import torch 


class ImageUtils:
    @staticmethod
    def convert_to_pil(image):
        if isinstance(image,np.ndarray):
            return Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        elif isinstance(image,Image.Image):
            return image
        else:
            raise ValueError("Unsupported input type. Use np.ndarray or PIL.Image.")
    
    @staticmethod
    def annotate_image(image,text,position=(10,30),font_scale=1,color=(0,225,245),thickness=2):
        if isinstance(image,Image.Image):
            image =cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

        annotated_image = image.copy()
        cv2.putText(annotated_image,text,position,cv2.FONT_HERSHEY_SIMPLEX,font_scale,color,thickness)
        return annotated_image
    

