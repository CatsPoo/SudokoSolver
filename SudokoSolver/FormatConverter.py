from PIL import Image
import pillow_heif
import cv2
import numpy as np

def convert_heic_to_jpeg(heic_path):
    # Open the HEIC file using pillow-heif
    heif_file = pillow_heif.open_heif(heic_path)
    
    # Convert to PIL Image
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    
    image_np = np.array(image)
    
    # Convert RGB to BGR format for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    return image_bgr


def get_File_Formate(imgpath:str):
    strings = imgpath.split('.')
    if(strings.__len__() <2):
        return None
    else: 
        return strings[-1]