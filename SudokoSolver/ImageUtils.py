import cv2
import numpy as np
from SudokoSolver.Utils import order_points

def crop_image(img,corners,offset=0):
    rect = order_points(corners)
    rect[0] = [rect[0][0]-offset,rect[0][1]-offset] #tl
    rect[1] = [rect[1][0]+offset,rect[1][1]-offset] # tr
    rect[2] = [rect[2][0]+offset,rect[2][1]+offset] #br
    rect[3] = [rect[3][0]-offset,rect[3][1]+offset] #bl
    (tl,tr,br,bl) = rect


    # (tl,tr,br,bl) = ((int(tl[0]-offset),int(tl[1]-offset)),(int(tr[0]+offset),int(tr[1]-offset)),(int(br[0]-offset),int(br[1]+offset)),(int(bl[0]+offset),int(bl[1]+offset)))
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped

def normlize_gray_image(grayImg):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    close = cv2.morphologyEx(grayImg,cv2.MORPH_CLOSE,kernel1)
    div = np.float32(grayImg)/(close)
    res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
    return res

def convert_image_to_gray_sale(img):
    #gray = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray

def proportional_resize_image(img,scale):
    (h,w) = img.shape[:2]
    dim = (h//scale,w//scale)
    return cv2.resize(img,dim, interpolation=cv2.INTER_AREA)

def image_to_vector(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(28,28))
    return img

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