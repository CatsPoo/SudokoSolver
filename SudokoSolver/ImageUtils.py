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
    gray = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray

def proportional_resize_image(img,scale):
    (h,w) = img.shape[:2]
    dim = (h//scale,w//scale)
    return cv2.resize(img,dim, interpolation=cv2.INTER_AREA)

def image_to_vector(image_path, size=(64, 64), grayscale=True):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale if needed
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image
    image = cv2.resize(image, size)
    
    # Normalize the image
    image = image / 255.0
    
    # Flatten the image to a 1D vector
    image_vector = image.flatten()
    
    return image_vector