import cv2
import numpy as np
from typing import List
from FormatConverter import convert_heic_to_jpeg, get_File_Formate
from Utils import proportional_resize_image, sort_list,convert_dounle_tuples_list_to_int,remove_last_row_and_column
class SudokoScanner:
    def __init__(self,boardImage) -> None:
        self.boardImagePath = boardImage
    
    def get_board_from_image(self) -> List[List[int]]:
        if(get_File_Formate(self.boardImagePath) == 'HEIC'):
            img = convert_heic_to_jpeg(self.boardImagePath)
        else:
            img = cv2.imread(self.boardImagePath)

    
        img = proportional_resize_image(img,6)
        crosses_centroids = self.get_board_crosses_centroids(img)
        strached_image = self.crop_board_from_image(img,[crosses_centroids[0][0],crosses_centroids[0][-1],crosses_centroids[-1][0],crosses_centroids[-1][-1]])
        cv2.imshow('asd',strached_image)
        cv2.waitKey(0)

    def get_board_crosses_centroids(self,img):
        grayImage = self.convert_image_to_gray_sale(img)
        normlizedImage = self.normlize_gray_image(grayImage)
        withoutBackgroundImage = self.clean_board_background(normlizedImage)
        crossesImage = self.get_crosses_points_image(withoutBackgroundImage)
        centroids_list =  self.get_crosses_points_list(crossesImage)
        sorted_centroid_list = sort_list(centroids_list)
        return convert_dounle_tuples_list_to_int(sorted_centroid_list)


    def normlize_gray_image(self,grayImg):
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

        close = cv2.morphologyEx(grayImg,cv2.MORPH_CLOSE,kernel1)
        div = np.float32(grayImg)/(close)
        res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
        return res

    def convert_image_to_gray_sale(self,img):
        gray = cv2.GaussianBlur(img,(5,5),0)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return gray
        
    
    def get_board_mask(self,normlizedImg):
        mask = np.zeros((normlizedImg.shape),np.uint8)
        thresh = cv2.adaptiveThreshold(normlizedImg,255,0,1,19,2)
        contour,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        best_cnt = None
        for cnt in contour:
            area = cv2.contourArea(cnt)
            if area > 1000:
                if area > max_area:
                    max_area = area
                    best_cnt = cnt

        cv2.drawContours(mask,[best_cnt],0,255,-1)
        cv2.drawContours(mask,[best_cnt],0,0,2)

        return mask

    def clean_board_background(self,normlizedImage):
        mask = self.get_board_mask(normlizedImage)
        return cv2.bitwise_and(normlizedImage,mask)
    
    def get_vertical_lines(self,withoutBackgroundImage):
        kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10))

        dx = cv2.Sobel(withoutBackgroundImage,cv2.CV_16S,1,0)
        dx = cv2.convertScaleAbs(dx)
        cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
        ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

        contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            x,y,w,h = cv2.boundingRect(cnt)
            if h/w > 5:
                cv2.drawContours(close,[cnt],0,255,-1)
            else:
                cv2.drawContours(close,[cnt],0,0,-1)
        close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
        return close.copy()
    
    def get_horitontal_lines(self,withoutBackgroundImage):
        kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
        dy = cv2.Sobel(withoutBackgroundImage,cv2.CV_16S,0,2)
        dy = cv2.convertScaleAbs(dy)
        cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
        ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

        contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            x,y,w,h = cv2.boundingRect(cnt)
            if w/h > 5:
                cv2.drawContours(close,[cnt],0,255,-1)
            else:
                cv2.drawContours(close,[cnt],0,0,-1)

        close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
        return close.copy()

    def get_crosses_points_image(self,withoutBackgroundImage):
        verticalLinesImage = self.get_vertical_lines(withoutBackgroundImage)
        horizontalLinesImage = self.get_horitontal_lines(withoutBackgroundImage)
        crossesImage = cv2.bitwise_and(verticalLinesImage,horizontalLinesImage)
        return crossesImage
    
    def get_crosses_points_list(self,crossesPointsImage):
        contour,hier = cv2.findContours(crossesPointsImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for cnt in contour:
            mom = cv2.moments(cnt)
            (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
            centroids.append((x,y))
        return centroids
    
    def crop_board_from_image(self,img,board_corners):
        min_x = min(point[0] for point in board_corners)
        max_x = max(point[0] for point in board_corners)
        min_y = min(point[1] for point in board_corners)
        max_y = max(point[1] for point in board_corners)

        bounding_box = np.array([[max_x, max_y], [min_x, max_y], [max_x, min_y], [min_x, min_y]][::-1])

        transform_matrix = cv2.getPerspectiveTransform(np.array(board_corners, dtype=np.float32), bounding_box.astype(np.float32))
        stretched_image = cv2.warpPerspective(img, transform_matrix, (img.shape[1], img.shape[0]))
        return stretched_image
    

