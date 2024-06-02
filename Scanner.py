import cv2
import numpy as np
from typing import List
from FormatConverter import convert_heic_to_jpeg, get_File_Formate
from Utils import sort_list,convert_dounle_tuples_list_to_int,remove_last_row_and_column
from ImageRecognitionUtils import crop_image, normlize_gray_image, convert_image_to_gray_sale,proportional_resize_image
from OCR import get_text_from_image
class SudokoScanner:
    def __init__(self,boardImagePath) -> None:
        self.img = None

        if(get_File_Formate(boardImagePath) == 'HEIC'):
            self.img = convert_heic_to_jpeg(boardImagePath)
        else:
            self.img = cv2.imread(boardImagePath)

        self.img = proportional_resize_image(self.img,5)
        
    def get_image(self):
        return self.img

    def get_board_image(self):
        return self.crop_board_from_image(self.img.copy())

    def get_board_from_image(self) -> List[List[int]]:

        cropped_board = self.crop_board_from_image(self.img.copy())
        
        # centroids_list = remove_last_row_and_column(centroids_list)
        
        # for i,l in enumerate(centroids_list):
        #     for c in l:
        #         cv2.circle(cropped_board, c, 2, (255,0,0), 2)
        
        cellsImageArray = self.get_array_of_cells_images(cropped_board)
        print(get_text_from_image(cellsImageArray[0][0]))
        cv2.imshow('asd',cellsImageArray[0][0])
        cv2.waitKey(0)

    def convert_cells_images_array_to_int_array(self,cellsImages):
        for l in cellsImages:
            for cell in l:
                pass

    def get_array_of_cells_images(self,board_image):
        cellsImageList = []
        centroids_list = self.get_board_crosses_centroids(board_image)

        for i,l in enumerate(remove_last_row_and_column(centroids_list)):
            cellsImageList.append([])
            for j,p in enumerate(l):
                cellImageCorners = [
                    centroids_list[i][j],
                    centroids_list[i][j+1],
                    centroids_list[i+1][j],
                    centroids_list[i+1][j+1],
                ]
                cellImg = crop_image(board_image,cellImageCorners)
                cellsImageList[-1].append(cellImg)
        
        return cellsImageList
                

    def get_board_crosses_centroids(self,img):
        grayImage = convert_image_to_gray_sale(img)
        normlizedImage = normlize_gray_image(grayImage)
        withoutBackgroundImage = self.clean_board_background(normlizedImage)
        crossesImage = self.get_crosses_points_image(withoutBackgroundImage)
        centroids_list =  self.get_crosses_points_list(crossesImage)
        sorted_centroid_list = sort_list(centroids_list)
        return convert_dounle_tuples_list_to_int(sorted_centroid_list)        
    
    def get_board_mask(self,normlizedImg):
        mask = np.zeros((normlizedImg.shape),np.uint8)
        #thresh = cv2.adaptiveThreshold(normlizedImg,255,0,1,19,2)
        edges_lines =cv2.Canny(normlizedImg,60,200)
        kernel = np.ones((5,5),np.uint8)
        closing_edges_lines = cv2.morphologyEx(edges_lines, cv2.MORPH_CLOSE, kernel,iterations=1)
        contour,hier = cv2.findContours(closing_edges_lines,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
        cv2.normalize(dx,dx,1,255,cv2.NORM_MINMAX)
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
        dy = cv2.Sobel(withoutBackgroundImage,cv2.CV_16S,0,1)
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
    
    def get_board_corners(self,img):
        crosses_centroids = self.get_board_crosses_centroids(img)
        return [crosses_centroids[0][0],crosses_centroids[0][-1],crosses_centroids[-1][-1],crosses_centroids[-1][0]]

    def crop_board_from_image(self,img):
        board_corners = self.get_board_corners(img)
        return crop_image(img,board_corners,5)
        

