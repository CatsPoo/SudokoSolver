import cv2
import pickle
from sklearn.pipeline import Pipeline 
import pandas as pd
import os
import numpy as np
from SudokoSolver.ImageUtils import image_to_vector

PATH = './temp_img.jpg'

def predict_digits(images_matrix):

    with open('./DigitRecognizer/fitted-model.pickle', 'rb') as file:
        # Deserialize and load the object from the file
        model: Pipeline = pickle.load(file)

    if(not model):
        return None
    
    images_list = []
    for row in images_matrix:
        for img in row:
            images_list.append(255 - img)

    prediction = model.predict(images_list)

    prediction = np.resize(prediction,(9,9))
    return prediction.tolist()
    
    
    