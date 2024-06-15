import cv2
import pickle
from sklearn.pipeline import Pipeline 
import pandas as pd
import os
import numpy as np
from SudokoSolver.ImageUtils import image_to_vector

PATH = './temp_img.jpg'

def predict_digit(img):

    with open('./DigitRecognizer/fitted-model.pickle', 'rb') as file:
        # Deserialize and load the object from the file
        model: Pipeline = pickle.load(file)

    if(not model):
        return None

    import numpy as np

    vectorsArr = []
    vectorsArr.append(image_to_vector(img))

    vectorsArr = np.array(vectorsArr)
    vectorsArr[-1] = 1 - vectorsArr[-1]


    prediction = model.predict(vectorsArr)[0]
    return prediction
    
    
    