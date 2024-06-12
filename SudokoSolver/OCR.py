import cv2
import pickle
from sklearn.pipeline import Pipeline 
import pandas as pd
import os

PATH = './temp_img.jpg'

def predict_digit(img):

    with open('./SudokoSolver/best-model.pickle', 'rb') as file:
        # Deserialize and load the object from the file
        model: Pipeline = pickle.load(file)

    if(not model):
        return None
    
    cv2.imwrite(PATH,img)
    predict_dict = {'image_path' : [PATH]}
    predict_df = pd.DataFrame(predict_dict)
    
    prediction = model.predict(predict_df)
    print(prediction)
    os.remove(PATH)
    
    
    