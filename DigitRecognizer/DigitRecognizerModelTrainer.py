import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score 
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin,TransformerMixin
from sklearn.neural_network import MLPClassifier
from SudokoSolver.ImageUtils import image_to_vector,convert_image_to_gray_sale
from SudokoSolver.Utils import get_list_shape
from sklearn.metrics import f1_score,make_scorer
from sklearn.model_selection import GridSearchCV
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical


IMAGE_PATH = 'image_path'
NUMBER = 'number'

class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, ):
        self.dataset = None
        
    def fit(self,x,y=None):
        return self
    
    def transform(self,x,y=None):
        X_transformed = np.empty((len(x),28,28))
        for i,row in enumerate(x):
            current_image = np.array(x[i])
            X_transformed[i] = self.vectorization_function(current_image)
            
        X_transformed = np.reshape(X_transformed,(X_transformed.shape[0], 28, 28, 1)).astype('float32')
        X_transformed = X_transformed/255
        self.dataset = X_transformed
        return X_transformed
    
    def vectorization_function(self,img):
        if(len(img.shape)!=2):
            img = convert_image_to_gray_sale(img)
        if(img.shape != (28,28)):
            return cv2.resize(img, (28,28), interpolation = 1)
        return img

    def set_invert_image(self,val):
        self.invert_image = val
    
    def set_debug(self,val):
        self.debug = val



class FitOptimezedModel(BaseEstimator, ClassifierMixin):
    def __init__(self,classifier_name,classifier):
        self.classifier_name = classifier_name
        #self.cv_count = cv_count
        self.classifier = classifier
        self.score=0
        #self.cv_result = 0
        #self.grid_search_cv = GridSearchCV(self.classifier, self.params, cv=self.cv_count,scoring=make_scorer(f1_score, average='micro'))

    
    def fit(self,X,y):
        if(isinstance(X, tuple)):
            x=x[0]

        X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state= 21)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        
        print('Starting fit model {} .....'.format(self.classifier_name))
        
        model = larger_model()
        log_dir = "logs/fit/"

        self.classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=10,callbacks=[])
        # Final evaluation of the model
        self.score = self.classifier.evaluate(X_test, y_test, verbose=0)

        print("Large CNN Error: %.2f%%" % (100-self.score[1]*100))
        return self

    def predict(self,x):
        res = []
        pred = self.classifier.predict(x,verbose = 1)
        for p in pred:
            res.append(p.argmax())
        return np.array(res)
    
        
    def get_best_score(self):
        return self.score[1]
        #self.grid_search_cv.best_score_

    def get_model_name(self):
        return self.classifier_name


    

def generate_model_pipeline():

    classifier_name = "model"
    classifier = larger_model()

    return Pipeline(steps=[
    ('vectorizer', Vectorizer()),
    ('model_trainer',FitOptimezedModel(classifier_name,classifier))
    ])
           

def larger_model():
        # create model
        model = keras.Sequential(
        [
            Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'),
            MaxPooling2D(),
            Conv2D(15, (3, 3), activation='relu'),
            MaxPooling2D(),
            Dropout(0.2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(50, activation='relu'),
            Dense(15, activation='softmax')
        ])
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


def build_dataset(base_dir):
    X = []
    y = []
    for dir in os.listdir(base_dir):
        if(dir == '10'): continue

        current_folder_path = os.path.join(base_dir,dir)
        for img_path in os.scandir(current_folder_path):
            img = image_to_vector(os.path.join(current_folder_path,str(img_path.name)))
            X.append(img)
            y.append(int(dir))
    X = np.array(X)
    y = np.array(y)
    return X,y
