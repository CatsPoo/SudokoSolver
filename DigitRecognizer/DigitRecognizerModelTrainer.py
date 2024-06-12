import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score 
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin,TransformerMixin
from sklearn.neural_network import MLPClassifier
from SudokoSolver.ImageUtils import image_to_vector
import os

IMAGE_PATH = 'image_path'
NUMBER = 'number'

class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self,size=(64, 64), grayscale=True,invert_image = False):
        self.size = size
        self.grayscale = grayscale
        self.invert_image = invert_image
        self.dataset = None
        
    def fit(self,x,y=None):
        return self
    
    def transform(self,x,y=None):
        X_transformed = x.copy()
        X_transformed['vectors'] = x[IMAGE_PATH].apply(image_to_vector, size=self.size, grayscale=self.grayscale,invert_image=self.invert_image)
        values_df = pd.DataFrame(X_transformed['vectors'].tolist(), index=X_transformed['vectors'].index)
        X_transformed = X_transformed.drop('vectors', axis=1).join(values_df)
        X_transformed = X_transformed.drop(IMAGE_PATH,axis=1)
        self.dataset = X_transformed
        return X_transformed
    
    def set_invert_image(self,val):
        self.invert_image = val



class FitOptimezedModel(BaseEstimator, ClassifierMixin):
    def __init__(self,classifier_name,classifier,params,cv_count):
        self.classifier_name = classifier_name
        self.cv_count = cv_count
        self.classifier = classifier
        self.params = params
        self.cv_result = 0
        #self.grid_search_cv = GridSearchCV(self.classifier, self.params[0], cv=self.cv_count,scoring=make_scorer(f1_score, average='micro'))

    
    def fit(self,x,y):
        if(isinstance(x, tuple)):
            x=x[0]

        temp_df = pd.concat([x,y],axis=1)
        temp_df = temp_df.sample(frac=1).reset_index(drop=True)
        y = temp_df[NUMBER]
        x = temp_df.drop(NUMBER,axis=1) 

        print('Starting fit model {} .....'.format(self.classifier_name))
        self.cv_result = cross_val_score(self.classifier,x,y,cv=self.cv_count)
        self.classifier.fit(x, y)
        print('model name: {}, score: {}'.format(self.classifier_name,self.get_best_score()))
        return self

    def predict(self,x):
        return self.classifier.predict(x)
        #return self.grid_search_cv.best_estimator_.predict(x)
        
    def get_best_score(self):
        return self.cv_result.mean()

    def get_model_name(self):
        return self.classifier_name

    

def generate_model_training_pipeline(
    classifier_name,
    classifier,
    params,
    cv_count = 5,):

    return Pipeline(steps=[
    ('vectorizer', Vectorizer()),
    ('model_trainer',FitOptimezedModel(classifier_name,classifier,params,cv_count))
    ])

def generate_piplines_from_classifiers_list(classifiers_list):   
    models_piplines = []
    for classifiers in classifiers_list:
        models_piplines.append(generate_model_training_pipeline(
            classifier_name = classifiers.clf_name,
            classifier = classifiers.clf,
            params = classifiers.params,

        ))
    return models_piplines

class Classifier:
    def __init__(self,clf_name,clf,params,scaler=None):
        self.clf_name = clf_name
        self.clf = clf
        self.params = params,
        self.scaler = scaler        

def fit_piplines(pipes,x,y):
    for pipe in pipes:
        pipe.fit_transform(x,y)

classifiers = [
    # Classifier('Random Forest',RandomForestClassifier(n_estimators=20),{}),
    Classifier('Random Forest',RandomForestClassifier(n_estimators=5000),{}),
    Classifier('MLPClassifier',MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=600, alpha=0.0001,solver='adam', random_state=42,activation='relu'),{}),
    Classifier('MLPClassifier',MLPClassifier(hidden_layer_sizes=(128,64), max_iter=600, alpha=0.0001,solver='adam', random_state=42,activation='relu'),{}),
    Classifier('MLPClassifier',MLPClassifier(hidden_layer_sizes=(64,32), max_iter=600, alpha=0.0001,solver='adam', random_state=42,activation='relu'),{}),
    Classifier('MLPClassifier',MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=600, alpha=0.0001,solver='adam', random_state=42,activation='relu'),{}),
    Classifier('MLPClassifier',MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=600, alpha=0.0001,solver='adam', random_state=42,activation='relu'),{}),
    Classifier('MLPClassifier',MLPClassifier(hidden_layer_sizes=(100,100), max_iter=600, alpha=0.0001,solver='adam', random_state=42,activation='relu'),{}),
    Classifier('MLPClassifier',MLPClassifier(hidden_layer_sizes=(100,70,50), max_iter=600, alpha=0.0001,solver='adam', random_state=42,activation='relu'),{}),
    Classifier('MLPClassifier',MLPClassifier(hidden_layer_sizes=(100,50,), max_iter=600, alpha=0.0001,solver='adam', random_state=42,activation='relu'),{}),
    Classifier('MLPClassifier',MLPClassifier(hidden_layer_sizes=(100,100,100,), max_iter=600, alpha=0.0001,solver='adam', random_state=42,activation='relu'),{}),
]

def get_best_model_pipeline(piplines_list):
    best_model_score = 0
    best_model_pipe = None

    for pipe in piplines_list:
        current_model_score = pipe['model_trainer'].get_best_score()
        if(current_model_score > best_model_score):
            best_model_score = current_model_score
            best_model_pipe = pipe
            
    return best_model_pipe
        


def build_dataset(base_dir):
    dataset_dict = {
        IMAGE_PATH: [],
        NUMBER:[]}
    for dir in os.listdir(base_dir):
        if(dir == 10): continue

        current_folder_path = os.path.join(base_dir,dir)
        for img in os.scandir(current_folder_path):
            dataset_dict[IMAGE_PATH].append(os.path.join(current_folder_path,str(img.name)))
            dataset_dict[NUMBER].append(int(dir))
    return pd.DataFrame(dataset_dict)


def train():
    df = build_dataset(os.path.join('DigitRecognizer','assets'))
    pipes = generate_piplines_from_classifiers_list(classifiers)
    
    for pipe in pipes:
        pipe.fit_transform(df[IMAGE_PATH],df[NUMBER])



# if(__name__ == '__main__'):
#     build_dataset(os.path.join('.','assets'))
#     train()