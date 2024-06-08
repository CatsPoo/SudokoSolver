import cv2
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing, metrics, pipeline, model_selection, feature_extraction 
from sklearn import naive_bayes, linear_model, svm, neural_network, neighbors, tree
from sklearn import decomposition, cluster
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score,make_scorer
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, SGDClassifier,LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier

from sklearn.exceptions import ConvergenceWarning
from SudokoSolver.ImageUtils import image_to_vector
import os

IMAGE_PATH = 'image_path'
NUMBER = 'number'

class Vectorizer(BaseEstimator, ClassifierMixin):
    def __init__(self,size=(64, 64), grayscale=True):
        self.size = size
        self.grayscale = grayscale
        
    def fit(self,x,y):
        return x,y
    
    def transform(self,x,y=None):
        X_transformed = x.copy()
        X_transformed['vectors'] = x[IMAGE_PATH].apply(image_to_vector, size=self.size, grayscale=self.grayscale)
        values_df = pd.DataFrame(X_transformed['vectors'].tolist(), index=X_transformed['vectors'].index)
        X_transformed = X_transformed.drop('vectors', axis=1).join(values_df)
        return X_transformed.drop(IMAGE_PATH,axis=1), y

    def fit_transform(self,x,y=None):
        x,y = self.fit(x,y)
        return self.transform(x,y)


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

        print('model name: {}, score: {}'.format(self.classifier_name,self.get_best_score()))
        return self

    def transform(self,x,y=None):
        return x,y

    def predicit(self,x):
        return self.classifier.ppredict(x)
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
    #Classifier('SVC',SVC(),{'C': [0.1,0.5, 1,3,5,10], 'kernel': ['linear', 'rbf']},MaxAbsScaler()),
    #Classifier('LinearSVC',LinearSVC(),{'C': [0.1,0.5, 1,3,5,10]},MaxAbsScaler()),
    Classifier('Random Forest',RandomForestClassifier(),{'n_estimators': [10,20,40,60, 100],'max_depth':[5,15,30,60,100]}),
    Classifier('Gradient Boosting',GradientBoostingClassifier(),{'n_estimators': [100, 200, 300],'learning_rate': [0.01, 0.1, 0.2],'max_depth': [3, 5, 7]}),
    Classifier('DecisionTreeClassifier',DecisionTreeClassifier(),{'max_depth':[12,28,30,31],'min_samples_split':[3,15,50,150,200]}),
    Classifier('KNeighborsClassifier',KNeighborsClassifier(),{'n_neighbors':[5,11,21,31,61,101],'p':[1,2]},MaxAbsScaler()),
    Classifier('MultinomialNB',MultinomialNB(),{'alpha': [0.01, 0.1, 1, 10],'fit_prior': [True, False],'class_prior': [None, [0.25, 0.25, 0.5], [0.5, 0.25, 0.25]],}),
    Classifier('GaussianNB',GaussianNB(),{'priors':[None, [0.3, 0.7], [0.4, 0.6], [0.7, 0.3]]}),
    Classifier('MLPClassifier',MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.0001,solver='adam', random_state=42),{})
]


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