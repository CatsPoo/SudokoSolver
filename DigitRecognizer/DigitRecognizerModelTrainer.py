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
from sklearn.metrics import precision_score, recall_score, f1_score
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
import warnings
from sklearn.exceptions import ConvergenceWarning
from ..ImageUtils import image_to_vector
import os
class Vectorizer(BaseEstimator, ClassifierMixin):
    def __init__(self,size=(64, 64), grayscale=True):
        self.size = size
        self.grayscale = grayscale
        
    def fit(self,x,y):
        return self
    
    def transform(self,x,y):
        vectors = x['image_url'].apply(image_to_vector, size=self.size, grayscale=self.grayscale)
        X_transformed = pd.concat([x[['id']], vectors], axis=1)
        return X_transformed, y

    def fit_transform(self,x,y):
        self.fit(x,y)
        return self.transform(x,y)


class FitOptimezedModel(BaseEstimator, ClassifierMixin):
    def __init__(self,classifier_name,classifier,params,cv_count):
        self.classifier_name = classifier_name
        self.cv_count = cv_count
        self.classifier = classifier
        self.params = params
        self.grid_search_cv = GridSearchCV(self.classifier, self.params, cv=self.cv_count,scoring='f1')

    
    def fit(self,x,y):
        if(isinstance(x, tuple)):
            x=x[0]
        self.grid_search_cv.fit(x,y)
        print('model name: {}, score: {}'.format(self.classifier_name,self.get_best_score()))
        return self

    def transform(self,x,y):
        return x,y

    def predicit(self,x):
        return self.grid_search_cv.best_estimator_.predict(x)
        
    def get_best_score(self):
        return self.grid_search_cv.best_score_

    def get_best_params(self):
        return self.grid_search_cv.best_params_

    def get_model_name(self):
        return self.classifier_name

    

def generate_model_training_pipeline(
    classifier_name,
    classifier,
    params,
    vectorizer = None,
    scaller = None,
    cv_count = 5,
    max_features=10000,
    ngram_range=(1,1),):

    return Pipeline(steps=[
    ('vectorizer', Vectorizer()),
    ('model_trainer',FitOptimezedModel(classifier_name,classifier,params,cv_count))
    ])


def build_dataset(base_dir):
    for dir in os.listdir(base_dir):
        print(dir)

def train():
    pass

if(__name__ == '__main__'):
    build_dataset(os.path.join('.','assets'))
    train()