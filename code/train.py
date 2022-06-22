# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:06:34 2022

@author: hagar
"""

# Train Model Using Scikit Learn

# import relevant modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle # file to save the trained model



df = pd.read_csv(os.path.join('..', '..', 'data', 'training_features.csv'))

# Separate the features from the target
X = df.drop('class', axis=1) # features
y = df['class'] # target value



# Split the data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)



# Train train different Classification Model

# Use different classifiers and see what works best
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}



# fit the models with the pipelines
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model



# Evaluate and Serialize Model 

for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))


# Put the trained model in a pkl file
file_name = os.path.join('..', '..', 'data', 'training_features.csv')
with open('trained_rf.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)