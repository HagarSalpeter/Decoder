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
import os
import argparse
from utils import get_feature_names

parser = argparse.ArgumentParser()
parser.add_argument('--property-type', choices=['shape', 'position'],
                    default='position')
args = parser.parse_args()

df_features = pd.read_csv(os.path.join('..', 'output', 'training_features.csv'),
                          index_col=0)

#run the pipline once for shape and once for pose
feature_names = get_feature_names(args.property_type)
print(feature_names)

df_features = df_features[['fn_video'] + feature_names]# features
df_features = df_features.loc[df_features['fn_video'].str.contains(args.property_type, regex=False)]
print(df_features)
print(list(df_features))
#X = df.loc[(df['class'][:-3]==name)] #take only relevant rows
y = df_features['fn_video'] # target value
X = df_features.drop(['fn_video'], axis=1)

# Split the data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1234)



# Train different Classification Model

# Use different classifiers and see what works best
pipelines = {
    #'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    #'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(RandomForestClassifier()),
    #'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

# fit the models with the pipelines
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

# Evaluate and Serialize Model 
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(f'Performance for {args.property_type}, algrorithm {algo}, is: {accuracy_score(y_test, yhat)}')


# Put the trained model in a pkl file\
os.makedirs(os.path.join('..', 'trained_models'), exist_ok=True)
file_name = os.path.join('..','trained_models', f'trained_rf_{args.property_type}.pkl')
with open(file_name, 'wb') as f:
    pickle.dump([fit_models['rf'], feature_names], f)
