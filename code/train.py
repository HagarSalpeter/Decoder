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

parser = argparse.ArgumentParser()
parser.add_argument('--property-type', choices=['shape', 'position'],
                    default='position')
args = parser.parse_args()

df = pd.read_csv(os.path.join('..', 'output', 'training_features.csv'))
df.drop('Unnamed: 0', axis=1)
# Separate the features from the target
# features for position: 
pos_features = ['fn_video','d_x_face0_r_hand0','d_y_face0_r_hand0','d_z_face0_r_hand0',
                'distance_face0_r_hand0','tan_alpha_pose']
# features for shape: 
shape_features = ['fn_video','d_x_r_hand8_x_r_hand5', 'd_y_r_hand8_y_r_hand5', 'd_z_r_hand8_z_r_hand5','d_r_hand8_r_hand5',
                  'd_x_r_hand12_x_r_hand9', 'd_y_r_hand12_y_r_hand9', 'd_z_r_hand12_z_r_hand9','d_r_hand12_r_hand9',
                  'd_x_r_hand16_x_r_hand13', 'd_y_r_hand16_y_r_hand13', 'd_z_r_hand16_z_r_hand13','d_r_hand16_r_hand13', 
                  'd_x_r_hand17_x_r_hand20', 'd_y_r_hand17_y_r_hand20', 'd_z_r_hand17_z_r_hand20','d_r_hand17_r_hand20', 
                  'd_x_r_hand4_x_r_hand6', 'd_y_r_hand4_y_r_hand6', 'd_z_r_hand4_z_r_hand6','d_r_hand4_r_hand6',
                  'd_x_r_hand3_x_r_hand5', 'd_y_r_hand3_y_r_hand5', 'd_z_r_hand3_z_r_hand5','d_r_hand3_r_hand5',
                  'd_x_r_hand8_x_r_hand12', 'd_y_r_hand8_y_r_hand12', 'd_z_r_hand8_z_r_hand12','d_r_hand8_r_hand12']

feature_lists = {'position':pos_features,
                 'shape':shape_features}

#run the pipline once for shape and once for pose
feature_names = feature_lists[args.property_type] 
   
X = df[feature_names]# features
X = df[df['fn_video'].str.contains(args.property_type, regex=False)]
#X = df.loc[(df['class'][:-3]==name)] #take only relevant rows
y = X['fn_video'] # target value
X = X.drop(['fn_video', 'Unnamed: 0'], axis=1)



# Split the data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1234)



# Train different Classification Model

# Use different classifiers and see what works best
pipelines = {
    #'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    #'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
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
    print(algo, accuracy_score(y_test, yhat))


# Put the trained model in a pkl file\
os.makedirs(os.path.join('..', 'trained_models'), exist_ok=True)
file_name = os.path.join('..','trained_models', f'trained_rf_{args.property_type}.pkl')
with open(file_name, 'wb') as f:
    pickle.dump([fit_models['rf'], feature_names], f)