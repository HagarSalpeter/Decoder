# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:36:16 2022

@author: hagar
"""

import argparse
import os
import numpy as np
from utils import extract_coordinates, extract_features
from utils import compute_predictions, load_model, load_video 

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', default='trained_rf_position')
parser.add_argument('--test-video', default='test.mp4')
parser.add_argument('--path2models',
                    default=os.path.join('..', 'trained_models')) 
parser.add_argument('--path2test_videos',
                    default=os.path.join('..','data','test_videos'))
parser.add_argument('--path2output', default=os.path.join('..', 'output')) 
parser.add_argument('--save-feature-csv', action='store_true', default=False) 

args = parser.parse_args()

# MODELS
#fn_model = f'trained_rf_{args.property_type}'
fn_model = os.path.join(args.path2models, args.model_name + '.pkl')
model, feature_names = load_model(fn_model)
print(f'Loaded model: {fn_model}')
print(f'Trained on features: {feature_names}')

# INPUT VIDEO
fn_video = os.path.join(args.path2test_videos, args.test_video)
cap = load_video(fn_video)
print(f'Loaded video: {fn_video}')

# COORDINATES
df_coords = extract_coordinates(cap,fn_video)
#fn_csv_coords = os.path.join(args.path2output, f'{fn_video}_all_coords.csv.csv')
print('Extracted coordinates:')
print(df_coords)
    
# FEATURES
df_features = extract_features(df_coords)
if args.save_feature_csv:
    df_features.to_csv(os.path.join(args.path2output, f'{fn_video}_features.csv'))
print('Extracted features:')
print(df_features)


# PREDICT
predicted_probs, predicted_classes = compute_predictions(model, df_features[feature_names])
print(predicted_probs.shape, predicted_classes.shape)
output = np.concatenate([predicted_probs, predicted_classes[:, None]], axis=1)

# SAVE
fn_predictions = os.path.join(args.path2output,
        f'{os.path.basename(fn_model)[:-4]}_{os.path.basename(fn_video)[:-4]}_predicted.csv')

print(f'csv file with predictions was saved to {fn_predictions}')
np.savetxt(fn_predictions, output, delimiter=',')
#predictions.to_csv(fn_predictions)

