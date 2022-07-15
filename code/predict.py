# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:36:16 2022

@author: hagar
"""

import argparse
import os
import numpy as np
import pandas as pd
from utils import extract_coordinates, extract_features, extract_class_from_fn
from utils import compute_predictions, load_model, load_video

parser = argparse.ArgumentParser()
parser.add_argument('--property-type', choices=['shape', 'position'],
                    default='shape')
parser.add_argument('--model-type', choices=['rf', 'lr', 'rc', 'gb'],
                    help = 'rf:random-forest; lr:logisitic-regrssion',
                    default='rf')
parser.add_argument('--fn-video', default='word_h0_01.mp4')
parser.add_argument('--path2models',
                    default=os.path.join('..', 'trained_models')) 
parser.add_argument('--path2test_videos',
                    default=os.path.join('..','data','test_videos'))
parser.add_argument('--path2output', default=os.path.join('..', 'output')) 
parser.add_argument('--save-features', action='store_true', default=True) 

args = parser.parse_args()

# LOAD MODEL
fn_model = f'model_{args.model_type}_{args.property_type}.pkl'
fn_model = os.path.join(args.path2models, fn_model)
model, feature_names = load_model(fn_model)
print(f'Loaded model: {fn_model}')
print(f'Trained on features: {feature_names}')

# LOAD VIDEO
fn_video = os.path.join(args.path2test_videos, args.fn_video)
cap = load_video(fn_video)
print(f'Loaded video: {fn_video}')

# EXTRACT COORDINATES
print('Extracting coordinates...')
df_coords = extract_coordinates(cap, args.fn_video)
    
# EXTRACT FEATURES
print('Extracting features...')
df_features = extract_features(df_coords)
if args.save_features:
    df_features.to_csv(os.path.join(args.path2output,
                                    f'{args.fn_video[:-4]}_features.csv'))
    print(f"Features saved to: {os.path.join(args.path2output, f'{args.fn_video[:-4]}_features.csv')}")
    
# PREDICT
predicted_probs, predicted_class = compute_predictions(model,
                                                       df_features[feature_names])
df_predictions = pd.DataFrame()
df_predictions['frame_number'] = df_features['frame_number']
df_predictions['predicted_class'] = predicted_class
df_predictions['predicted_class'] = df_predictions.apply(lambda row: extract_class_from_fn(row['predicted_class']),
                                                                                           axis=1)
for c in range(predicted_probs.shape[1]):
    df_predictions[f'p_class_{c+1}'] = predicted_probs[:, c]

# SAVE
fn_predictions = f'predictions_{args.model_type}_{args.property_type}_{args.fn_video[:-4]}.csv'
fn_predictions = os.path.join(args.path2output, fn_predictions)
df_predictions.to_csv(fn_predictions)
print(df_predictions)
print(f'csv file with predictions was saved to {fn_predictions}')
