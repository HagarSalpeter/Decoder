# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:36:16 2022

@author: hagar
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--test-video', default='test1.avi')
parser.add_argument('--path2models', default='')
parser.add_argument('--path2test_videos', default='')
args = parser.parse_args()

# MODELS
fn = f'trained_rf_position'
fn = os.path.join(args.path2models, fn)
model_position = load_model(fn)
fn = f'trained_rf_shape'
fn = os.path.join(args.path2models, fn)
model_shape = load_model(fn)

# INPUT VIDEO
fn_video = os.path.join(args.path2test_videos, args.test_video)
cap = load_video(fn_video)
csv_coords = extract_coordinates(cap)
csv_features = extract_features(csv_coord)

# PREDICT
compute_predictions(model_position, csv_features[pick_position_features])
print(f'csv file with predictions was saved to {}')
compute_predictions(model_shape, csv_features[pick_shape_features])
print(f'csv file with predictions was saved to {}')

