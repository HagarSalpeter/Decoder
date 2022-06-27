# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:36:16 2022

@author: hagar
"""

imports...
argparse part...(args variable containing all user choices)

# MODELS
fn = f'trained_rf_position'
fn = os.path.join(args.path2models, fn)
model_position = load_model(fn)
fn = f'trained_rf_shape'
fn = os.path.join(args.path2models, fn)
model_shape = load_model(fn)

# INPUT VIDEO
fn_video = 'test1.avi'
fn_video = os.path.join(args.path2test_videos, fn_video)
cap = load_video(fn_video)
csv_coords = extract_coordinates(cap)
csv_features = extract_features(csv_coord)

# PREDICT
predictions_position, accuracy_position = compute_predictions(model_position, csv_features[pick_position_features])
predictions_shape, accuracy_shape = compute_predictions(model_shape, csv_features[pick_shape_features])

# SAVE STUFF..
# need to see how to merge both files
