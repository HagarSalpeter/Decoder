# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:36:16 2022

@author: hagar
"""

imports...
argparse part...(args variable containing all user choices)

# MODELS
model_position = load_model(args.path2models, 'position', ...)
model_shape = load_model(args.path2models, 'shape', ...)

# INPUT VIDEO
cap = load_video(args.path2...)
csv_coords = extract_coordinates(cap)
csv_features = extract_features(csv_coord)

# PREDICT
predictions_position, accuracy_position = compute_predictions(model_position, csv_features[pick_position_features])
predictions_shape, accuracy_shape = compute_predictions(model_shape, csv_features[pick_shape_features])

# SAVE STUFF..
# need to see how to merge both files