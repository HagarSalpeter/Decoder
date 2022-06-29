# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:08:44 2022

@author: hagar
"""
import argparse
import os
import pandas as pd

from utils import load_model, load_video, mark_pred_on_video

parser = argparse.ArgumentParser()
parser.add_argument('--model-type', choices=['rf', 'lr', 'rc', 'gb'],
                    help = 'rf:random-forest; lr:logisitic-regrssion',
                    default='rf')
parser.add_argument('--fn-video', default='test.mp4')
parser.add_argument('--path2video', default=os.path.join('..', 'data',
                                                         'test_videos'))
parser.add_argument('--path2predictions', default=os.path.join('..',
                                                               'output')) 
parser.add_argument('--show-video', action='store_true', default=False)
args = parser.parse_args()

# LOAD VIDEO
fn_video = os.path.join(args.path2video, args.fn_video)
cap = load_video(fn_video)
print(f'Visualization for: {fn_video}')
print(cap.__sizeof__())
# LOAD PREDICTIONS
fn_predictions_pos = f'predictions_{args.model_type}_position_{args.fn_video[:-4]}.csv'
df_predictions_pos = pd.read_csv(os.path.join(args.path2predictions, fn_predictions_pos))
fn_predictions_shape = f'predictions_{args.model_type}_shape_{args.fn_video[:-4]}.csv'
df_predictions_shape = pd.read_csv(os.path.join(args.path2predictions, fn_predictions_shape))


print(df_predictions_pos, df_predictions_shape)
mark_pred_on_video(cap, fn_video,
                   df_predictions_pos, df_predictions_shape,
                   show=args.show_video)

