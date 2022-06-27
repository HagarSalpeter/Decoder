#!/usr/bin/env python
# coding: utf-8
import argparse
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import os
import csv
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils import extract_coordinates, extract_features
from utils import load_video

parser = argparse.ArgumentParser()
parser.add_argument('--show-video', action='store_true', default=False)
parser.add_argument('--path2data', default=os.path.join('..', 'data',
                                                        'training_videos'))
parser.add_argument('--path2output', default=os.path.join('..', 'output'))
args = parser.parse_args()
    

positions_list = ['position_00','position_01','position_02',
          'position_03','position_04'] 

shapes_list = ['shape_00','shape_01','shape_02','shape_03',
         'shape_04','shape_05','shape_06','shape_07']

classes_list = [positions_list, shapes_list]

df = pd.DataFrame()

for lst in classes_list:
    classes = lst

    for fn in classes:
        cap = load_video(fn)
        df_coords = extract_coordinates(cap)
        df = pd.concat([df,df_coords])

