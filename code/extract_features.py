# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:23:06 2022

@author: hagar
"""

import pandas as pd
import os
import numpy as np
from utils import extract_features

# open csv with all the coords
df = pd.read_csv(os.path.join('..', 'output', 'all_coords_face_hand.csv'))


#create the df of relevant feature

df_features = df[['fn_video','frame_number',
                  'x_face0','y_face0','z_face0', #upper lip center
                  'x_face94','y_face94','z_face94', #nose
                  'x_face133','y_face133','z_face133', #inner left eye
                  'x_face362','y_face362','z_face362', #inner right eye
                  'x_face152','y_face152','z_face152', #tip of chin
                  'x_face234','y_face234','z_face234',
                  'x_face454','y_face454','z_face454',
                  'x_r_hand0','y_r_hand0','z_r_hand0',
                  'x_r_hand3','y_r_hand3','z_r_hand3',
                  'x_r_hand4','y_r_hand4','z_r_hand4',
                  'x_r_hand5','y_r_hand5','z_r_hand5',
                  'x_r_hand6','y_r_hand6','z_r_hand6',
                  'x_r_hand8','y_r_hand8','z_r_hand8', #tip of index finger
                  'x_r_hand9','y_r_hand9','z_r_hand9',
                  'x_r_hand12','y_r_hand12','z_r_hand12', #tip of middle finger
                  'x_r_hand13','y_r_hand13','z_r_hand13',
                  'x_r_hand16','y_r_hand16','z_r_hand16',
                  'x_r_hand17','y_r_hand17','z_r_hand17',
                  'x_r_hand20','y_r_hand20','z_r_hand20'
                  ]] #relevant cols

def axis_distance(df_name,col1,col2):
    return df_name[col1] - df_name[col2]

def coords_distance(df_name,d_x,d_y,d_z):
    return np.sqrt((df_name[d_x])**2 + (df_features[d_y])**2 + (df_features[d_z])**2)

#face width to normalize the distance
df_features['face_width_x'] = axis_distance(df_features,'x_face234','x_face454')
df_features['face_width_y'] = axis_distance(df_features,'y_face234','y_face454')
df_features['face_width_z'] = axis_distance(df_features,'z_face234','z_face454')
df_features['face_width'] = coords_distance(df_features,'face_width_x','face_width_y','face_width_z')

def normalized_axis_distance(df_name,col1,col2):
    return (df_name[col1] - df_name[col2])/df_name['face_width']

def normalized_coords_distance(df_name,d_x,d_y,d_z):
    return (np.sqrt((df_name[d_x])**2 + (df_features[d_y])**2 + (df_features[d_z])**2))/df_name['face_width']

#features for position 
#base of hand to upper lip center center   
df_features['d_x_face0_r_hand0'] = normalized_axis_distance(df_features,'x_r_hand0','x_face0')
df_features['d_y_face0_r_hand0'] = normalized_axis_distance(df_features,'y_r_hand0','x_face0')
df_features['d_z_face0_r_hand0'] = normalized_axis_distance(df_features,'z_r_hand0','z_face0')
df_features['distance_face0_r_hand0'] = normalized_coords_distance(df_features,'d_x_face0_r_hand0','d_y_face0_r_hand0','d_z_face0_r_hand0')
df_features['tan_alpha_pose0'] = df_features['d_y_face0_r_hand0']/df_features['d_x_face0_r_hand0'] # tan of alpha - the angle between the face center, hand and the horizontal axis

#top of index to inner left eye
df_features['d_x_face133_r_hand8'] = normalized_axis_distance(df_features,'x_r_hand8','x_face133')
df_features['d_y_face133_r_hand8'] = normalized_axis_distance(df_features,'y_r_hand8','x_face133')
df_features['d_z_face133_r_hand8'] = normalized_axis_distance(df_features,'z_r_hand8','z_face133')
df_features['distance_face133_r_hand8'] = normalized_coords_distance(df_features,'d_x_face133_r_hand8','d_y_face133_r_hand8','d_z_face133_r_hand8')
df_features['tan_alpha_pose1'] = df_features['d_y_face133_r_hand8']/df_features['d_x_face133_r_hand8'] # tan of alpha - the angle between the face center, hand and the horizontal axis

#top of index to inner right eye
df_features['d_x_face362_r_hand8'] = normalized_axis_distance(df_features,'x_r_hand8','x_face362')
df_features['d_y_face362_r_hand8'] = normalized_axis_distance(df_features,'y_r_hand8','x_face362')
df_features['d_z_face362_r_hand8'] = normalized_axis_distance(df_features,'z_r_hand8','z_face362')
df_features['distance_face362_r_hand8'] = normalized_coords_distance(df_features,'d_x_face362_r_hand8','d_y_face362_r_hand8','d_z_face362_r_hand8')
df_features['tan_alpha_pose2'] = df_features['d_y_face362_r_hand8']/df_features['d_x_face362_r_hand8'] # tan of alpha - the angle between the face center, hand and the horizontal axis

#top of index to chin
df_features['d_x_face152_r_hand8'] = normalized_axis_distance(df_features,'x_r_hand8','x_face152')
df_features['d_y_face152_r_hand8'] = normalized_axis_distance(df_features,'y_r_hand8','x_face152')
df_features['d_z_face152_r_hand8'] = normalized_axis_distance(df_features,'z_r_hand8','z_face152')
df_features['distance_face152_r_hand8'] = normalized_coords_distance(df_features,'d_x_face152_r_hand8','d_y_face152_r_hand8','d_z_face152_r_hand8')
df_features['tan_alpha_pose3'] = df_features['d_y_face152_r_hand8']/df_features['d_x_face152_r_hand8'] # tan of alpha - the angle between the face center, hand and the horizontal axis

#top of index to nose
df_features['d_x_face94_r_hand8'] = normalized_axis_distance(df_features,'x_r_hand8','x_face94')
df_features['d_y_face94_r_hand8'] = normalized_axis_distance(df_features,'y_r_hand8','x_face94')
df_features['d_z_face94_r_hand8'] = normalized_axis_distance(df_features,'z_r_hand8','z_face94')
df_features['distance_face94_r_hand8'] = normalized_coords_distance(df_features,'d_x_face94_r_hand8','d_y_face94_r_hand8','d_z_face94_r_hand8')
df_features['tan_alpha_pose4'] = df_features['d_y_face94_r_hand8']/df_features['d_x_face94_r_hand8'] # tan of alpha - the angle between the face center, hand and the horizontal axis

#top of middle to inner left eye
df_features['d_x_face133_r_hand12'] = normalized_axis_distance(df_features,'x_r_hand12','x_face133')
df_features['d_y_face133_r_hand12'] = normalized_axis_distance(df_features,'y_r_hand12','x_face133')
df_features['d_z_face133_r_hand12'] = normalized_axis_distance(df_features,'z_r_hand12','z_face133')
df_features['distance_face133_r_hand12'] = normalized_coords_distance(df_features,'d_x_face133_r_hand12','d_y_face133_r_hand12','d_z_face133_r_hand12')
df_features['tan_alpha_pose5'] = df_features['d_y_face133_r_hand12']/df_features['d_x_face133_r_hand12'] # tan of alpha - the angle between the face center, hand and the horizontal axis

#top of index to middle right eye
df_features['d_x_face362_r_hand12'] = normalized_axis_distance(df_features,'x_r_hand12','x_face362')
df_features['d_y_face362_r_hand12'] = normalized_axis_distance(df_features,'y_r_hand12','x_face362')
df_features['d_z_face362_r_hand12'] = normalized_axis_distance(df_features,'z_r_hand12','z_face362')
df_features['distance_face362_r_hand12'] = normalized_coords_distance(df_features,'d_x_face362_r_hand12','d_y_face362_r_hand12','d_z_face362_r_hand12')
df_features['tan_alpha_pose6'] = df_features['d_y_face362_r_hand12']/df_features['d_x_face362_r_hand12'] # tan of alpha - the angle between the face center, hand and the horizontal axis

#top of middle to chin
df_features['d_x_face152_r_hand12'] = normalized_axis_distance(df_features,'x_r_hand12','x_face152')
df_features['d_y_face152_r_hand12'] = normalized_axis_distance(df_features,'y_r_hand12','x_face152')
df_features['d_z_face152_r_hand12'] = normalized_axis_distance(df_features,'z_r_hand12','z_face152')
df_features['distance_face152_r_hand12'] = normalized_coords_distance(df_features,'d_x_face152_r_hand12','d_y_face152_r_hand12','d_z_face152_r_hand12')
df_features['tan_alpha_pose7'] = df_features['d_y_face152_r_hand12']/df_features['d_x_face152_r_hand12'] # tan of alpha - the angle between the face center, hand and the horizontal axis

#top of middle to nose
df_features['d_x_face94_r_hand12'] = normalized_axis_distance(df_features,'x_r_hand12','x_face94')
df_features['d_y_face94_r_hand12'] = normalized_axis_distance(df_features,'y_r_hand12','x_face94')
df_features['d_z_face94_r_hand12'] = normalized_axis_distance(df_features,'z_r_hand12','z_face94')
df_features['distance_face94_r_hand12'] = normalized_coords_distance(df_features,'d_x_face94_r_hand12','d_y_face94_r_hand12','d_z_face94_r_hand12')
df_features['tan_alpha_pose8'] = df_features['d_y_face94_r_hand12']/df_features['d_x_face94_r_hand12'] # tan of alpha - the angle between the face center, hand and the horizontal axis

#features for shape
pairs = [('8','5'),('12','9'),('16','13'),('17','20'),('4','6'),('3','5'),('8','12'),('7','11')]
features_pairs =[]
names = ['x_r_hand','y_r_hand','z_r_hand']

for pair in pairs:
    for name in names:
        features_pairs.append([name+pair[0], name+pair[1]])
    
#get delta features
deltas =[]    
for i in features_pairs:
    df_features[f'd_{i[0]}_{i[1]}'] = normalized_axis_distance(df_features,i[0],i[1])
    deltas.append(f'd_{i[0]}_{i[1]}')

delta_triplets = [['d_x_r_hand8_x_r_hand5', 'd_y_r_hand8_y_r_hand5', 'd_z_r_hand8_z_r_hand5','d_r_hand8_r_hand5'],
                  ['d_x_r_hand12_x_r_hand9', 'd_y_r_hand12_y_r_hand9', 'd_z_r_hand12_z_r_hand9','d_r_hand12_r_hand9'],
                  ['d_x_r_hand16_x_r_hand13', 'd_y_r_hand16_y_r_hand13', 'd_z_r_hand16_z_r_hand13','d_r_hand16_r_hand13'], 
                  ['d_x_r_hand17_x_r_hand20', 'd_y_r_hand17_y_r_hand20', 'd_z_r_hand17_z_r_hand20','d_r_hand17_r_hand20'], 
                  ['d_x_r_hand4_x_r_hand6', 'd_y_r_hand4_y_r_hand6', 'd_z_r_hand4_z_r_hand6','d_r_hand4_r_hand6'],
                  ['d_x_r_hand3_x_r_hand5', 'd_y_r_hand3_y_r_hand5', 'd_z_r_hand3_z_r_hand5','d_r_hand3_r_hand5'],
                  ['d_x_r_hand8_x_r_hand12', 'd_y_r_hand8_y_r_hand12', 'd_z_r_hand8_z_r_hand12','d_r_hand8_r_hand12'],
                  ['d_x_r_hand7_x_r_hand11', 'd_y_r_hand7_y_r_hand11', 'd_z_r_hand7_z_r_hand11','d_r_hand7_r_hand11']
                  ]

#get distance features
for j in delta_triplets:
    df_features[j[3]] = normalized_coords_distance(df_features,j[0],j[1],j[2])

    

# extract the df to a csv file
df_features.to_csv(os.path.join('..', 'output', 'training_features.csv'))
