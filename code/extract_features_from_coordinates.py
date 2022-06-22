# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:23:06 2022

@author: hagar
"""

import pandas as pd
import os
import numpy as np

# open csv with all the coords
df = pd.read_csv(os.path.join('all_the_coords_face_hand.csv'))

coll_names = ['class']
num_coords_face = 468
num_coords_hand = 21
num_coords_pose = 33

# generate collomns names
for val in range(0, num_coords_face):
    coll_names += ['x_face{}'.format(val), 'y_face{}'.format(val), 'z_face{}'.format(val), 'v_face{}'.format(val)]

for val in range(0, num_coords_hand):
    coll_names += ['x_r_hand{}'.format(val), 'y_r_hand{}'.format(val), 'z_r_hand{}'.format(val), 'v_r_hand{}'.format(val)]

#for val in range(0, num_coords_pose):
#    coll_names += ['x_pose{}'.format(val), 'y_pose{}'.format(val), 'z_pose{}'.format(val), 'v_pose{}'.format(val)]

df.columns = coll_names #change the coll names of the df

#create the df of relevant feature

df_features = df[['class','x_face0','y_face0','z_face0','v_face0','x_r_hand0','y_r_hand0','z_r_hand0','v_r_hand0']] #tack relevant cols - face0 is upper lip center, hand0 is the base of the hand
df_features['d_x'] = df_features['x_face0'] - df_features['x_r_hand0'] # distance in x axis
df_features['d_y'] = df_features['y_face0'] - df_features['y_r_hand0'] # distance in y axis
df_features['d_z'] = df_features['z_face0'] - df_features['z_r_hand0'] # distance in z axis
df_features['tan_alpha'] = df_features['d_y']/df_features['d_x'] # tan of alpha - the angle between the face center, hand and the horizontal axis
df_features['distance_hand_face'] = np.sqrt((df_features['d_x'])**2 + (df_features['d_y'])**2 + (df_features['d_z'])**2) # the distance between the palm and the face

# extract the df to a csv file
df_features.to_csv('training_features.csv')
