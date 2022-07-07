#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 21:20:23 2022

@author: yl254115
"""

import pickle
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt


def open_cartoon(fn_cartoon):
    cartoon = cv2.imread(fn_cartoon, cv2.IMREAD_COLOR)
    return cartoon


def mark_pred_on_video(cap, fn_video,
                       df_predictions_pos, df_predictions_shape,
                       velocity,
                       p_thresh=0.5,
                       show=False):
    
    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions
    
    size = (int(cap.get(3)), int(cap.get(4)))
    print(size)
    marked_video = cv2.VideoWriter(f'{fn_video[:-4]}_marked.avi',
                                   cv2.VideoWriter_fourcc(*'XVID'),30,
                                    size)
    
    n_frames = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=n_frames)
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        i_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # POSITION
            predicted_class_pos = df_predictions_pos.iloc[[i_frame]]['predicted_class'].values[0]
            predicted_probs_pos = df_predictions_pos.iloc[[i_frame]][f'p_class_{predicted_class_pos + 1}'].values[0]
            
            # SHAPE
            predicted_class_shape = df_predictions_shape.iloc[[i_frame]]['predicted_class'].values[0]
            predicted_probs_shape = df_predictions_shape.iloc[[i_frame]][f'p_class_{predicted_class_shape + 1}'].values[0]
            
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                     )
            
            # Right hand landmarks
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
            
       
            # Write prediction on video:
            # font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Get status box
            # cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # # Display Class
            # cv2.putText(image, 'Position',
            #              (95,12), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            # cv2.putText(image, 'Shape',
            #              (15,12), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            
            
            if predicted_probs_pos > p_thresh and predicted_probs_shape > p_thresh and velocity[i_frame]<3e-3:
                fn_cartoon = f'pos_n{predicted_class_pos}_shape_n{predicted_class_shape}.png' 
                fn_cartoon = os.path.join('../data/cartoons/', fn_cartoon)
                cartoon = open_cartoon(fn_cartoon)
                if cartoon is None:
                    raise("Cartoon image not found")
                # cv2.putText(image, str(predicted_class_pos),
                #          (90,40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # cv2.putText(image, str(predicted_class_shape),
                #          (15,40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # cv2.addWeighted(image, alpha, cartoon, 1-alpha, 0)
                # cartoon = image_resize(cartoon, height = 100)
                height, width, channels = cartoon.shape
                offset = np.array((100, 80)) #top-left point from which to insert the smallest image. height first, from the top of the window
                image[offset[0]:offset[0] + height,
                       offset[1]:offset[1] + width] = cartoon
    
                
            # if predicted_probs_shape > p_thresh:
                    
    
            if show:
                cv2.imshow('cued_estimated', image)
            # print(image)
            marked_video.write(image)
    
    
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            i_frame += 1
            pbar.update(1)
    
    marked_video.release()
    cap.release()
    cv2.destroyAllWindows()
    print("The video was successfully saved")


def plot_predictions(df_predictions_pos, df_predictions_shape,
                     thresh=0.5):
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    df_predictions_pos = df_predictions_pos.filter(regex=("p_class*"))
    df_predictions_shape = df_predictions_shape.filter(regex=("p_class*"))
    
    probs_pos = df_predictions_pos.to_numpy()
    # probs_pos[probs_pos<thresh] = np.nan
    ax.plot(probs_pos, ls='-', lw=2,
            label=['pos_' + s for s in df_predictions_pos.columns])
    
    probs_shape = df_predictions_shape.to_numpy()
    # probs_shape[probs_shape<thresh] = np.nan
    ax.plot(probs_shape, ls='--', lw=2,
            label=['shape_' + s for s in df_predictions_shape.columns])
    
    ax.set_xlabel('Frame number', fontsize=16)
    ax.set_ylabel('Probability', fontsize=16)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # ax.set_ylim((0.7, 1.1))
    
    plt.subplots_adjust(right=0.8)
    
    return fig, ax


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized