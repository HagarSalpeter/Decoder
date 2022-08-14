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
                       velocity, acceleration,
                       velocity_thresh=0.01,
                       acceleration_thresh=0.003,
                       p_thresh=0.5,
                       text_factor=1,
                       show=False):
    
    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions
    
    size = (int(cap.get(3)), int(cap.get(4)))
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
                print('!'*100)
                break
            
            # POSITION
            curr_row_pos = df_predictions_pos[df_predictions_pos['frame_number']==i_frame]
            # SHAPE
            curr_row_shape = df_predictions_shape[df_predictions_shape['frame_number']==i_frame]
            if curr_row_shape.empty or curr_row_pos.empty:
                i_frame += 1
                pbar.update(1)
                continue
            
            # GET PREDICTIONS
            i_df = curr_row_shape.index
            predicted_class_pos = curr_row_pos['predicted_class'].values[0]
            predicted_probs_pos = curr_row_pos[f'p_class_{predicted_class_pos + 1}'].values[0]
            
            predicted_class_shape = curr_row_shape['predicted_class'].values[0]
            predicted_probs_shape = curr_row_shape[f'p_class_{predicted_class_shape + 1}'].values[0]
            
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
            
       
            # Open Cartoon        
            fn_cartoon = f'pos_n{predicted_class_pos}_shape_n{predicted_class_shape}.png' 
            fn_cartoon = os.path.join('../data/cartoons/', fn_cartoon)
            cartoon = open_cartoon(fn_cartoon)
            height, width, channels = cartoon.shape
            
            # Write prediction on video:
            font = cv2.FONT_HERSHEY_SIMPLEX
            line_type = cv2.LINE_8 #LINE_AA            
            
            x1_box = int(50/text_factor)
            x2_box = int(100/text_factor)
            y1_box = int(80/text_factor)
            y2_box = y1_box+height
            
            x_cartoon = int(80/text_factor)
            y_cartoon = int(100/text_factor)

            x1_text = int(x_cartoon/text_factor)
            x2_text = int((x_cartoon+300)/text_factor)
            dy_text = int(50/text_factor)
            y_text = int(800/text_factor)
            
            offset = np.array((x_cartoon, y_cartoon)) #top-left point from which to insert the smallest image. height first, from the top of the window
            image[offset[0]:offset[0] + height,
            offset[1]:offset[1] + width] = cartoon

            
            cv2.putText(image, 'SHAPE:', (x1_text, y_text),
                        font, 1/text_factor, (255, 255, 255), 2, line_type)
            for i_shape in range(1, 9):
                y_text += dy_text
                cv2.putText(image, f'p_class_{i_shape}', (x1_text, y_text),
                            font, 1/text_factor, (255, 255, 255), 2, line_type)
                p_class = curr_row_shape[f'p_class_{i_shape}'].values[0]
                cv2.putText(image, str(p_class), (x2_text, y_text),
                            font, 1/text_factor, (255, 255, 255), 2, line_type)
            
            y_text += dy_text
            cv2.putText(image, 'POSITION:', (x1_text, y_text),
                        font, 1/text_factor, (255, 255, 255), 2, line_type)
            for i_pos in range(1, 6):
                y_text += dy_text
                cv2.putText(image, f'p_class_{i_pos}', (x1_text, y_text),
                            font, 1/text_factor, (255, 255, 255), 2, line_type)
                p_class = curr_row_pos[f'p_class_{i_pos}'].values[0]
                cv2.putText(image, str(p_class), (x2_text, y_text),
                            font, 1/text_factor, (255, 255, 255), 2, line_type)
            
            y_text += dy_text
            cv2.putText(image, 'Velocity', (x1_text, y_text),
                        font, 1/text_factor, (255, 255, 255), 2, line_type)
            cv2.putText(image, f'{velocity[i_df][0]:1.5f}', (x2_text, y_text),
                        font, 1/text_factor, (255, 255, 255), 2, line_type)
        
            
            # Mark prediction
            if predicted_probs_pos > p_thresh and \
                predicted_probs_shape > p_thresh and \
                    velocity[i_df][0]<velocity_thresh:# and \
                        
                        cv2.rectangle(image, (x1_box, y1_box), (x2_box, y2_box),
                                      (16, 255, 16), -1)
              
                    
    
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
