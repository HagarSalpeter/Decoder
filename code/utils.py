# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:37:05 2022

@author: hagar
"""
import pickle
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm 

def load_model(filename):
    with open(filename, 'rb') as f:
        model, feature_names = pickle.load(f)
    return model, feature_names


def load_video(path2file):
    cap = cv2.VideoCapture(path2file)
    cap.set(3,640) # camera width
    cap.set(4,480) # camera height
    return cap


def extract_class_from_fn(fn):
    '''
    get class number from filename, e.g.,
    '4' from 'position_04.mp4'
    '''
    st = fn.find('_') + 1
    ed = fn.find('.')
    return int(fn[st:ed])

def extract_coordinates(cap, fn_video, show_video=False):
    
    
    mp_drawing = mp.solutions.drawing_utils # Drawing helpers
    mp_holistic = mp.solutions.holistic # Mediapipe Solutions
    
    
    columns = ['fn_video', 'frame_number']
    num_coords_face = 468
    num_coords_hand = 21
    
    # generate columns names
    for val in range(0, num_coords_face):
        columns += ['x_face{}'.format(val), 'y_face{}'.format(val),
                      'z_face{}'.format(val), 'v_face{}'.format(val)]
    
    for val in range(0, num_coords_hand):
        columns += ['x_r_hand{}'.format(val), 'y_r_hand{}'.format(val),
                      'z_r_hand{}'.format(val), 'v_r_hand{}'.format(val)]
    
    df_coords = pd.DataFrame(columns=columns)

    n_frames = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=n_frames)

    # Initiate holistic model
    i_frame = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            i_frame += 1
            #print(f'{i_frame}/{n_frames}')
            if not ret:
                break
            # Recolor Feed

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)


            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            
            # 4. Pose Detections
            if show_video:
                # Draw face landmarks
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                         )
                
                # Right hand landmarks
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                         mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                         mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                         )
                # Pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                          )
                cv2.imshow('cued_estimated', image)

            
            # Export coordinates
            try:
                # Extract Face landmarks
                
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z,
                                           landmark.visibility] for landmark in face]).flatten())
               
               # Extract right hand landmarks
                r_hand = results.right_hand_landmarks.landmark
                r_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z,
                                             landmark.visibility] for landmark in r_hand]).flatten())

              
                
                #Create the row that will be written in the file
                row = [fn_video, i_frame] + face_row +r_hand_row
                df_coords = df_coords.append(dict(zip(columns, row)),
                                             ignore_index=True)

            except:
                pass


            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            pbar.update(1)
            
    cap.release()
    cv2.destroyAllWindows()
    
    return df_coords
 
    
    
def extract_features(df_coords):
    #create the df of relevant feature
    df_features = df_coords[['x_face0','y_face0','z_face0',
                      'x_face234','y_face234','z_face234',
                      'x_face454','y_face454','z_face454',
                      'x_r_hand0','y_r_hand0','z_r_hand0',
                      'x_r_hand3','y_r_hand3','z_r_hand3',
                      'x_r_hand4','y_r_hand4','z_r_hand4',
                      'x_r_hand5','y_r_hand5','z_r_hand5',
                      'x_r_hand6','y_r_hand6','z_r_hand6',
                      'x_r_hand8','y_r_hand8','z_r_hand8',
                      'x_r_hand9','y_r_hand9','z_r_hand9',
                      'x_r_hand12','y_r_hand12','z_r_hand12',
                      'x_r_hand13','y_r_hand13','z_r_hand13',
                      'x_r_hand16','y_r_hand16','z_r_hand16',
                      'x_r_hand17','y_r_hand17','z_r_hand17',
                      'x_r_hand20','y_r_hand20','z_r_hand20'
                      ]] #relevant cols
    
    def axis_distance(df_name,col1,col2):
        return df_name[col1] - df_name[col2]
    
    def coords_distance(df_name,d_x,d_y,d_z):
        return np.sqrt((df_name[d_x])**2 + (df_features[d_y])**2 + (df_features[d_z])**2)
    
    df_features['fn_video'] = df_coords['fn_video']
    df_features['frame_number'] = df_coords['frame_number']
    
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
    df_features['d_x_face0_r_hand0'] = normalized_axis_distance(df_features,'x_r_hand0','x_face0')
    df_features['d_y_face0_r_hand0'] = normalized_axis_distance(df_features,'y_r_hand0','x_face0')
    df_features['d_z_face0_r_hand0'] = normalized_axis_distance(df_features,'z_r_hand0','z_face0')
    df_features['distance_face0_r_hand0'] = normalized_coords_distance(df_features,'d_x_face0_r_hand0','d_y_face0_r_hand0','d_z_face0_r_hand0')
    df_features['tan_alpha_pose'] = df_features['d_y_face0_r_hand0']/df_features['d_x_face0_r_hand0'] # tan of alpha - the angle between the face center, hand and the horizontal axis
    
    #features for shape
    pairs = [('8','5'),('12','9'),('16','13'),('17','20'),('4','6'),('3','5'),('8','12')]
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
                      ['d_x_r_hand8_x_r_hand12', 'd_y_r_hand8_y_r_hand12', 'd_z_r_hand8_z_r_hand12','d_r_hand8_r_hand12']]
    
    #get distance features
    for j in delta_triplets:
        df_features[j[3]] = normalized_coords_distance(df_features,j[0],j[1],j[2])
    

    return df_features


def get_feature_names(property_name):
    if property_name == 'position':
        feature_names = ['d_x_face0_r_hand0','d_y_face0_r_hand0','d_z_face0_r_hand0',
                         'distance_face0_r_hand0','tan_alpha_pose']
    elif property_name == 'shape':
        feature_names = ['d_x_r_hand8_x_r_hand5', 'd_y_r_hand8_y_r_hand5', 'd_z_r_hand8_z_r_hand5','d_r_hand8_r_hand5',
                         'd_x_r_hand12_x_r_hand9', 'd_y_r_hand12_y_r_hand9', 'd_z_r_hand12_z_r_hand9','d_r_hand12_r_hand9',
                         'd_x_r_hand16_x_r_hand13', 'd_y_r_hand16_y_r_hand13', 'd_z_r_hand16_z_r_hand13','d_r_hand16_r_hand13',
                         'd_x_r_hand17_x_r_hand20', 'd_y_r_hand17_y_r_hand20', 'd_z_r_hand17_z_r_hand20','d_r_hand17_r_hand20',
                         'd_x_r_hand4_x_r_hand6', 'd_y_r_hand4_y_r_hand6', 'd_z_r_hand4_z_r_hand6','d_r_hand4_r_hand6',
                         'd_x_r_hand3_x_r_hand5', 'd_y_r_hand3_y_r_hand5', 'd_z_r_hand3_z_r_hand5','d_r_hand3_r_hand5',
                         'd_x_r_hand8_x_r_hand12', 'd_y_r_hand8_y_r_hand12', 'd_z_r_hand8_z_r_hand12','d_r_hand8_r_hand12']
    return feature_names



    
def compute_predictions(model, df_features):
    '''
    model - sklean model 
    df_features - dataframe with n_samples X n_features
    '''
    X = df_features.to_numpy()
    predicted_class = model.predict(X)
    predicted_probs = model.predict_proba(X)

    return predicted_probs, np.asarray(predicted_class)
 
 

def mark_pred_on_video(cap, fn_video,
                       df_predictions_pos, df_predictions_shape,
                       p_thresh=0.8,
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
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # Display Class
            cv2.putText(image, 'Position',
                         (95,12), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if predicted_probs_pos > p_thresh:
                cv2.putText(image, str(predicted_class_pos),
                         (90,40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image, 'Shape',
                         (15,12), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if predicted_probs_shape > p_thresh:
                cv2.putText(image, str(predicted_class_shape),
                         (15,40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
            
            # Display Probability
            # cv2.putText(image, 'PROB'
            #             , (15,12), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(image, str(round(predicted_position[np.argmax(position_prob)],2))
            #             , (10,40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                
                
            # except:
            #     pass
    
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
