# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:49:56 2022

@author: hagar
"""

import pickle
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import os
import csv
import numpy as np

# Make detections with the trained model from the pickle file
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)
    

# Load Video
fn = 'word_h0_01'
cap = cv2.VideoCapture(f"videos/{fn}.mp4")
cap.set(3,640)
cap.set(4,480)

show=False

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
# mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
# mp_pose = mp.solutions.pose


# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        # Make Detections
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if show:
            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(0,0,0), thickness=1, circle_radius=1)
                                     )
            
            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                     )
    
            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                     )
    
            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                     )
                        

        # Export coordinates
        try:
            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
           
           # Extract right hand landmarks
            r_hand = results.right_hand_landmarks.landmark
            r_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in r_hand]).flatten())

           # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            #Create the row that will be written in the file
            row = face_row+r_hand_row
            
            # Name colls
            coll_names = []
            num_coords_face = len(results.face_landmarks.landmark)
            num_coords_hand = len(results.right_hand_landmarks.landmark)


            # generate collomns names
            for val in range(0, num_coords_face):
                coll_names += ['x_face{}'.format(val), 'y_face{}'.format(val), 'z_face{}'.format(val), 'v_face{}'.format(val)]

            for val in range(0, num_coords_hand):
                coll_names += ['x_r_hand{}'.format(val), 'y_r_hand{}'.format(val), 'z_r_hand{}'.format(val), 'v_r_hand{}'.format(val)]
            
            

            # Make Detections
            X = pd.DataFrame([row], columns = coll_names)
            predicted_position = model.predict(X)[0]
            position_prob = model.predict_proba(X)[0]
            
                        
            
            # Append prediction class and probability 
            row.insert(0, predicted_position)
            row.insert(1, position_prob)
            
            # Export to CSV
            file_name = os.path.join('..', '..', 'data', 'position_estimation_by_frames.csv')
            with open(file_name, mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            
        except:
            pass
                        
        cv2.imshow('cued_estimated', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
