#!/usr/bin/env python
# coding: utf-8

import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import os
import csv
import numpy as np


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
# mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
# mp_pose = mp.solutions.pose


show=False

file = 'all_the_coords_shapes.csv'
data_file = os.path.join('..', 'output', f'{file}') # need to save the file in the data folder

with open(data_file,mode='w', newline='') as f: 
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # csv_writer.writerow(landmarks)

col_names = ['class']
num_coords_face = 468
num_coords_hand = 21

# generate columns names
for val in range(0, num_coords_face):
    col_names += ['x_face{}'.format(val), 'y_face{}'.format(val), 'z_face{}'.format(val), 'v_face{}'.format(val)]

for val in range(0, num_coords_hand):
    col_names += ['x_r_hand{}'.format(val), 'y_r_hand{}'.format(val), 'z_r_hand{}'.format(val), 'v_r_hand{}'.format(val)]

with open(data_file, mode='a', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(col_names)
    

position = ['position_00','position_01','position_02',
          'position_03','position_04'] 

shape = ['shape_00','shape_01','shape_02','shape_03','shape_04','shape_05','shape_06','shape_07']
classes = shape


for label in classes:
    # Load Video
    fn = label
    class_name = label # name of the video
    cap = cv2.VideoCapture(os.path.join('..', 'data', 'training_videos', f'{fn}.mp4'))
    cap.set(3,640) #camera width
    cap.set(4,480) #camera hight
    n_frames = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))


    # Initiate holistic model
    i_frame = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            i_frame += 1
            print(f'{i_frame}/{n_frames}')
            if not ret:
                break
            # Recolor Feed

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)


            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            
            # 4. Pose Detections
            if show:
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

                # Append class name 
                row.insert(0, class_name)

                # Export to CSV
                with open(data_file, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)

            except:
                pass

            if show:
                cv2.imshow('cued_estimated', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    print(f'{label} was learned')
    cap.release()
    cv2.destroyAllWindows()
