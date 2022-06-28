#!/usr/bin/env python
# coding: utf-8

# # 0. Install and Import Dependencies

# In[ ]:


#get_ipython().system('pip install mediapipe opencv-python pandas scikit-learn')


# In[4]:


import mediapipe as mp # Import mediapipe
import cv2 # Import opencv


# In[77]:


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
mp_pose = mp.solutions.pose


# # 1. Make Some Detections

# # Landmarks map
# <img src=https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png />

# In[30]:


relevant_landmarks = [0,16,18,20,22]


# In[72]:


get_ipython().run_line_magic('pinfo2', 'mp_drawing.draw_landmarks')


# In[83]:


cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)
        
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
                                 
        
        # 2. Right hand
        #mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
         #                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
          #                       mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
           #                      )

        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# # 2. Capture Landmarks & Export to CSV
# <!--<img src="https://i.imgur.com/8bForKY.png">-->
# <!--<img src="https://i.imgur.com/AzKNp7A.png">-->

# In[7]:


import csv
import os
import numpy as np


# In[84]:


num_coords = len(results.pose_landmarks.landmark)
num_coords


# In[85]:


landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]


# In[86]:


landmarks


# In[87]:


with open('coords_position2.csv',mode='w', newline='') as f: 
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)


# In[97]:


class_name = "04"


# In[98]:


# Load Video
fn = 'position_04'
cap = cv2.VideoCapture(f"videos/position/{fn}.mp4")
cap.set(3,640)
cap.set(4,480)


# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        
        #4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
                        

        # Export coordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose]).flatten())
            
            
            row = pose_row
            
            # Append class name 
            row.insert(0, class_name)
            
            # Export to CSV
            with open('coords_position2.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 
            
        except:
            pass
                        
        cv2.imshow('cued_estimated', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# # 3. Train Custom Model Using Scikit Learn

# ## 3.1 Read in Collected Data and Process

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('coords_position2.csv')


# In[3]:


df.head()


# In[125]:


df.tail()


# In[119]:


df[df['class']==1]


# In[16]:


X = df.drop('class', axis=1) # features
y = df['class'] # target value


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)


# In[24]:


y_test


# ## 3.2 Train Machine Learning Classification Model

# In[25]:


from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[26]:


pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    #'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}


# In[27]:


fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model


# In[28]:


fit_models


# In[133]:


fit_models['rc'].predict(X_test)


# ## 3.3 Evaluate and Serialize Model 

# In[29]:


from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 


# In[30]:


for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))


# In[13]:


fit_models['rc'].predict(X_test)


# In[14]:


y_test


# In[15]:


with open('body_language.pkl', 'wb') as f:
    pickle.dump(fit_models['rc'], f)


# # 4. Make Detections with Model

# In[175]:


with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)


# In[176]:


model


# In[31]:


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
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            
            # Concate rows
            row = pose_row+face_row
            
            # Name colls
            num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)
            col_name = []
            for val in range(1, num_coords+1):
                col_name += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

            # Make Detections
            X = pd.DataFrame([row], columns = col_name)
            predicted_position = model.predict(X)[0]
            position_prob = model.predict_proba(X)[0]
            
            
            
            
            # Append prediction class and probability 
            row.insert(0, predicted_position)
            row.insert(1, position_prob)
            
            # Export to CSV
            with open('position_estimation_by_frames.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            
        except:
            pass
                        
        cv2.imshow('cued_estimated', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# In[189]:




