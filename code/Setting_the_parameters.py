# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:04:22 2022

@author: hagar
"""

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
