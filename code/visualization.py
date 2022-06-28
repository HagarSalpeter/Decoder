# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:08:44 2022

@author: hagar
"""
import argparse
import os

from utils import load_model, load_video 

parser = argparse.ArgumentParser()
parser.add_argument('--show-video', action='store_true', default=False)
parser.add_argument('--path2data', default=os.path.join('..', 'data',
                                                        'test_videos'))
parser.add_argument('--path2cav_pred', default=os.path.join('..', 'output')) #need to correct this
parser.add_argument('--path2output', default=os.path.join('..', 'output'))
args = parser.parse_args()

# Load Video

fn_video = os.path.join(args.path2data, fn_video+'.mp4')
print(f'Visualization for: {fn_video}')
cap = load_video(fn_video)
df_prediction = pd.read_csv(path2cav_pred)


marked_video = cv2.VideoWriter(f'{fn_video}_marked.avi',cv2.VideoWriter_fourcc(*'MJPG'),30, size)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        i_frame += 1
        if not ret:
            break
        
        prediction = df_prediction['frame_number'].loc[i_frame]['predicted_class']
        probability = df_prediction['frame_number'].loc[i_frame]['predicted_probability']
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
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    
   
        # Write prediction on video:
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get status box
        cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
        
        # Display Class
        cv2.putText(image, 'Predicted Position',
                     (95,12), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, predicted_position,
                     (90,40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
        
        # Display Probability
        cv2.putText(image, 'PROB'
                    , (15,12), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(round(predicted_position[np.argmax(position_prob)],2))
                    , (10,40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            
            
        except:
            pass


        cv2.imshow('cued_estimated', image)
        marked_video.write(image)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


marked_video.release()
cap.release()
cv2.destroyAllWindows()
print("The video was successfully saved")