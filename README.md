# Automatic Cued-Speech Recognition (ACSR)#
Automatic detection of hand position and shape for cued-speech recognition. The decoding is based on MediaPipe, which is used to extract skeleton landmarks for hand and face. Then decoder are trained on featurescomputed from the landmarks. Results are marked on the input video and saved as a separate file. 

## Pipeline
To train an ACSR and test it on your video, follow the steps below:

[An example of ACSR on a test video](data/test_videos/test_marked.avi)

1. Extract skeleton coordinates (landmarks - see below) from traing videos in `data/training_videos` and saves a csv file to `output/`:

`extract_coordinates.py --show-video`

2. Computes various features based on the extracted skeleton coordinates (hand-nose distance, finger lengths, etc.). Results are saved as a csv file to `output/`:

`extract_features.py`

3. Train two separate random-field models for hand position and shape detection. Results are then saved to `trained_models/`:

`train.py --property-type position --model-type rf`
`train.py --property-type shape --model-type rf`

4. Generate predictions for hand position and shape for a new test video. Similarlty to the train videos, the test video first goes through coordinate extraction and feature computations, and then predictions are made based on the two trained models for hand position and shape (previous stage). Results are saved to `output/`:

`predict.py --property-type position --model-type rf --fn-video test.mp4`
`predict.py --property-type shape --model-type rf --fn-video test.mp4`

5. Mark the predictions for hand position and shape on the test video, together with the marking of the landmarks. The new video is saved at the same path (e.g., `data/test_videos/`) with an additional `_marked` (e.g., `test_marked.mp4`):

`mark_video.py --model-type rf --fn-video test.mp4`

### Landmark pose map: ###
![alt text](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)

### Landmark hand map: ###
![alt text](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)
