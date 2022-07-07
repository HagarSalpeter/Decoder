python3 extract_coordinates.py #--show-video
python3 extract_features.py
python3 train.py --property-type position
python3 train.py --property-type shape
python3 predict.py --property-type position
python3 predict.py --property-type shape
python mark_video.py # --model-type rf --fn-video test.mp4 --show-video
