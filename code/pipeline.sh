python3 extract_coordinates.py --gender male --cropping cropped #--show-video
python3 extract_features.py --gender male --cropping cropped
python3 train.py --property-type position --gender male --cropping cropped
python3 train.py --property-type shape --gender male --cropping cropped
python3 predict.py --property-type position --gender male --cropping cropped
python3 predict.py --property-type shape --gender male --cropping cropped
python mark_video.py # --model-type rf --fn-video test.mp4 --show-video
