#!/bin/bash
set -x

model_type='rf' # 'rf', 'lr'

# python3 extract_coordinates.py --gender male --cropping cropped #--show-video # !-SLOW-! For TRAINING VIDEOs ONLY

python3 extract_features.py --gender male --cropping cropped
python3 train.py --property-type position --gender male --cropping cropped --model-type $model_type
python3 train.py --property-type shape --gender male --cropping cropped --model-type $model_type

for fn_video in 'sent_01.mp4' 'sent_02.mp4' 'sent_03.mp4' 'sent_04.mp4' 'sent_05.mp4'; do
	python3 predict.py --property-type position --fn-video $fn_video --model-type $model_type
	python3 predict.py --property-type shape --fn-video $fn_video --model-type $model_type
	python mark_video.py --fn-video $fn_video --model-type $model_type
done
