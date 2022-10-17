#!/bin/bash
set -x

model_type='rf' # 'rf', 'lr'
gender='female' # male/female
cropping='cropped' # raw/cropped
path2test_videos='../stimuli/words/mp4'

# python3 extract_coordinates.py --gender $gender --cropping $cropping #--show-video # !-SLOW-! For TRAINING VIDEOs ONLY
echo python3 extract_features.py --gender $gender --cropping $cropping
echo python3 train.py --property-type position --gender $gender --cropping $cropping --model-type $model_type
echo python3 train.py --property-type shape --gender $gender --cropping $cropping --model-type $model_type

for fn_video in 'word_h0_01.mp4' 'word_h0_02.mp4'; do
	echo python3 predict.py --property-type position --gender $gender --cropping $cropping --path2test-videos $path2test_videos --fn-video $fn_video --model-type $model_type
	echo python3 predict.py --property-type shape --gender $gender --cropping $cropping --path2test-videos $path2test_videos --fn-video $fn_video --model-type $model_type
	echo python mark_video.py --gender $gender --cropping $cropping --path2video $path2test_videos --fn-video $fn_video --model-type $model_type --textgrid
done
