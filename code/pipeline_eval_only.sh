#!/bin/bash
set -x

model_type='rf' # 'rf', 'lr'

for fn_video in '001-1.mp4' '001-2.mp4' '002-1.mp4' '002-2.mp4'; do
	python3 predict.py --property-type position --fn-video $fn_video --model-type $model_type
	python3 predict.py --property-type shape --fn-video $fn_video --model-type $model_type
	python mark_video.py --fn-video $fn_video --model-type $model_type --text-factor 3
done
