#!/bin/bash
set -x

model_type='rf' # 'rf', 'lr'

for fn_video in 'sent_01.mp4' 'sent_02.mp4' 'sent_03.mp4' 'sent_04.mp4' 'sent_05.mp4'; do
	python3 predict.py --property-type position --fn-video $fn_video --model-type $model_type
	python3 predict.py --property-type shape --fn-video $fn_video --model-type $model_type
	python mark_video.py --fn-video $fn_video --model-type $model_type
done
