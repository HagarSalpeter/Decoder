
set -x

for fn_video in 'sent_01.mp4' 'sent_02.mp4' 'sent_03.mp4' 'sent_04.mp4' 'sent_05.mp4'; do
        python extract_wav_from_video.py --fn-video $fn_video
done
 
