

path2video='../stimuli/sentences/mp4/'

for i in `seq -w 1 99`
	do
		fn_video='sent_'$i'.mp4'
		python mark_video.py --path2video $path2video --fn-video $fn_video --textgrid
done
