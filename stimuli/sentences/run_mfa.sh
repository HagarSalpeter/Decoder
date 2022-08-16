# The conda environment must be activated before running

path2mfa='/home/yl254115/miniconda3/envs/aligner/bin/mfa'
path2input='/home/yl254115/projects/LPC_2022/Decoder/stimuli/sentences/mfa'
path2dict='/home/yl254115/Documents/MFA/pretrained_models/dictionary/french_mfa.dict'
language_model='french_mfa'
path2output='/home/yl254115/projects/LPC_2022/Decoder/stimuli/sentences/mfa'

echo 'The following commands should be launched:'
echo '------------------------------------------'
echo conda activate aligner
echo $path2mfa align --clean $path2input $path2dict $language_model $path2output
