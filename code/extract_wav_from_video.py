#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:28:31 2022

@author: yl254115
"""

import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--fn-video', default='sent_01.mp4')
parser.add_argument('--path2video', default=os.path.join('..', 'data',
                                                         'test_videos'))
args = parser.parse_args()

command = f"ffmpeg -i {os.path.join(args.path2video, args.fn_video)} -ab 160k -ac 2 -ar 44100 -vn {os.path.join(args.path2video, 'audio_only', args.fn_video+'.wav')}"

subprocess.call(command, shell=True)
