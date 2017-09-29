from moviepy.editor import *
import os
from tqdm import tqdm           # Show a progress bar

clip = VideoFileClip("dog_video.mp4")
frames = int(clip.fps * clip.duration)
for f in tqdm(range(frames)):
    img_path = os.path.join('video_output/' + str(f).zfill(7) + '.jpg')
    clip.save_frame(img_path, f)


# Faster but opencv must be installed on the system
# Convert the video to images and store to video output
#import cv2
#vc = cv2.VideoCapture("dog_video.mp4")
#while True:
#    c = 1
#
#    if vc.isOpened():
#        rval, frame = vc.read()
#    else:
#        rval = False
#
#    while rval:
#        rval, frame = vc.read()
#        cv2.imwrite('video_output/' + str(c).zfill(7) + '.jpg', frame)
#        c = c + 1
#        cv2.waitKey(1)
#    vc.release()