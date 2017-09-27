from moviepy.editor import *
import os

if not os.path.exists('video_output'):
    os.mkdir('video_output')

#clip = VideoFileClip("dog_video.mp4") # or .avi, .webm, .gif ...
#for frame in clip.iter_frames():
#    write_clip = ImageSequenceClip(frame, fps=1)
#    write_clip.write_gif('image.gif')
    #frame.write_videofile("video_output/my_new_clip")
#clip.write_videofile("clip", codec='.png')

# Write a sequence of clips to a movie
#from moviepy.editor import *
#clip  = VideoFileClip("original_file.mp4")
#new_frames = [ some_effect(frame) for frame in clip.iter_frames()]
#new_clip = ImageSequenceClip(new_frames, fps=clip.fps)
#new_clip.write_videofile("new_file.mp4")

#Assemble pictures in a folder, write to a videofile and gif
#from moviepy.editor import ImageSequenceClip
#clip = ImageSequenceClip("some/folder/path", fps=10)
#clip.to_videofile("myvideo.mp4", fps=10) # many options available
#clip.to_gif("mygif.gif", fps=10) # many options available

# Convert the video to images and store to video output
import cv2
vc = cv2.VideoCapture("dog_video.mp4")
while True:
    c = 1

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()
        cv2.imwrite('video_output/' + str(c).zfill(7) + '.jpg', frame)
        c = c + 1
        cv2.waitKey(1)
    vc.release()