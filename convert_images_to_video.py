# Assemble pictures in a folder, write to a videofile and gif
from moviepy.editor import ImageSequenceClip
clip = ImageSequenceClip("video_output/output", fps=24)
clip.to_videofile("video_output/output/output.mp4", fps=24) # many options available



#clip.to_gif("video_output/output/mygif.gif", fps=10) # many options available