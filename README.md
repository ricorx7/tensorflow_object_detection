# tensorflow_object_detection
Video and Tensorflow Object Detection API

You can run the install script to install all the dependencies.
```bash
./install.sh
```

This will clone the github Tensorflow Object Detection API.  It will install all the dependencies.  It will create all the needed folders.

More details can be found here:
[http://www.sroboto.com/2017/09/pass-video-into-tensorflow-object.html](http://www.sroboto.com/2017/09/pass-video-into-tensorflow-object.html)


# Convert video to files
This will convert the video dog_video.mp4 to images in video_output/ folder.
```bash
python convert_video_to_images.py
```


# Process the images
This will process the images.  The processed images will be in video_output/output
```bash
cd models\research\object_detection
python main.py
```


# Convert Processed images back to Video
This will convert the images in video_output/output to a video output.mp4
```bash
python convet_images_to_video.py
```