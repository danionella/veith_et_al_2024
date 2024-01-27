# Loop functions
## How to use
Collection of classes that implement flexible user functions for e.g. live video analysis or triggering stimuli during video and audio acquisition.
A specific class is activated if its name is defined in the `config.json`.

## Independent functions
See collection of classes in `indep_func.py`  
A function that starts looping once the recording starts. It cannot set any triggers and does not receive a video stream.

## Video functions
See collection of classes in `video_func.py`  
A function that receives a live video stream (can be downsampled in space and time). The function can set a trigger to activate the triggered function.

## Triggered functions
See collection of classes in `triggered_func.py`  
A function that is triggered once the trigger is set by the video function.