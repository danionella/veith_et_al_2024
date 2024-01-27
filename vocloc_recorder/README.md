# Synced audio and video acquisition

## Install
### Hardware requirements
- `vocloc_recorder` was developed and tested on Ubuntu Linux (and sporadically on Windows 10). 
- we currently use FLIR cameras

### Installation procedure
- Install camera drivers and python adapters (e.g. [SpinnakerSDK](https://www.flir.eu/products/spinnaker-sdk) / [spinnaker-python](https://pypi.org/project/spinnaker-python/))
- clone or download this repository, cd the directory containing setupy.py, then:
```
conda env create -f requirements.yml
conda activate vocloc
pip install -e .
```
Install the frozen copy of https://github.com/portugueslab/arrayqueues into the environment.
```
cd arrayqueues
pip install -e .
```

## How to use
call 

    python vocloc_recorder.py config.json C:\\myrecordings\ 3600
    
   last argument is recording time in s.  

Or start the gui

    python run.py config.json


## Functionalities
The software allows synced acquisition of video and audio via NI-DAQ systems.  
Multiple processes stream audio and video to distributing processes for analysis and saving.  
One can add live analysis to the video stream and trigger a triggered function by writing a new class in one of the loops modules. These classes become a menu of functionalities which can be selected via the config file.

## Explanation of the config file 
    {
      "general": {
        "encoderPath": "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe", PATH TO ENCODER EXECUTABLE
        "logPath": ".\\log.txt", PATH TO FOLDER WHERE LOG FILE IS SAVED
        "savePath": "C:\\"
      },
      "audio": { PARAMETER REGARDING AUDIO AND SYNCRONISATION
        "triggerDigital": true, RECORD CAMERA TRIGGER AT DIGITAL PORT
        "save": true, SAVE AUDIO
        "rate": 51200, SAMPLINGRATE OF AUDIO FILE
        "clockStr": "cDAQ2", DEVICE FOR CLOCK OF ALL TASKS
        "chStr": [ AUDIO CHANNELS TO RECORD
          "cDAQ2Mod1/ai0",
          "cDAQ2Mod1/ai1",
          "cDAQ2Mod1/ai2",
          "cDAQ2Mod1/ai3"
        ],
        "chStrTrig": "cDAQ2Mod3/port0/line1", CHANNEL WHERE THE CAMERA TRIGGER ARRIVES
        "peakAmp": 4, EXPECTED PEAK AMPLITUDE OF AUDIOFILES (V) 
        "szChunk": 100000, SIZE OF A READOUT CHUNK (NSAMPLES EVENT OF DAQ)
        "playBack": false, SOUNDPREVIEW
        "audioDeviceName": "Microsoft Sound Mapper - Output" DEVICE FOR THE PREVIEW
      },
      "video": { PARAMETERS REGARDING THE VIDEO ENCODING
        "save": true, WHETHER TO SAVE
        "framerate": 24, FRAMERATE TO ENCODE, SHOULD MATCH TRIGGER RATE
        "qmin": 17, QUALITY PARAMETERS
        "qmax": 21,
        "bitrate": 250
      },
      "preview": { VIDEO PREVIEW
      "active": true,
      "sub_spatial": 1 TEMPORAL DOWNSAMPLING OF PREVIEW STREAM
      },
      "trigger": { PARAMETERS REGARDING THE CAMERA TRIGGER
        "chStr": [ WHERE TO SEND THE TRIGGER, ALSO SUPPORTS DIGITIAL OUT
          "cDAQ2Mod3/port0/line2"
        ],
        "HIGH": 4, HIGH VOLTAGE IN V (IF ANALOG CHANNEL)
        "LOW": 0, LOW VOLTAGE IN V (IF ANALOG CHANNEL)
        "rate": 24, RATE AT WHICH TO TRIGGER IN HZ
        "duration": 1 DURATION OF TRIGGER PULSES IN ms
      },
      "camera": { PARAMETERS REGARDING THE CAMERA 
        "interface": "Pylon", WHICH CAMERA TO USE: Pylon/Spinnaker
        "exposure": 4, CAMERA EXPOSURE IN ms
        "width": 1024, ROI SIZE AND OFFSET IN px
        "height": 1024,
        "xoff": 208,
        "yoff": 4,
        "triggered": true TRIGGERED OR FREE RUNNING
      },
      "vis": { CAMERA PREVIEW WINDOW SIZE
        "camWindowSize": [
          600,
          600
        ]
      },
      "audio_out": { OPTIONAL AUDIO OUTPUT CHANNELS
        "chStr": [
          "cDAQ2Mod2/ao0",
          "cDAQ2Mod2/ao1"
        ]
      },
      "indepFunc": { DEFINE A CUSTOM FUNCTION THAT RUNS IN LOOP, see indep_func.py
        "active": false,
        "className": "MyTest",
        "allowSaving": true
      },
      "videoFunc": { DEFINE A CUSTOM VIDEO FUNCTION THAT RUNS IN LOOP, see video_func.py
        "active": true,
        "className": "MyPreviewWithTrigger",
        "sub_spatial": 1,
        "sub_t": 1,
        "allowSaving": true
      },
      "triggeredFunc": { DEFINE A CUSTOM FUNCTION THAT RUNS WHEN TRIGGERED, see triggered_func.py
        "active": true,
        "className": "MyTest",
        "allowSaving": true
      },
      "custom_config": { PARAMETERS FOR CUSTOM FUNCTIONS
        "fn_stimset": "",
        "power_switch_on": 0,
        "serial_port": "/dev/ttyACM0",
        "power_switch_delay": 0.075
      }
    }

### veith_et_al_2024
Protocol for this publication.

Steps
- run `sound_targeting_field.py` to record the speaker's impulse responses, generate target waveforms and run sound conditioning, which computes the speaker signals to deliver the target sounds.
- enter the path to the `DATEstimset_field.h5` file into the config.
- The following config was used in these experiments:  


        {
          "general": {
            "encoderPath": "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
            "logPath": ".\\log.txt",
            "savePath": "C:\\"
          },
          "audio": {
            "triggerDigital": true,
            "save": true,
            "rate": 51200,
            "clockStr": "cDAQ1",
            "chStr": [
              "cDAQ1Mod1/ai0"
            ],
            "chStrTrig": "cDAQ1Mod3/port0/line1",
            "peakAmp": 4,
            "szChunk": 100000,
            "playBack": false,
            "audioDeviceName": "Microsoft Soundmapper - Output"
          },
          "video": {
            "save": true,
            "framerate": 120,
            "qmin": 22,
            "qmax": 22,
            "bitrate": 250
          },
          "preview": {
            "active": true,
            "sub_spatial": 1
          },
          "trigger": {
            "chStr": [
              "cDAQ1Mod3/port0/line2"
            ],
            "HIGH": 4,
            "LOW": 0,
            "rate": 120,
            "duration": 1
          },
          "camera": {
            "interface": "Pylon",
            "exposure": 1.6,
            "width": 336,
            "height": 336,
            "xoff": 964,
            "yoff": 888,
            "triggered": true
          },
          "vis": {
            "camWindowSize": [
              336,
              336
            ]
          },
          "audio_out": {
            "chStr": [
              "cDAQ1Mod2/ao0",
              "cDAQ1Mod2/ao1",
              "cDAQ1Mod2/ao2",
              "cDAQ1Mod2/ao3"
            ]
          },
          "indepFunc": {
            "active": false,
            "className": "MyTest",
            "allowSaving": true
          },
          "videoFunc": {
            "active": true,
            "className": "MyLiveBlobOrientation",
            "sub_spatial": 1,
            "sub_t": 8,
            "allowSaving": true
          },
          "triggeredFunc": {
            "active": true,
            "className": "MyDAQPlaybackAtPositions",
            "allowSaving": true
          },
          "custom_config": {
            "fn_stimset": "./2023-08-15_21-56-18_field/2023-08-15_21-56-18stimset_field.h5",
            "power_switch_on": 0,
            "serial_port": "/dev/ttyACM0",
            "power_switch_delay": 0.075
          }
        }

The sound conditioning relies on a custom build stepper motor system, that moves the hydrophone in the tank.
To write your own stepper, modify the stepper class in `utils>motor.py`.
