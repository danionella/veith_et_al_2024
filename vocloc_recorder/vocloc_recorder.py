# General Purpose Imports

import json
import datetime
import sys
from collections import deque
import numpy as np
import multiprocessing
from queue import Empty, Queue, Full
import time
import os
import logging
import logging.handlers
import traceback

logger = logging.getLogger(__name__)
# Audio Imports
# import pyaudio
import soundfile as sf

# Video Imports
import cv2
import av

# Nidaq imports
import nidaqmx
import nidaqmx.system
from nidaqmx.stream_writers import AnalogSingleChannelWriter, AnalogMultiChannelWriter, DigitalSingleChannelWriter
from nidaqmx.stream_readers import AnalogUnscaledReader, DigitalMultiChannelReader
from nidaqmx.constants import RegenerationMode, Slope
from nidaqmx.constants import AcquisitionType, Edge, TerminalConfiguration, WAIT_INFINITELY
from nidaqmx.utils import flatten_channel_string

# Camera Imports
try:
    # from pyspin import PySpin
    from pypylon import pylon
except ModuleNotFoundError:
    logger.warning("Pylon Package is not found.")

# Array queues, https://github.com/portugueslab/arrayqueues/
from arrayqueues.shared_arrays import ArrayQueue, IndexedArrayQueue


def video_distributer(conf, qLog, qT, q, qSavT, qSav, qPrev, qImgCust, evtStopAcq, worker_configurer):
    """
    Thread function to receive images and timestamps from the streamer and distribute them to other methods.
    :param conf: Configuration Dictionary
    :param qLog:  Queue Object, for logging messages
    :param qT:  Queue Object, for receiving camera's timestamps
    :param q:  Queue Object, for receiving camera's images
    :param qSavT:  Queue Object, for sending camera's timestamps to saving thread
    :param qSav:  Queue Object, for sending camera's images to saving thread
    :param qPrev: Queue Object, for sending camera's images to preview thread
    :param qImgCust: Queue Object, for sending camera's images to a custom video function
    :param evtStopAcq: Event, which indicates whether to acquire images
    :param worker_configurer:
    :return:
    """
    # SetupLogger
    worker_configurer(qLog)
    logger = logging.getLogger()
    logger.log(logging.INFO, 'Started Video Distributer Thread')
    t_last = datetime.datetime.now() - datetime.timedelta(seconds=1)

    sub_t_for_preview = max(1, conf["trigger"]["rate"] // 4)  # preview framerate between 4fps and 8fps
    print(f'Preview framerate: {conf["trigger"]["rate"] / sub_t_for_preview} Hz')
    while not evtStopAcq.is_set():
        try:
            t, idx, im = q.get(timeout=0.5)  # index and timestamp from streaming queue
            tstmp = qT.get(timeout=0.5)  # timestamp from camera
            # distribute
            if conf['video']['save']:
                qSav.put(im, timestamp=t)  # propagate streaming timestamp
                qSavT.put(tstmp)
            if conf['preview']['active'] and idx % sub_t_for_preview == 0:
                qPrev.put(im[::conf['preview']['sub_spatial'], ::conf['preview']['sub_spatial']], timestamp=t)
            if conf['videoFunc']['active'] and idx % conf['videoFunc']['sub_t'] == 0:
                qImgCust.put(im[::conf['videoFunc']['sub_spatial'], ::conf['videoFunc']['sub_spatial']],
                             timestamp=t)  # propagate streaming timestamp
        except Empty:
            pass
        except Full as err:
            if (t - t_last).total_seconds() > 1:
                logger.log(logging.DEBUG, f'Lost frames at time {t}')
                traceback.print_tb(err.__traceback__)
            t_last = t
    del im, tstmp
    logger.log(logging.DEBUG, 'End of distributer thread')
    return


def video_saver(conf, qLog, qSavT, qSav, evtStopAcq, worker_configurer):
    """
    Thread function for saving a video and timestamps.
    :param conf: Configuration Dictionary
    :param qLog: Queue Object, for logging messages
    :param qSavT: Queue Object, for receiving camera's timestamps
    :param qSav: Queue Object, for receiving camera's images
    :param evtStopAcq: Event, which indicates whether to acquire images
    :param worker_configurer:
    :return:
    """
    # SetupLogger
    worker_configurer(qLog)
    logger = logging.getLogger()
    logger.log(logging.INFO, 'Started Video Saving Thread')

    # Filename
    filename = conf['pSavVideo']
    # check if target file exists; if so: delete it
    if os.path.exists(filename):
        logger.info('file exists: deleting ...')
        os.remove(filename)

    # Open Videostream to File
    container = av.open(filename, mode='w')
    fps = conf['video']['framerate']
    stream = container.add_stream('h264_nvenc', rate=fps)
    # stream.codec_context.options['qscale']='30'
    stream.options = {'qmax': str(conf['video']['qmax']), 'qmin': str(conf['video']['qmax'])}
    stream.width = conf['camera']['width']
    stream.height = conf['camera']['height']
    stream.pix_fmt = 'yuv420p'

    logger.debug("Opening File for timestamps")

    if conf['video']['save']:
        f_time = open(conf['pSavTimestamp'], 'wb')

    save_count = 0
    while not evtStopAcq.is_set():
        try:
            t, idx, im = qSav.get(timeout=0.5)
            tstmp = int(qSavT.get(timeout=0.5))
            f_time.write(tstmp.to_bytes(8, 'big'))
            save_count += 1
            frame_av = av.VideoFrame.from_ndarray(im, format='gray')
            for packet in stream.encode(frame_av):
                container.mux(packet)
            del im
        except Empty:
            pass

    # Flush encoding buffer
    if conf['video']['save']:
        logger.debug("closing timestamp file")
        f_time.close()
        logger.log(logging.DEBUG, 'Trying to close video file')
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        logger.log(logging.DEBUG, 'Closed video file')
        logger.log(logging.DEBUG, 'imcount from video file:%s', str(save_count))
    return


def video_streamer_pylon(conf, qLog, qT, q, evtStopAcq, evtCamStarted, worker_configurer, vCamFrame):
    '''
    Camera Control Thread for Pylon cameras, which streams images to the distributer thread
    :param conf: configuration dictionary
    :param qLog: Queue Object, for logging messages
    :param qT: Queue Object, for sending camera's timestamps to distributer
    :param q: Queue Object, for sending camera's images to distributer
    :param evtStopAcq: Event, which indicates whether to acquire images
    :param evtCamStarted: Event, which is set once the camera has started
    :param worker_configurer:
    :param vCamFrame: Value Object, for keeping track of the current camera frame index
    :return:
    '''

    # SetupLogger
    worker_configurer(qLog)
    logger = logging.getLogger()
    logger.log(logging.INFO, 'Started Video Streaming Thread')

    # Initialize Camera
    # Create an instant camera object with the camera device found first.
    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    cam.Open()
    cam.MaxNumBuffer = 200
    cam.ExposureAuto.SetValue('Off')
    cam.Width.Value = conf['camera']['width']
    cam.Height.Value = conf['camera']['height']
    cam.OffsetX.Value = conf['camera']['xoff']
    cam.OffsetY.Value = conf['camera']['yoff']
    # Ensure desired exposure time does not exceed the maximum
    exposure_time_to_set = conf['camera']['exposure'] * 1000  # in micros
    cam.ExposureTime.SetValue(exposure_time_to_set)
    # Set Gain to zero and turn of exposure auto
    cam.GainAuto.SetValue('Off')
    cam.Gain.SetValue(0)
    # Set Trigger Mode On
    if conf['camera']['triggered']:
        cam.TriggerMode.SetValue('On')
        # cam.AcquisitionFrameRateEnable.SetValue('Off')
    else:
        cam.TriggerMode.SetValue('Off')
        cam.AcquisitionFrameRate.SetValue(conf['video']['framerate'])
        cam.AcquisitionFrameRateEnable.SetValue('On')
    cam.TriggerDelay.SetValue(0)
    cam.TriggerSelector.SetValue('FrameStart')
    cam.TriggerSource.SetValue('Line3')
    cam.TriggerActivation.SetValue('RisingEdge')
    # Trigger On Line 3 FrameStart Rising Edge, Exposure Out Line 2.
    cam.LineSelector.SetValue('Line4')
    try:
        cam.LineInverter.SetValue('On')
    except:
        cam.LineInverter.SetValue(True)
    cam.LineMode.SetValue('Output')
    cam.LineSource.SetValue('ExposureActive')
    cam.PixelFormat.SetValue('Mono8')

    # Begin acquiring images
    cam.StartGrabbing()
    logger.log(logging.DEBUG, 'Acquiring images...')
    evtCamStarted.set()
    imCount = 0
    while not evtStopAcq.is_set():
        # Retrieve next received image and ensure image
        try:
            image_result = cam.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)
        except pylon.TimeoutException:
            logger.log(logging.DEBUG, 'Timeout in camera SDK')
            continue
        if image_result.GrabSucceeded():
            with vCamFrame.get_lock():
                vCamFrame.value = imCount  # holds the current camera frame index
            imCount += 1
            im = image_result.Array
            tstmp = image_result.GetTimeStamp()
            q.put(im)  # the datetime is automatically added to the IndexedArrayQueue
            qT.put(np.array(tstmp))  # the camera timestamp is forwarded explicitly
            del im, tstmp
            image_result.Release()
        else:
            logger.log(logging.DEBUG, "Error: ", image_result.ErrorCode, image_result.ErrorDescription)
            image_result.Release()
            pass

    cam.StopGrabbing()
    logger.log(logging.INFO, 'Trying To Stop Acquisition')
    logger.log(logging.DEBUG, 'Stopped Acquisition')

    # Deinitialize camera
    cam.Close()
    # Release system instance
    # system.ReleaseInstance()
    logger.log(logging.DEBUG, 'Cleared Camera Object')
    # evtCamStarted.clear()
    logger.log(logging.DEBUG, 'Ready to join')
    logger.log(logging.DEBUG, 'imcount from camera:%s', str(imCount))
    return


def video_streamer_spinnaker(conf, qLog, qT, q, evtStopAcq, evtCamStarted, worker_configurer, vCamFrame):
    '''
   DOES NOT WORK
   Camera Control Thread For Spinnaker Cameras
    :param conf: configuration dictionary
    :param q: queue, which sends images to videoSaver
    :param qLog: queue, for logging messages
    :param evtStopAcq: event, which indicates whether to acquire images
    :param evtCamStarted:  event. which is set once the camera has started
    :return:
    '''
    # SetupLogger
    worker_configurer(qLog)
    logger = logging.getLogger()
    logger.log(logging.INFO, 'Started Video Streaming Thread')

    # Initialize Camera
    print("Start")
    system = PySpin.System.GetInstance()
    camList = system.GetCameras()
    cam = camList.GetByIndex(0)
    cam.Init()
    cam.AcquisitionStop()  # reset camera
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    if cam.ExposureTime.GetAccessMode() != PySpin.RW:
        print('Unable to set exposure time. Aborting...')
    cam.DeviceLinkThroughputLimit.SetValue(500000000)
    cam.TLStream.StreamBufferCountMax()
    # Ensure desired exposure time does not exceed the maximum
    exposure_time_to_set = conf['camera']['exposure'] * 1000  # in micros
    exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set)
    cam.ExposureTime.SetValue(exposure_time_to_set)

    # Set Gain to zero and turn of exposure auto
    cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    cam.Gain.SetValue(0)
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)

    # Set image size and offset
    cam.Height.SetValue(conf['camera']['height'])
    cam.Width.SetValue(conf['camera']['width'])
    cam.OffsetX.SetValue(conf['camera']['xoff'])
    cam.OffsetY.SetValue(conf['camera']['yoff'])

    cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
    cam.LineMode.SetValue(PySpin.LineMode_Output)
    cam.LineSource.SetValue(PySpin.LineSource_ExposureActive)

    if conf['camera']['triggered']:
        # Set Trigger Mode On
        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        cam.TriggerSource.SetValue(PySpin.TriggerSource_Line3)
        cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
        cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
        cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
        # Trigger On Line 3 FrameStart Rising Edge, Exposure Out Line 2.
        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
    else:
        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        # Setup video saving
    width = conf['camera']['width']
    height = conf['camera']['height']

    fps = conf['video']['framerate']

    # Begin acquiring images
    cam.BeginAcquisition()

    logger.log(logging.DEBUG, 'Acquiring images...')
    evtCamStarted.set()
    t0 = time.time()
    imCount = 0
    while not evtStopAcq.is_set():
        try:
            # # Retrieve next received image and ensure image completion
            image_result = cam.GetNextImage(1000)

            if image_result.IsIncomplete():
                logger.log(logging.DEBUG, 'Image incomplete with image status %d...' % image_result.GetImageStatus())
            else:
                # Print image information
                width = image_result.GetWidth()
                height = image_result.GetHeight()
                # logger.log(logging.DEBUG,'Grabbed Image %d, width = %d, height = %d' % (imCount, width, height))
                # Convert image to Mono8
                image_converted = image_result.Convert(PySpin.PixelFormat_Mono8)
                tstmp = image_result.GetTimeStamp()
                im = image_converted.GetNDArray()

                with vCamFrame.get_lock():
                    vCamFrame.value = imCount  # holds the current camera frame index
                    imCount += 1
                q.put(im)  # the datetime is automatically added to the IndexedArrayQueue
                qT.put(np.array(tstmp))  # the camera timestamp is forwarded explicitly
                del im, tstmp
                # Release image
                image_result.Release()

        except PySpin.SpinnakerException as ex:
            image_result.Release()
            logger.log(logging.DEBUG, 'Error: %s' % ex)
            logger.log(logging.DEBUG, 'If this is the end of the recording, this might be expected')
            pass

    # End acquisition
    logger.log(logging.DEBUG, 'Trying To Stop Acquisition')
    cam.EndAcquisition()
    logger.log(logging.DEBUG, 'Stopped Acquisition')
    # Deinitialize camera
    cam.DeInit()
    logger.log(logging.DEBUG, 'Deinitialize camera')
    # del cam
    camList.Clear()
    # Release system instance
    # system.ReleaseInstance()
    logger.log(logging.DEBUG, 'Cleared Camera Object')
    # evtCamStarted.clear()
    logger.log(logging.DEBUG, 'Ready to join')
    logger.log(logging.DEBUG, 'imcount from camera:%s', str(imCount))
    return


def audio_parser(conf, qLog, qAudio, evtRunning, worker_configurer):
    '''
    Receives Audio, Saves, Visualizes, Possibly Redistributes...
    :param conf: configuration dicitionary
    :param qAudio: queue, to get chunks to saved
    :param qLog: queue, for logging messages
    :param evtRunning: event, which indicates whether data is still acquired
    :return:
    '''
    # SetupLogger
    worker_configurer(qLog)
    logger = logging.getLogger()
    logger.log(logging.INFO, 'Started Audio Saving Thread')

    nChan = len(conf['audio']['chStr'])
    samplerate = conf['audio']['rate']
    # How many samples to read per Chunk

    if conf['audio']['save']:
        if os.path.exists(conf['pSavAudio']):
            logger.log(logging.INFO, 'INFO: file exists: deleting ...')
            os.remove(conf['pSavAudio'])
            os.remove(conf['pSavTrigger'])
        audio_file = sf.SoundFile(conf['pSavAudio'], mode='w', samplerate=samplerate, channels=nChan, subtype='PCM_24',
                                  format='FLAC')

        trigger_file = open(conf['pSavTrigger'], 'w')
    evtRunning.wait()
    logger.log(logging.INFO, "INFO: Start Saving to disk")

    # If playback enabled start stream
    # Create a Bufferqueue
    if conf['audio']['playBack']:
        paud = pyaudio.PyAudio()
        qPlayback = Queue()
        logger.log(logging.INFO, 'Setup Audio Playback')

        def playback_callback(in_data, frame_count, time_info, status):
            arr = qPlayback.get()
            # TODO Now the last channels does not get recorded because its the camera trigger. Should be parameters
            if evtRunning.is_set():
                return (arr[:-1, :].mean(0).astype('Int16'), pyaudio.paContinue)
            else:
                return (arr[:-1, :].mean(0).astype('Int16'), pyaudio.paComplete)

        for ii in range(paud.get_device_count()):
            if conf['audio']['audioDeviceName'] == paud.get_device_info_by_index(ii)['name']:
                audioDeviceIndex = ii

        stream = paud.open(format=pyaudio.paInt16,
                           channels=1,
                           rate=conf['audio']['rate'],
                           output=True,
                           frames_per_buffer=conf['audio']['szChunk'],
                           output_device_index=audioDeviceIndex,
                           stream_callback=playback_callback
                           )

    # Saving and Buffering LOOP

    while evtRunning.is_set():
        if not (qAudio.qsize() == 0):
            chunk = qAudio.get(1)
            qAudio.task_done()
            if conf['audio']['save']:
                audio_file.write(chunk[:-1, :].T << 8)
                if conf['audio']['triggerDigital']:
                    chunk[-1, :].astype('int32').astype('bool').tofile(trigger_file)
                else:
                    # TODO threshold depends on min max value in streamer
                    (chunk[-1, :].astype('int32') > 630).astype('bool').tofile(trigger_file)
            # IF PLAYBACK BUFFER
            # TODO Start Stream befores loop
            if conf['audio']['playBack']:
                stream.start_stream()
                qPlayback.put(chunk)
    logger.log(logging.DEBUG, 'chunk was dtype %s', str(chunk.dtype))
    logger.log(logging.INFO, ' Emptying Buffer')
    while not (qAudio.qsize() == 0):
        chunk = qAudio.get(10)
        qAudio.task_done()
        if conf['audio']['save']:
            audio_file.write(chunk[:-1, :].T << 8)
        if conf['audio']['playBack']:
            qPlayback.put(chunk)
        del chunk
    if conf['audio']['save']:
        audio_file.close()
        trigger_file.close()
        logger.log(logging.INFO, 'File Closed')

    logger.log(logging.INFO, 'Audio Finished Writing To Disk')

    if conf['audio']['playBack']:
        try:
            stream.stop_stream()
        except OSError:
            logger.log(logging.WARNING, 'Stop Stream Timed Out')
        stream.close()
        paud.terminate()


def audio_streamer(conf, qLog, qAudio, evtStopAcq, evtRunning, worker_configurer):
    '''
    Streams form NI Daqdevice through python queue to other processes -> currently audio_parser
    :param conf: configuration dicitionary
    :param qLog: queue, for logging messages
    :param qAudio: queue, which receives chuncks from buffer
    :param evtStopAcq: event, which indicates that the acquisition should stop
    :param evtRunning: event, which indicates, that the task is running
    :return:
    '''
    # Setup Logging
    worker_configurer(qLog)
    logger = logging.getLogger()
    logger.log(logging.INFO, 'Started Audio Streaming Thread')

    nChan = len(conf['audio']['chStr'])
    nSamples = 10000  # Since continous does not matter
    # How many samples to read per Chunk
    szChunk = conf['audio']['szChunk']
    nChunks = int(np.ceil(nSamples / szChunk))
    # If chunks are fraction -> t will be longer
    tnew = nChunks * szChunk / conf['audio']['rate']

    # Shared sample CLock and trigger for DI an DO
    smpClockStr = "/" + conf['audio']['clockStr'] + '/ai/SampleClock'
    trigStr = "/" + conf['audio']['clockStr'] + '/ai/StartTrigger'

    # Task for Audio, Camera Trigger Record and Camera trigger send
    taskAI = nidaqmx.Task()
    if conf['audio']['triggerDigital']:
        taskDI = nidaqmx.Task()

    # TRIGGER OUT
    # Switch between analogue and digital trigger
    # Trigger Waveform
    nSmpTrig = int(conf['audio']['rate'] / conf['trigger']['rate'])
    tt = np.arange(0, nSmpTrig) * 1000.0 / conf['audio']['rate']  # time in ms
    trigger_wv = (tt < conf['trigger']['duration']) * conf['trigger']['HIGH'] + (
            tt >= conf['trigger']['duration']) * \
                 conf['trigger']['LOW']

    if 'ao' in conf['trigger']['chStr'][0]:
        taskOut = nidaqmx.Task()
        # Configure AO for camera trigger
        taskOut.ao_channels.add_ao_voltage_chan(conf['trigger']['chStr'][0], max_val=conf['trigger']['HIGH'],
                                                min_val=conf['trigger']['LOW'])
        taskOut.timing.cfg_samp_clk_timing(conf['audio']['rate'], source=smpClockStr,
                                           sample_mode=AcquisitionType.CONTINUOUS,
                                           samps_per_chan=10)
        taskOut.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
        writer = AnalogSingleChannelWriter(taskOut.out_stream)

        writer.write_many_sample(trigger_wv.astype('float64'))
        taskOut.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source=trigStr, trigger_edge=Slope.RISING)


    else:
        taskOut = nidaqmx.Task()
        taskOut.do_channels.add_do_chan(conf['trigger']['chStr'][0])
        taskOut.timing.cfg_samp_clk_timing(conf['audio']['rate'], source=smpClockStr,
                                           sample_mode=AcquisitionType.CONTINUOUS,
                                           samps_per_chan=nSamples)
        taskOut.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
        taskOut.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source=trigStr, trigger_edge=Slope.RISING)
        writer = DigitalSingleChannelWriter(taskOut.out_stream)
        # TODO this fails when we use a higher port than 0
        line_n = int(conf['trigger']['chStr'][0].split('line')[1])
        trigger_wv_bool = ((trigger_wv > 0) * 2 ** line_n).astype('uint32')
        writer.write_many_sample_port_uint32(trigger_wv_bool)

    minmax = conf['audio']['peakAmp']
    for iCh, chStr in enumerate(conf['audio']['chStr']):
        taskAI.ai_channels.add_ai_voltage_chan(chStr,
                                               max_val=minmax, min_val=-minmax,
                                               terminal_config=TerminalConfiguration.DEFAULT)
    if conf['audio']['triggerDigital']:
        taskDI.di_channels.add_di_chan(conf['audio']['chStrTrig'])
        taskDI.timing.cfg_samp_clk_timing(conf['audio']['rate'], source=smpClockStr,
                                          sample_mode=AcquisitionType.CONTINUOUS,
                                          samps_per_chan=nSamples)
        taskDI.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source=trigStr, trigger_edge=Slope.RISING)
        taskDI.in_stream.input_buf_size = 100 * conf['audio']['szChunk']
        readerTrig = DigitalMultiChannelReader(taskDI.in_stream)

        dataContainerTrigger = np.zeros([1, szChunk], dtype=np.uint32)
        dataContainer = np.zeros((nChan, szChunk), 'int32')
    else:
        taskAI.ai_channels.add_ai_voltage_chan(conf['audio']['chStrTrig']
                                               , max_val=5.2, min_val=-5.2,
                                               terminal_config=TerminalConfiguration.DEFAULT)
        dataContainer = np.zeros((nChan + 1, szChunk), 'int32')
    taskAI.timing.cfg_samp_clk_timing(conf['audio']['rate'], sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=10)
    readerUnscaled = AnalogUnscaledReader(taskAI.in_stream)
    # TODO CHOOSE BUFFER SIZE WISELY
    taskAI.in_stream.input_buf_size = 100 * conf['audio']['szChunk']

    # TODO THIS CALLBACK should write directly to qAudio
    buffer = deque()
    if conf['audio']['triggerDigital']:
        def callbackAudioToBuffer(taskHandle, eventType, nSamples, callbackData):
            readerUnscaled.read_int32(dataContainer, number_of_samples_per_channel=szChunk, timeout=1)
            readerTrig.read_many_sample_port_uint32(dataContainerTrigger, number_of_samples_per_channel=szChunk,
                                                    timeout=1)
            chk = np.vstack([dataContainer, dataContainerTrigger]).astype('int32')
            # samples=taskAI.read(number_of_samples_per_channel=nSmpTrig)
            # qAudio.put(dataContainer.copy())
            buffer.append(chk)
            return 0
    else:
        def callbackAudioToBuffer(taskHandle, eventType, nSamples, callbackData):
            readerUnscaled.read_int16(dataContainer, number_of_samples_per_channel=szChunk, timeout=1)
            buffer.append(dataContainer.astype('int32'))
            return 0

    taskAI.register_every_n_samples_acquired_into_buffer_event(szChunk, callbackAudioToBuffer)

    taskOut.start()
    if conf['audio']['triggerDigital']:
        taskDI.start()
    taskAI.start()
    evtRunning.set()

    # taskAI.wait_until_done(WAIT_INFINITELY)
    while not evtStopAcq.is_set():
        if not len(buffer) == 0:
            qAudio.put(buffer.popleft())

    logger.log(logging.INFO, " stopping and closing tasks")
    taskOut.stop()
    taskAI.stop()
    if conf['audio']['triggerDigital']:
        taskDI.stop()
        taskDI.close()
        del taskDI
    taskOut.close()
    taskAI.close()
    qAudio.close()

    del taskOut
    del taskAI

    evtRunning.clear()
    logger.log(logging.INFO, " audio recording stopped")
    return


def indep_function(conf, qLog, evtStop, worker_configurer):
    '''

    :param conf: configuration dicitionary
    :param qLog: queue, for logging messages
    :param evtStopAcq: event, which indicates that the acquisition should stop
    :return:
    '''
    # SetupLogger
    worker_configurer(qLog)
    logger = logging.getLogger()
    logger.log(logging.DEBUG, 'Start loops independent function.')

    from loops import indep_func
    clss = getattr(indep_func, conf['indepFunc']['className'])

    instance = clss(conf)  # init
    while not evtStop.is_set():
        instance.loop()  # loop
    instance.close()  # close
    return


def video_function(conf, qLog, qImgCust, evtStop, worker_configurer, evtTrigger, q_video2triggered):
    '''
    Thread function to encode video from camera with help of ffmpeg
    :param conf: Configuration Dictionary
    :param qLog: queue, for logging messages
    :param qImgCust: Queue Object which receives new camera images
    :param evtStop:
    :param worker_configurer:
    :param evtTrigger: multiprocessing event to start the triggered independent function
    :param q_video2triggered: queue, for sending info from video_func to triggered_func
    :return:
    '''
    # SetupLogger
    # worker_configurer(qLog)
    logger = logging.getLogger()
    logger.log(logging.DEBUG, 'Start loops video processing function.')

    from loops import video_func
    clss = getattr(video_func, conf['videoFunc']['className'])
    instance = clss(conf)  # init

    while not evtStop.is_set():
        instance.loop(qImgCust, evtTrigger, q_video2triggered)
    instance.close()
    return


def triggered_indep_function(conf, qLog, evtTrigger, evtStop, vCamFrame, q_video2triggered):
    '''
    Thread function to encode video from camera with help of ffmpeg
    :param conf: Configuration Dictionary
    :param qLog: queue, for logging messages
    :param evtTrigger: function waits for this trigger
    :param evtStop: close
    :param vCamFrame: value, current camera index
    :param q_video2triggered: queue, for sending info from video_func to triggered_func

    The process can be terminated without upon_trigger execution by:
    evtStop.set() # make clear that you want to stop
    evtTrigger.set() # escape the waiting
    :return:
    '''
    worker_configurer(qLog)
    logger = logging.getLogger()
    logger.log(logging.DEBUG, 'Start loops triggered independent function.')

    from loops import triggered_func
    clss = getattr(triggered_func, conf['triggeredFunc']['className'])
    instance = clss(conf)  # init
    while not evtStop.is_set():
        evtTrigger.wait()  # make sure you don't end up here forever by triggering if you want to close
        if not evtStop.is_set():  # don't execute when closing
            instance.upon_trigger(vCamFrame, q_video2triggered)
        evtTrigger.clear()  # wait for new event
    instance.close()


# The size of the rotated files is made small so you can see the results easily.
def listener_configurer(path_to_logfile):
    root = logging.getLogger()
    # h = logging.handlers.RotatingFileHandler('mptest.log', 'a', 300, 10)
    # logging.handlers.StreamHandler()
    h = logging.FileHandler(path_to_logfile)
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    root.addHandler(h)


def log_listen_proc(queue, configurer, path_to_logfile):
    configurer(path_to_logfile)
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def worker_configurer(queue):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.DEBUG)


class recorder:
    def __init__(self, confpath, p_toplevel):
        with open(confpath) as f:
            self.conf = json.load(f)
        self.p_toplevel = p_toplevel

    def setup(self, conf):
        '''SetUp Or Re-Create Multiple Python Process For Recording

        Returns
        -------

        '''
        if self.conf['camera']['interface'] == 'Pylon':
            self.video_streamer = video_streamer_pylon
        elif self.conf['camera']['interface'] == 'Spinnaker':
            self.video_streamer = video_streamer_spinnaker
        else:
            raise NameError('This camera interface is unkown')
        os.environ['IMAGEIO_FFMPEG_EXE'] = self.conf['general']['encoderPath']

        # Video Queues
        # Define buffer for images (one can use a small for testing, i.e. to detect runaway at high frame rates)
        byteperframe = self.conf['camera']['width'] * self.conf['camera']['height']  # 8 bit integer
        bufferframes_live = max(1, 0.1 * self.conf['trigger']['rate'])
        bufferframes = max(1, 12 * self.conf['trigger']['rate'])  # 12 seconds
        mbytes_live = bufferframes_live * byteperframe / 1e6
        mbytes = bufferframes * byteperframe / 1e6
        self.q_img = IndexedArrayQueue(mbytes_live)
        self.q_tstmp = ArrayQueue(1)  # 1MB data
        self.q_sav = IndexedArrayQueue(mbytes)
        self.q_sav_tstmp = ArrayQueue(1)
        self.q_prev = IndexedArrayQueue(mbytes)
        self.q_log = multiprocessing.Queue(-1)

        # Audio Queues
        self.q_audio = multiprocessing.JoinableQueue()

        # Custom function queues
        self.q_img_loop = IndexedArrayQueue(mbytes_live)
        self.q_video2triggered = ArrayQueue(1)  # send info from video function to triggered function

        # Value that indicates current camera frame
        self.v_cam_frame = multiprocessing.Value('i', 0)

        # Events for controlling processes
        self.evt_audio_running = multiprocessing.Event()
        self.evt_stop_acq_vid = multiprocessing.Event()
        self.evt_stop_acq_audio = multiprocessing.Event()
        self.evt_cam_started = multiprocessing.Event()
        self.evt_stop_save = multiprocessing.Event()
        self.evt_stop_distr = multiprocessing.Event()
        # Custom processes
        self.evt_stop_indep_process = multiprocessing.Event()
        self.evt_stop_video_process = multiprocessing.Event()
        self.evt_trigger = multiprocessing.Event()
        self.evt_stop_triggered_indep_process = multiprocessing.Event()

        # Processes
        self.listener = multiprocessing.Process(target=log_listen_proc,
                                                args=(self.q_log, listener_configurer, conf['general']['logPath']))

        self.audio_record = multiprocessing.Process(
            name='audioRecorder',
            target=audio_streamer,
            args=(
                conf, self.q_log, self.q_audio, self.evt_stop_acq_audio, self.evt_audio_running, worker_configurer),
        )
        self.audio_parse = multiprocessing.Process(
            name='audioParser',
            target=audio_parser,
            args=(conf, self.q_log, self.q_audio, self.evt_audio_running, worker_configurer),
        )

        self.indep_process = multiprocessing.Process(
            name='independentProcess',
            target=indep_function,
            args=(conf, self.q_log, self.evt_stop_indep_process, worker_configurer),
        )

        self.video_stream = multiprocessing.Process(
            name='vStreamer',
            target=self.video_streamer,
            args=(conf, self.q_log, self.q_tstmp, self.q_img, self.evt_stop_acq_vid, self.evt_cam_started,
                  worker_configurer, self.v_cam_frame),
        )
        self.video_distribute = multiprocessing.Process(
            name='vDistributer',
            target=video_distributer,
            args=(
                conf, self.q_log, self.q_tstmp, self.q_img, self.q_sav_tstmp, self.q_sav, self.q_prev, self.q_img_loop,
                self.evt_stop_distr, worker_configurer),
        )
        self.video_save = multiprocessing.Process(
            name='vSaver',
            target=video_saver,
            args=(conf, self.q_log, self.q_sav_tstmp, self.q_sav, self.evt_stop_save, worker_configurer),
        )
        self.video_process = multiprocessing.Process(
            name='videoProcess',
            target=video_function,
            args=(
                conf, self.q_log, self.q_img_loop, self.evt_stop_video_process, worker_configurer, self.evt_trigger,
                self.q_video2triggered),
        )
        self.triggered_indep_process = multiprocessing.Process(
            name='triggeredIndependentProcess',
            target=triggered_indep_function,
            args=(conf, self.q_log, self.evt_trigger, self.evt_stop_triggered_indep_process, self.v_cam_frame,
                  self.q_video2triggered)
        )

    def start_recording(self, name, allow_saving=True):
        ''' Start Recording

        Starts recording, but doesn't block, and no clean up

        Parameters
        ----------
        name: str
            recording name

        Returns
        -------

        '''

        conf = self.conf

        self.listener.start()
        worker_configurer(self.q_log)
        self.logger = logging.getLogger()

        if not allow_saving:
            conf['video']['save'] = False
            conf['audio']['save'] = False
            conf["indepFunc"]["allowSaving"] = False
            conf["triggeredFunc"]["allowSaving"] = False
            conf["videoFunc"]["allowSaving"] = False
        self.conf_used = conf

        now = datetime.datetime.now()
        name = now.strftime('%Y%m%d_%H%M_') + name
        self.name = name
        if conf['video']['save'] or conf['audio']['save']:
            p_sav = os.path.join(self.p_toplevel, name)
            self.p_sav = p_sav
            if not os.path.exists(p_sav):
                os.makedirs(p_sav)
            conf['pSavBaseName'] = os.path.join(p_sav, name + '_')
        # Load Configuration, dictionary passed to all workers
        if conf['video']['save']:
            conf['pSavVideo'] = os.path.join(p_sav, name + '_Video.mp4')
            conf['pSavTimestamp'] = os.path.join(p_sav, name + '_Timestamps.bin')
        if conf['audio']['save']:
            conf['pSavAudio'] = os.path.join(p_sav, name + '_Audio.flac')
            conf['pSavTrigger'] = os.path.join(p_sav, name + '_Trigger.bin')

        self.setup(conf)

        self.logger.info(f"Started at {now.strftime('%Y%m%d %H:%M')}")

        # Start custom functions
        if conf['triggeredFunc']['active']:
            self.triggered_indep_process.start()
        if conf['videoFunc']['active']:
            self.video_process.start()
        if conf['indepFunc']['active']:
            self.indep_process.start()

        time.sleep(2)

        # Start Audio Parser
        self.audio_record.start()
        self.audio_parse.start()
        if conf['audio']['save']:
            print("Audio will be saved.")  # handled within parser
        else:
            print("Audio won't be saved.")
        self.evt_audio_running.wait()
        time.sleep(2)

        # Start Videostream and writer , wait until camera started
        if conf['video']['save']:
            print("Video will be saved.")
            self.video_save.start()
        else:
            print("Video won't be saved.")
        self.video_distribute.start()
        self.video_stream.start()
        self.evt_cam_started.wait()

        # Logging
        self.logger.log(logging.DEBUG, "Video Stream Started As Process %s", str(self.video_stream.pid))
        self.logger.log(logging.DEBUG, "Video Distributer Started As Process %s", str(self.video_distribute.pid))
        if conf['video']['save']:
            self.logger.log(logging.DEBUG, "Video Save Started As Process %s", str(self.video_save.pid))
        self.logger.log(logging.DEBUG, "Audio Record Started As Process %s", str(self.audio_record.pid))
        self.logger.log(logging.DEBUG, "Audio Parse Started As Process %s", str(self.audio_parse.pid))

        return

    def stop_recording(self):
        '''

        Returns
        -------

        '''
        conf = self.conf_used
        self.evt_stop_acq_vid.set()
        time.sleep(5)
        self.logger.debug("Stopped video acquisition. Empty buffers...")
        # Wait for buffers
        self.evt_stop_distr.set()
        time.sleep(5)
        print('Stopped distributing images')
        if conf['video']['save']:
            self.evt_stop_save.set()
            time.sleep(5)
            self.logger.debug('Stopped saving images')
        self.video_stream.join()
        self.logger.debug("Closed Streamer")
        self.video_distribute.join()
        self.logger.debug("Closed Distributer")
        if conf['video']['save']:
            self.video_save.join()
            self.logger.debug('Closed Saver')
        time.sleep(2)
        self.evt_stop_acq_audio.set()
        self.audio_record.join()
        print('Audio Record Joined')
        self.audio_parse.join()

        # Save config file
        if conf['video']['save'] or conf['audio']['save']:
            with open(os.path.join(self.p_sav, self.name + '_conf.json'), 'w+') as f:
                json.dump(conf, f)

        # Logging
        # if conf['video']['save'] and conf['audio']['save']:
        #    cap = cv2.VideoCapture(conf['pSavVideo'])
        #   self.logger.log(logging.DEBUG, 'Recorded ' + str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) + ' frames')
        #  trig = np.fromfile(conf['pSavTrigger'], dtype='bool').astype('int')
        # n_trigs = np.sum(np.diff(trig) > 0)
        # self.logger.log(logging.DEBUG, 'Recorded ' + str(n_trigs) + ' triggers')

        if conf['triggeredFunc']['active']:
            self.evt_stop_triggered_indep_process.set()  # make clear that you want to stop
            time.sleep(.3)
            self.evt_trigger.set()  # escape the waiting
            time.sleep(.5)
            self.triggered_indep_process.join()
        if conf['videoFunc']['active']:
            self.evt_stop_video_process.set()
            time.sleep(.2)
            self.video_process.join()
        if conf['indepFunc']['active']:
            self.evt_stop_indep_process.set()
            time.sleep(.2)
            self.indep_process.join()
        self.q_img.clear()
        self.q_tstmp.clear()
        self.q_sav.clear()
        self.q_sav_tstmp.clear()
        self.q_prev.clear()
        self.q_img_loop.clear()
        self.q_video2triggered.clear()
        print('Cleared all queues')

        self.logger.log(logging.DEBUG, "Exit")
        # self.q_log.put_nowait(None)
        # self.listener.join()

    def record(self, t, name, allow_saving=True):
        ''' Start Recording



        Parameters
        ----------
        t: int
            Number of Seconds
        name: str
            Name of Recording

        Returns
        -------

        '''
        self.start_recording(name, allow_saving)
        # Wait
        print(f"Recording time: {t} seconds")
        time.sleep(t)
        self.stop_recording()

    def preview(self):
        self.start_recording(name, allow_saving=False)
        input('Stop Preview?')
        self.stop_recording()


def main(argv):
    confpath = argv[0]
    p_toplevel = argv[1] + '\\'
    t_rec = int(argv[2])

    # Name
    name = 'Test'
    r = recorder(confpath, p_toplevel)
    # r.preview()
    r.record(t_rec, name, allow_saving=False)


if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv[1:])
