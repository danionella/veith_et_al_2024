import os
import glob
import shutil
import numpy as np
import time
import scipy
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import datetime
import soundfile as sf

from misc.sound_targeting.utils import io, motor, daq

def main():
    SFR = SoundFieldRecording()

    if 1:
        SFR.make_stimset()
    if 1:
        SFR.test_stimset(fn=None)
        # SFR.test_stimset(fn="2024-01-19_14-11-08stimset_field_stimonly.h5")
    if 0:
        SFR.measure_pfield_constancy(
            fn="2024-01-23_17-46-17stimset_field.h5",
            outdir="/home/recordings")
    if 0:
        SFR.measure_accelfield_constancy(
            fn="2024-01-12_19-30-27_field/2024-01-12_19-30-27stimset_field.h5",
            outdir="/home/recordings")

    # SFR.test_kernel_additivity()


class SoundFieldRecording:
    def __init__(self, conf=None, mkpaths=True):
        # DAQ CONFIG
        self.conf = conf
        if conf is None:
            print("Use default config of SoundFieldRecording class...")
            self.conf = {
                "audio": {
                    "rate": 51200.0,
                    "clockStr": "cDAQ1",
                    "chStr": ["cDAQ1Mod1/ai0"],
                    "chStrTrig": "cDAQ1Mod3/port0/line1",
                    "peakAmp": 4,  # 1.6
                    "szChunk": 100000.0,
                    "audioDeviceName": "Microsoft Soundmapper - Output"
                },
                "accel": {
                    "chStr": ["cDAQ1Mod1/ai3", "cDAQ1Mod1/ai4", "cDAQ1Mod1/ai5"],
                    "sensitivity_lookup": {"cDAQ1Mod1/ai3": 98.2, "cDAQ1Mod1/ai4": 99.0, "cDAQ1Mod1/ai5": 103.3},
                },
                "audio_out": {
                    "chStr": ["cDAQ1Mod2/ao0", "cDAQ1Mod2/ao1", "cDAQ1Mod2/ao2", "cDAQ1Mod2/ao3"]
                }
            }
        self.sr = self.conf["audio"]["rate"]

        # CONVERSION PARAMS
        self.params = dict()
        # conversion of measurements to pressure
        # turn measurement into volts, see manual NI 9231 for unscaled recordings, 24 bit
        meas_2_vDAQ = 1  # 2 ** 23 * (610715 * 10 ** -12)  # ~5.123
        # volts at AS-1 hydrophone, 50dB gain of inverted single line PA4 amplifier
        vDAQ_2_vAS1 = 1 / 10 ** (50 / 20)
        # pressure from AS-1 sensitivity 40uV / Pascal
        vAS1_2_p = 1 / (40e-6)
        # all at once
        self.m2p = meas_2_vDAQ * vDAQ_2_vAS1 * vAS1_2_p
        self.m2a = 1  # acceleration channels return m/s^2
        self.params["m2p"] = self.m2p
        self.params["m2a"] = self.m2a
        # medium
        self.params["rho"] = 1000.0
        self.params["c"] = 1500.0
        self.port = '/dev/ttyACM0'  # port for arduino that drives stepper motors

        # OUTPUT PATHS
        if mkpaths:
            now = datetime.datetime.now()
            self.nowstr = now.strftime('%Y-%m-%d_%H-%M-%S')
            self.pth = r'{}_field'.format(self.nowstr)
            if os.path.exists(self.pth):
                shutil.rmtree(self.pth)
            os.makedirs(self.pth)

            self.todaystr = now.strftime('%Y-%m-%d')
            self.pth_today = r'{}_field'.format(self.todaystr)
            if not os.path.exists(self.pth_today):
                os.makedirs(self.pth_today)
            self.plotpth = self.pth

    def measure_pfield_constancy(self, fn, outdir):
        ss = io.load_from_h5(fn)
        sounds_cmap0 = ss['sounds_cdmap0'].copy()

        ss_const = {}
        ss_const['stimset_filename'] = fn
        ss_const['constancy_param'] = ss['test0_params'].copy()
        ss_const['constancy_param'].pop('test_static')
        ss_const['constancy'] = {}

        del ss

        ss_const['constancy'][self.nowstr] = self.run_tests_only_p(sounds_cmap0, **ss_const['constancy_param'])

        # Save
        ss_const['globalparams'] = self.params
        ss_const['globalconfig'] = self.conf
        data_fn = os.path.join(outdir, self.nowstr + 'measure_pfield_constancy.h5')
        io.save_to_h5(data_fn, ss_const, compression='lzf')
        print(f"\nSaved data to\n{data_fn}")

    def measure_accelfield_constancy(self, fn, outdir):
        ss = io.load_from_h5(fn)
        sounds_cmap0 = ss['sounds_cdmap0'].copy()

        ss_const = {}
        ss_const['stimset_filename'] = fn
        ss_const['constancy_param'] = ss['test0_params'].copy()
        ss_const['constancy_param'].pop('test_static')
        ss_const['constancy_accel'] = {}

        del ss

        ss_const['constancy_accel'][self.nowstr] = self.run_tests_only_accel(sounds_cmap0,
                                                                             **ss_const['constancy_param'])

        # Save
        ss_const['globalparams'] = self.params
        ss_const['globalconfig'] = self.conf
        data_fn = os.path.join(outdir, self.nowstr + 'measure_accelfield_constancy.h5')
        io.save_to_h5(data_fn, ss_const, compression='lzf')
        print(f"\nSaved data to\n{data_fn}")

    def make_stimset(self):
        ss = {}
        # measure impulse responses of all speakers at all positions
        # dim: nsamples, ny, nx, nspeaker, repeats

        # position of origin relative to inner tank
        self.params["origin_cm_yx"] = [2, 2]

        fn = os.path.join(self.pth_today, f"{self.todaystr}_irfield.h5")
        if os.path.exists(fn):
            ss = io.load_from_h5(fn)
            print(f"Loaded today's impulse response field:\n{fn}\n  array shape: {ss['irfield'].shape}")
        else:
            ss['irfield_params'] = {'startx': 0,  # 0
                                    'starty': 0,  # 0
                                    'stopx': 6,  # 6
                                    'stopy': 6,  # 6
                                    'nx': 5,  # 5
                                    'ny': 5,  # 5
                                    'repeats': 40,  # 40
                                    'dp_param': {"duration": 0.03, "offset": 0.005,
                                                 "uptime": 0}}  # 0.1e-3 #todo remove uptime option if not used
            print(ss['irfield_params'])
            ss['irfield'] = self.get_irfield(**ss['irfield_params'])
            io.save_to_h5(fn, ss, compression='lzf')
            print(f"Measured impulse response field\n  array shape: {ss['irfield'].shape}")

        # turn into kernels for pressure and acceleration
        # key: observable, value: nsamples, ny, nx, nspeaker
        ss['kernelfield_params'] = ss['irfield_params'].copy()
        ss['kernelfield_params']['kernel_highpassf'] = 100  # 100
        ss['kernelfield_params']['kernel_lowpassf'] = 0
        ss['kernelfield_params']['kernel_off_smooth_fraction'] = 0.1
        ss['kernelfield'] = self.get_kernelfield(ss['irfield'], **ss['kernelfield_params'])
        # ss['kernelfield_ft'] = self.get_kernelfield_ft(ss['irfield'], **ss['kernelfield_params'])
        print(f"Computed kernel field for: \n  {ss['kernelfield'].keys()}")
        print(f"Kernel duration is pulse duration minus pulse offset.")

        # read all wavs from one folder, list of 1d wav data
        ss['wavs_params'] = {'fd': r"./wav_folder"}
        ss['wavs'] = self.get_wavs(**ss['wavs_params'])
        print(f"Found these sounds: \n  {ss['wavs'].keys()}")

        # define target sound modes
        # key: stimulus name, value: dict("target", "sp_map"), where "target" key: observable, value: target waveform
        sd_trick = True

        if sd_trick:
            print("Using trick paradigm...")
            print("2 different stimuli (inverted/noninverted x left/right x trick/notrick + 4 pressure/motion only)")
            # Trick condition
            ss['sounds_tg_params'] = {'p_amplitude': 223.87211385683378, 'pad_silence': 0.02, 'distance2monopole': 0.03,
                                      # 251.189 281.838 316.227 354.813 398.107 446.684 (before correction -> divide by 5.123)
                                      'smp_horizon': int((16 / 400) * self.sr)}  # 16 frames at 400fps
            ss['sounds_tg'] = self.get_targets_trick(ss['wavs'], **ss['sounds_tg_params'])

        print(f"Made target wavorms for: \n  {ss['sounds_tg'].keys()}")
        print(f"Targets are delayed by the kernel duration.")

        # make map for conditioned sounds
        # key: stimulus name, value: nspeakers,nsamples,ny,nx
        # climb duration: must be smaller than kernel duration, fall duration smaller than pad_silence
        # for 'debugging' it makes sense to switch off filtering of the returned sounds.

        if sd_trick:
            peakAmp = self.conf["audio"]["peakAmp"]
            ss['sounds_cdmap0_params'] = {'lowcut': 200, 'highcut': 1200, 'env_time': 0.03, 'peakAmp': peakAmp}  # 0.02
            ss['sounds_cdmap0'] = self.get_conditioned_bc_trick(ss['kernelfield'], ss['sounds_tg'],
                                                                **ss['sounds_cdmap0_params'])

        print(f"Made conditioned soundmap for:\n  {ss['sounds_cdmap0'].keys()}")

        # Save
        ss['globalparams'] = self.params
        ss['globalconfig'] = self.conf
        data_fn = os.path.join(self.pth, self.nowstr + 'stimset_field_stimonly.h5')
        io.save_to_h5(data_fn, ss, compression='lzf')
        print(f"\nSaved data to\n{data_fn}")
        return

    def test_stimset(self, fn=None):
        if fn is None:
            fn = os.path.join(self.pth, self.nowstr + 'stimset_field_stimonly.h5')
        if os.path.exists(fn):
            ss = io.load_from_h5(fn)
            print(f"Loaded today's stimulus field.")
        else:
            print(f"File {fn} does not exist.")

        # Test conditioned sounds. Tests also naives non-conditioned playbacks if test_static=True
        print("\nTest generated stimuli...\n")
        ss['test0_params'] = ss['irfield_params'].copy()  # holds the grid locations
        ss['test0_params']['repeats'] = 8  #
        ss['test0_params'].pop('dp_param')
        ss['test0_params']['test_static'] = 1
        ss['test0'] = self.run_tests(ss['sounds_cdmap0'], ss['wavs'], **ss['test0_params'])

        # Save
        data_fn = os.path.join(os.path.split(fn)[0], self.nowstr + 'stimset_field.h5')
        io.save_to_h5(data_fn, ss, compression='lzf')
        print(f"\nSaved data to\n{data_fn}")

        if 0:
            # Correct for differences
            print("\nCalculate new coditioned sounds from mismatches.")
            ss['sounds_cdmap1_params'] = {**ss['sounds_cdmap0_params'], **ss['irfield_params']}
            ss['sounds_cdmap1_params'].pop('repeats')
            ss['sounds_cdmap1_params'].pop('dp_param')
            ss['sounds_cdmap1_params']['env_time'] = 0.015
            ss['sounds_cdmap1'] = self.get_corrected(ss['sounds_cdmap0'], ss['test0'], ss['kernelfield'],
                                                     ss['sounds_tg'],
                                                     **ss['sounds_cdmap1_params'])

            # Run test again
            print("\nTest improved stimuli...")
            ss['test1_params'] = ss['test0_params'].copy()
            ss['test1_params']['test_static'] = 1
            ss['test1'] = self.run_tests(ss['sounds_cdmap1'], ss['wavs'], **ss['test1_params'])

            # Save
            ss['globalparams'] = self.params
            ss['globalconfig'] = self.conf
            data_fn = os.path.join(self.pth, self.nowstr + 'stimset_field.h5')
            io.save_to_h5(data_fn, ss, compression='lzf')
            print(f"\nSaved data to\n{data_fn}")
        return

    def test_kernel_additivity(self):
        ss = {}
        sps = [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, -1, 0, 0]]  # left right both
        # measure impulse responses of all speakers at all positions
        # dim: nsamples, ny, nx, nspeaker, repeats
        ss['irfield_params'] = {'startx': 0,  # 0
                                'starty': 0,  # 0
                                'stopx': 8,  # 8
                                'stopy': 8,  # 8
                                'nx': 4,  # 7
                                'ny': 4,  # 7
                                'repeats': 20,  # 20
                                'sps': sps,
                                'dp_param': {"duration": 0.05, "offset": 0.005,
                                             "uptime": 0}}  # 0.1e-3 #todo remove uptime option if not used
        print(ss['irfield_params'])
        ss['irfield'] = self.get_irfield(**ss['irfield_params'])
        print(f"Measured impulse response field\n  array shape: {ss['irfield'].shape}")

        # turn into kernels for pressure and acceleration
        # key: observable, value: nsamples, ny, nx, nspeaker
        ss['kernelfield_params'] = ss['irfield_params'].copy()
        ss['kernelfield_params']['kernel_highpassf'] = 60
        ss['kernelfield_params']['kernel_lowpassf'] = 0
        ss['kernelfield_params']['kernel_off_smooth_fraction'] = 0.1
        ss['kernelfield_params'].pop('sps')
        ss['kernelfield'] = self.get_kernelfield(ss['irfield'], **ss['kernelfield_params'])

        # Save
        ss['globalparams'] = self.params
        ss['globalconfig'] = self.conf
        data_fn = os.path.join(self.pth, self.nowstr + 'kernel_additivity.h5')
        io.save_to_h5(data_fn, ss, compression='lzf')
        print(f"\nSaved data to\n{data_fn}")

    def get_stimulus_field_set(self, fd_stimset):
        name = os.path.split(fd_stimset)[1]
        fn = glob.glob(os.path.join(fd_stimset, "*stimset_field.h5"))[0]
        print(f"Load: {fn}")
        stimset = io.load_from_h5(fn)
        return stimset, name

    def get_stimulus_field_set_fn(self, fn_stimset):
        name = os.path.split(fn_stimset)[1]
        print(f"Load: {fn_stimset}")
        stimset = io.load_from_h5(fn_stimset)
        return stimset, name

    def get_irfield(self, startx, stopx, nx, starty, stopy, ny, repeats, dp_param, sps=None):
        # kronecker delta pulse
        dp = self.deltapulse(samplerate=self.sr, **dp_param)
        dp = dp / np.max(abs(dp))
        if sps is None:
            sps = self.get_first_speakers()
        print(f"Measure IR response for these speakers:\n {sps}")
        # turn into sound with correct number of channels
        sds = []
        for sp in sps:
            sds.append(np.array(sp)[:, None] * dp[None, :])
        # play each sound at each position
        ir_field = self.measure_pfield(sds, startx, stopx, nx, starty, stopy, ny, repeats)
        return ir_field

    def deltapulse(self, samplerate, duration=0.01, offset=0.005, uptime=0):
        """
        Create a kronecker delta pulse
        """
        nsp = int(duration * samplerate)
        start = int(offset * samplerate)
        wav = np.zeros(nsp)
        edgesmpls = max(1, int(self.sr * uptime) // 2)
        rise = np.arange(1, edgesmpls + 1) / edgesmpls
        fall = rise[::-1]
        if edgesmpls > 1:
            wav[start:start + edgesmpls] = rise
            wav[start + edgesmpls:start + 2 * edgesmpls] = fall
        else:
            wav[start] = 1
        return wav

    def measure_pfield(self, sds, startx, stopx, nx, starty, stopy, ny, repeats):
        # sds is list of sounds
        nsounds = len(sds)
        sdsamples = [sd.shape[1] for sd in sds]
        nsamples = np.max(sdsamples)  # should have all the same number of samples atm
        Daq = daq.DAQPlaybackMultiStim(conf=self.conf, wavsamples=nsamples)

        # grid
        St = motor.AdafruitStepper(self.port)
        yscm, xscm = np.meshgrid(np.linspace(starty, stopy, ny), np.linspace(startx, stopx, nx), indexing='ij')
        ysth, xsth = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        xscm[1::2, :] = xscm[1::2, ::-1]  # shorter path
        xsth[1::2, :] = xsth[1::2, ::-1]  # same trafo for index
        xscm = xscm.flatten()
        yscm = yscm.flatten()
        xsth = xsth.flatten()
        ysth = ysth.flatten()

        # show planned path
        if 1:
            plt.figure(figsize=(8, 8 * (stopy - starty) // (stopx - startx)))
            plt.plot(xscm, -yscm, 'o-')
            plt.scatter(xscm, -yscm, s=600, marker='x', c='green')
            plt.show()

        # measure
        p_field = np.zeros((nsamples, ny, nx, nsounds, repeats))
        for i, (x, y) in enumerate(zip(xscm, yscm)):
            xth, yth = xsth[i], ysth[i]
            print(f"xth: {xth}, yth: {yth}")
            print(f"{x}cm, {y}cm")
            St.goto_cm(x, y)
            time.sleep(.3)  # wait to rest
            # repeats
            for j, sd in enumerate(sds):
                print(f"playback: {repeats} repeats.")
                for k in range(repeats):
                    p_field[:, yth, xth, j, k] = self.m2p * np.squeeze(
                        Daq.play_and_record(sd))  # origin top left, image/matrix convention
                    time.sleep(.05)
        Daq.close()
        return p_field

    def get_kernelfield(self, ir_field, startx, stopx, nx, starty, stopy, ny, repeats, dp_param, kernel_highpassf,
                        kernel_lowpassf, kernel_off_smooth_fraction):
        pfield = ir_field.copy()
        # filtering of IR responses
        if kernel_highpassf > 0:
            # pad to get rid of filter artefacts
            nsmpls = int(self.sr)
            pfield = self.padsamples(pfield, add_samples=nsmpls, axis=0, only_end=False)
            pfield = self.butter_highpass_filter(pfield.T, cut=kernel_highpassf, fs=self.sr, repeat=4).T
            pfield = pfield[nsmpls:-nsmpls]
            print(f" High-pass filtered IR pressure responses at {kernel_highpassf}Hz")

        if kernel_lowpassf > 0:
            # pad to get rid of filter artefacts
            nsmpls = int(self.sr)
            pfield = self.padsamples(pfield, add_samples=nsmpls, axis=0, only_end=False)
            pfield = self.butter_lowpass_filter(pfield.T, cut=kernel_lowpassf, fs=self.sr, repeat=4).T
            pfield = pfield[nsmpls:-nsmpls]
            print(f" Low-pass filtered IR pressure responses at {kernel_lowpassf}Hz")

        # average over repeats
        pfield = pfield.mean(4)

        # remove initial delay of deltapulse
        offset = int(dp_param["offset"] * self.sr)
        pfield = pfield[offset:]

        # Clippings to avoid artifacts at kernel onset/offset
        length = len(pfield)
        # onset
        nsmp = max(1, int(1e-4 * self.sr))
        env_start = self.get_env_start(length, climb_samples=nsmp)
        print(
            f" Smoothed onset for initial {nsmp} audio samples of kernel. Check whether response onset is affected!")
        # offset
        fall_samples = int(kernel_off_smooth_fraction * length)
        print(
            f" Smooth offset for last {fall_samples} audio samples of kernel. Check whether response offset is affected!")
        env_end = self.get_env_end(length, fall_samples=fall_samples)
        # multiply envelope along time axis
        pfield = pfield.T
        pfield *= env_start
        pfield *= env_end
        pfield = pfield.T

        # todo: calculate maximally supported frequency given the step size
        kernelfield = self.pfield2obs(pfield, startx, stopx, nx, starty, stopy, ny, self.params["rho"])
        return kernelfield

    def get_env_end(self, length, fall_samples, gaussfrac=0.4):
        env = np.ones(length)
        env[-fall_samples:] = self.smoothing(fall_samples, frac=gaussfrac)
        return env

    def get_env_start(self, length, climb_samples, gaussfrac=0.4):
        env = np.ones(length)
        env[:climb_samples] = self.smoothing(climb_samples, frac=gaussfrac)[::-1]
        return env

    def smoothing(self, nsamples, frac=0.4):
        """
        returns half a gaussian of nsamples length that starts at 1 and stops at 0
        :param nsamples: length of the half gaussian
        :param frac: standard deviation of the gaussian as sample fraction
        """
        nsamples = int(nsamples)
        x = np.linspace(-nsamples, nsamples, nsamples * 2)
        m = 0
        s = frac * nsamples
        gauss = lambda x, m, s: np.exp(-((x - m) / s) ** 2)
        smooth = gauss(x, m, s)[nsamples:]
        smooth -= smooth[-1]  # set last value to 0
        smooth /= smooth[0]  # normalize
        return smooth

    def butter_highpass_filter(self, data, cut, fs, order=4, repeat=4):
        def butter_highpass(cut, fs, order=4):
            nyq = 0.5 * fs
            highcut = cut / nyq
            b, a = butter(order, highcut, btype='highpass')
            return b, a

        b, a = butter_highpass(cut, fs, order=order)
        y = data
        for _ in range(repeat):
            y = filtfilt(b, a, y)
        return y

    def butter_lowpass_filter(self, data, cut, fs, order=3, repeat=4):
        def butter_lowpass(cut, fs, order=5):
            nyq = 0.5 * fs
            lowcut = cut / nyq
            b, a = butter(order, lowcut, btype='lowpass')
            return b, a

        b, a = butter_lowpass(cut, fs, order=order)
        y = data
        for _ in range(repeat):
            y = filtfilt(b, a, y)
        return y

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4, repeat=4):
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a

        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = data
        for _ in range(repeat):
            y = filtfilt(b, a, y)
        return y

    def get_wavs(self, fd):
        fns = sorted(glob.glob(os.path.join(fd, "*.wav")))
        wavs = {}
        lengths = []
        for fn in fns:
            name = os.path.split(fn)[1]
            wav, sr = sf.read(fn)
            assert sr == self.sr
            wav /= np.max(abs(wav))
            wavs[name] = wav
            lengths.append(len(wav))

        # pad all sounds to same duration
        maxlength = np.max(lengths)
        for k, w in wavs.items():
            wavs[k] = self.pad2targetsamples(w, maxlength, axis=0)
        return wavs

    def get_targets_trick(self, wavs, p_amplitude, pad_silence, distance2monopole, smp_horizon):
        # playback from left and right x trick condition x inversion

        condition = "Conditioned"  # single speaker & opposing, orthogonal speaker pair
        active_speakers = [[1, 0, 0, 0], [0, 1, 0, 0]]
        trick_factors = [1, -1]  # 1 is normal, -1 activates opposing pressure pair
        targets = {}

        # left/right speaker
        for sp in active_speakers:
            if sp == [1, 0, 0, 0]:
                asign = 1
                splabel = "sp0"
            elif sp == [0, 1, 0, 0]:
                asign = -1  # acceleration inverts if speaker is on other side
                splabel = "sp1"
            # use three speakers (left or right + opposing orthogonal pair)
            sp_map = [sp, [0, 0, 1, 0], [0, 0, 0, 1]]  # mapping for sound targeting with three observables

            # trick conditions
            for trick_factor in trick_factors:
                if trick_factor == 1:
                    tricklabel = ""
                elif trick_factor == -1:
                    tricklabel = "_pressureInv"

                # for each stimulus
                for name, wav in wavs.items():
                    if pad_silence is not None:
                        # extend duration of stim to catch echoes as daq recording lasts only for duration
                        wav = self.padsamples(wav, add_samples=int(pad_silence * self.sr), axis=0,
                                              only_end=True)
                    p = p_amplitude * wav
                    # zero padding to shift target to a later point in time at which convolutions loose memory
                    p = self.padsamples(p, smp_horizon, axis=0, only_end=False)  # beginning and end
                    # targets have even number of samples (for rfft processing later)
                    p = self.padsamples(p, len(p) % 2, axis=0, only_end=True)
                    # Only pressure is inverted in the trick condition, not acceleration
                    target = {"pressure": trick_factor * p,
                              "acceleration_x": asign * self.a_sphere(p, r=distance2monopole),  # self.a_plane(p)
                              "acceleration_y": 0 * p}

                    tmp = {}
                    tmp["target"] = target
                    tmp["sp_map"] = sp_map
                    key = name + '_' + condition + '_' + splabel + tricklabel
                    targets[key] = tmp

        if 1:  # add motion-only and pressure only stimuli
            for name, wav in wavs.items():
                if pad_silence is not None:
                    # extend duration of stim to catch echoes as daq recording lasts only for duration
                    wav = self.padsamples(wav, add_samples=int(pad_silence * self.sr), axis=0,
                                          only_end=True)
                p = p_amplitude * wav
                # zero padding to shift target to a later point in time at which convolutions loose memory
                p = self.padsamples(p, smp_horizon, axis=0, only_end=False)  # beginning and end
                # targets have even number of samples (for rfft processing later)
                p = self.padsamples(p, len(p) % 2, axis=0, only_end=True)

                # acceleration only
                target = {"pressure": 0 * p,
                          "acceleration_x": self.a_sphere(p, r=distance2monopole),
                          "acceleration_y": 0 * p}

                tmp = {}
                tmp["target"] = target
                tmp["sp_map"] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]  # all speakers help
                key = name + '_' + condition + '_' + "motion-only"
                targets[key] = tmp

                # pressure only
                target = {"pressure": p,
                          "acceleration_x": 0 * p,
                          "acceleration_y": 0 * p}

                tmp = {}
                tmp["target"] = target
                tmp["sp_map"] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]  # all speakers help
                key = name + '_' + condition + '_' + "pressure-only"
                targets[key] = tmp

        return targets

    def get_conditioned(self, kernelfield, targets, lowcut, highcut, env_time):
        wav_cond = {}
        nkernelsamples, ny, nx, nspeakers = kernelfield["pressure"].shape
        print(f"Band-pass filters conditioned sounds between {lowcut}Hz and {highcut}Hz")
        print(f"Smoothes start and end of sounds")
        for name, d in targets.items():
            tg = d["target"]
            sp = d["sp_map"]
            nsamples = len(tg["pressure"])
            soundmap = np.zeros((nspeakers, nsamples, ny, nx))
            for i in range(ny):
                for j in range(nx):
                    # kernel for each position
                    ks = {k: [v[:, i, j, ii] for ii in range(nspeakers)] for k, v in
                          kernelfield.items()}  # list of kernel waveform, one for each speaker
                    sd = self.find_sound(ks, tg, sp)

                    # filter signal
                    sd = self.butter_bandpass_filter(sd, lowcut=lowcut, highcut=highcut, fs=self.sr, order=4, repeat=4)

                    # smooth to zero
                    env_samples = max(2, int(self.sr * env_time))  # at least 2 samples
                    env_start = self.get_env_start(sd.shape[1], climb_samples=env_samples)
                    env_end = self.get_env_end(sd.shape[1], fall_samples=env_samples)
                    sd *= env_start
                    sd *= env_end
                    soundmap[:, :, i, j] = sd  # nspeakers,nsamples,ny,nx
            wav_cond[name] = soundmap
        return wav_cond

    def get_conditioned_bc(self, kernelfield, targets, lowcut, highcut, env_time, bc_tolerance, relative_weights,
                           peakAmp=None):
        wav_cond = {}
        nkernelsamples, ny, nx, nspeakers = kernelfield["pressure"].shape
        print(f"Band-pass filters conditioned sounds between {lowcut}Hz and {highcut}Hz")
        print(f"Smoothes start and end of sounds")

        keys = list(targets.keys())
        print("Start conditioning...")
        for name in keys:
            print(name)
            d = targets[name]
            tg = d["target"]
            sp = d["sp_map"]
            nsamples = len(tg["pressure"])
            if nsamples % 2:
                print("Make sure that the targets have even number of samples (rfft is taken).")

            soundmap = np.zeros((nspeakers, nsamples, ny, nx))
            for i in range(ny):
                for j in range(nx):
                    # kernel for each position
                    ks = {k: [v[:, i, j, ii] for ii in range(nspeakers)] for k, v in
                          kernelfield.items()}  # list of kernel waveform, one for each speaker
                    print(i, j)
                    print("Use sound conditioning with bounds...")
                    sd = self.find_sound_bc(ks, tg, sp, bc_tolerance=bc_tolerance, relative_weights=relative_weights,
                                            wav_enforced=None, apply_to=0)

                    # filter signal
                    sd = self.butter_bandpass_filter(sd, lowcut=lowcut, highcut=highcut, fs=self.sr, order=4, repeat=4)

                    # smooth to zero
                    env_samples = max(2, int(self.sr * env_time))  # at least 2 samples
                    env_start = self.get_env_start(sd.shape[1], climb_samples=env_samples)
                    env_end = self.get_env_end(sd.shape[1], fall_samples=env_samples)
                    sd *= env_start
                    sd *= env_end

                    # rescale if needed
                    if peakAmp is not None:
                        sd_peak = abs(sd).max()
                        if sd_peak > peakAmp:
                            sd *= peakAmp / sd_peak
                            print(f"Signal was at {sd_peak}V and was rescaled to {peakAmp}V")

                    soundmap[:, :, i, j] = sd  # nspeakers,nsamples,ny,nx
            wav_cond[name] = soundmap
        return wav_cond

    def get_conditioned_bc_trick(self, kernelfield, targets, lowcut, highcut, env_time, peakAmp=None):
        wav_cond = {}
        nkernelsamples, ny, nx, nspeakers = kernelfield["pressure"].shape
        print(f"Band-pass filters conditioned sounds between {lowcut}Hz and {highcut}Hz")
        print(f"Smoothes start and end of sounds")

        keys_notrick = [k for k in list(targets.keys()) if "pressureInv" not in k]
        keys_trick = [k for k in list(targets.keys()) if "pressureInv" in k]
        keys = keys_notrick + keys_trick
        print("First optimize for normal conditions then for trick conditions...")
        for name in keys:
            print(name)
            notrick = "pressureInv" not in name
            pressure_or_motion_only = "only" in name
            if not notrick:
                matching_name = name.split('_pressureInv')[0]
                pickspeaker = int(name.split('_sp')[1][0])

            d = targets[name]
            tg = d["target"]
            sp = d["sp_map"]
            nsamples = len(tg["pressure"])
            if nsamples % 2:
                print("Make sure that the targets have even number of samples (rfft is taken).")

            soundmap = np.zeros((nspeakers, nsamples, ny, nx))
            for i in range(ny):
                for j in range(nx):
                    # kernel for each position
                    ks = {k: [v[:, i, j, ii] for ii in range(nspeakers)] for k, v in
                          kernelfield.items()}  # list of kernel waveform, one for each speaker
                    print(i, j)
                    if notrick:
                        if not pressure_or_motion_only:
                            print("Suppress opposing pair")
                            sd = self.find_sound_bc(ks, tg, sp, bc_tolerance=5, relative_weights=[1, .1, .1],
                                                    # 2, [1,.1,.1] #1.5
                                                    wav_enforced=None, apply_to=0)
                        else:
                            print("All speakers equally active")
                            sd = self.find_sound_bc(ks, tg, sp, bc_tolerance=5, relative_weights=[1, 1, 1, 1],
                                                    # 2, [1,.1,.1] #1.5
                                                    wav_enforced=None, apply_to=0)
                    else:
                        print("Trick condition with non-trick waveform for single speaker.")
                        wav_enforced = wav_cond[matching_name][pickspeaker, :, i, j]
                        sd = self.find_sound_bc(ks, tg, sp, bc_tolerance=5, relative_weights=[1, 1, 1],  # 2, [1,1,1]
                                                wav_enforced=wav_enforced, apply_to=0)

                    # filter signal
                    sd = self.butter_bandpass_filter(sd, lowcut=lowcut, highcut=highcut, fs=self.sr, order=4, repeat=4)

                    # smooth to zero
                    env_samples = max(2, int(self.sr * env_time))  # at least 2 samples
                    env_start = self.get_env_start(sd.shape[1], climb_samples=env_samples)
                    env_end = self.get_env_end(sd.shape[1], fall_samples=env_samples)
                    sd *= env_start
                    sd *= env_end

                    # rescale if needed
                    if peakAmp is not None:
                        sd_peak = abs(sd).max()
                        if sd_peak > peakAmp:
                            sd *= peakAmp / sd_peak
                            print(f"Signal was at {sd_peak}V and was rescaled to {peakAmp}V")

                    soundmap[:, :, i, j] = sd  # nspeakers,nsamples,ny,nx
            wav_cond[name] = soundmap
        return wav_cond

    def find_sound(self, ks, tg, sp):
        """
        Computes sound that can generate the desired target signals, given kernels.
        Kernels are given for each speaker. However, one could couple several speakers, a mapping which
        is defined by the sp variable.

        To solve the system of equations in the Fourier domain,
        target signals and kernels are zero-padded to equal length.

        L: Length of target waveform (duration)
        K: Length of impulse response (duration)
        Mt: Number of target wav_folder
        Mi: Number of kernels
        M=Mi=Mt, number of target wav_folder must be equal to number of kernels

        :param ks: dict, kernels. (key: observable, val: list of kernel wav_folder, one for each speaker)
        :param tg: dict, targets. (key: observable, val: target wav_folder). Target waveform should be zero-padded at start and end by several kernel-durations
        :param sp: array, speaker choice: picks Mi (combinations of) kernels,
                    e.g. [[1,0],[0,1]] for a signal=speaker mapping in the case of two observables and two speakers.
        :return: array, dim: (K+L) x M, sound that generates target wav_folder
        """
        # check whether kernels and targets refer to the same observables
        if not set(tg.keys()).issubset(set(ks.keys())):
            raise ValueError("Target wav_folder are defined in terms of observables for which no kernels are known.")

        # dict to array
        keys = sorted(tg.keys())
        tgs = np.array([tg[key] for key in keys]).T  # L x Mt
        irs = np.array([np.dot(sp, ks[key]) for key in keys]).T  # K x Mi x Mt

        # work in rescaled space, so that each observable is weighted equally by a solver
        maxvals = abs(irs).max(0).mean(
            0) + 1e-15  # get typical scale for each observable, based on mean over kernels maxs
        print(f"Normalize by {maxvals}")
        tgs /= maxvals[None, :]
        irs /= maxvals[None, None, :]

        # zero padding for equal duration
        length = len(tgs)
        if length < len(irs):
            raise ValueError("Target wav_folder should be longer than the kernel duration (and zero-padded).")
        irs_pad = self.padsamples(irs, length - len(irs), axis=0, only_end=True)
        tgs_pad = tgs

        # fourier domain
        tgs_pad_f = np.fft.fft(tgs_pad.T).T  # (K+L) x Mt
        irs_pad_f = np.fft.fft(irs_pad.T).T  # (K+L) x Mi x Mt
        # solve
        if irs_pad_f.ndim == 3:
            irs_pad_f = np.swapaxes(irs_pad_f, 1, 2)

        #####EXACT SOLUTION#####
        # sg_pad_f = np.linalg.solve(irs_pad_f, tgs_pad_f)  # (K+L) x Mi

        #####LEAST SQUARES SOLUTION#####
        sg_pad_f = []
        for (ipf, tpf) in zip(irs_pad_f, tgs_pad_f):
            spf, spfres = np.linalg.lstsq(ipf, tpf, rcond=None)[:2]
            sg_pad_f.append(spf)
        sg_pad_f = np.array(sg_pad_f)

        # transform back
        sg_pad = np.real(np.fft.ifft(sg_pad_f.T).T)  # (K+L) x Mi
        sound = np.dot(sg_pad, sp).T  # Number of connected speakers x (K+L)
        return sound

    def find_sound_bc(self, ks, tg, sp, bc_tolerance=1, relative_weights=None, wav_enforced=None, apply_to=0):
        """
        :param bc_tolerance: float, an estimate of the maximal signal loadings is
        calculated from the target waveform loadings for each Fourier component.
        Bc_tolerance is the factor by which the speaker loading can exceed the target's loading.
        :param relative_weights: array, n-speaker dimensional. A weight for each speaker as specified in sp.
        The boundaries that are derived from the target waveform and weighted by bc_tolerance are now weighted
        specifically for each speaker. By setting some weights -> 0, sparse solutions can be enforced.
        :param wav_enforced: complex array, waveform dimensional. One may want to constrain the least square algorithm
        to reproduce a speaker activation.
        :param apply_to: int: to which speaker should wav be enforced, axis refers to speaker in sp space.
        """

        def real_to_complex(z):
            return z[:len(z) // 2] + 1j * z[len(z) // 2:]

        def complex_to_real(z):
            return np.concatenate((np.real(z), np.imag(z)), axis=0)

        def complex_2D_to_real(z):
            h, w = z.shape
            A = np.zeros((2 * h, 2 * w))
            A[:h, :w] = np.real(z)
            A[h:, w:] = np.real(z)
            A[:h, w:] = -np.imag(z)
            A[h:, :w] = np.imag(z)
            return A

        # check whether kernels and targets refer to the same observables
        if not set(tg.keys()).issubset(set(ks.keys())):
            raise ValueError("Target wav_folder are defined in terms of observables for which no kernels are known.")
        # dict to array
        keys = sorted(tg.keys())
        tgs = np.array([tg[key] for key in keys]).T  # L x Mt
        irs = np.array([np.dot(sp, ks[key]) for key in keys]).T  # K x Mi x Mt
        ntargets = tgs.shape[-1]

        # work in rescaled space, so that each observable is weighted equally by a solver
        maxvals = abs(irs).max(0).mean(
            0) + 1e-15  # get typical scale for each observable, based on mean over kernels maxs
        tgs /= maxvals[None, :]
        irs /= maxvals[None, None, :]

        # zero padding for equal duration
        length = len(tgs)
        if length < len(irs):
            raise ValueError("Target wav_folder should be longer than the kernel duration (and zero-padded).")
        irs_pad = self.padsamples(irs, length - len(irs), axis=0, only_end=True)
        tgs_pad = tgs

        # fourier domain
        tgs_pad_f = np.fft.rfft(tgs_pad.T).T  # L x Mt
        irs_pad_f = np.fft.rfft(irs_pad.T).T  # L x Mi x Mt
        # solve
        if irs_pad_f.ndim == 3:
            irs_pad_f = np.swapaxes(irs_pad_f, 1, 2)  # L x Mt x Mi

        #####LEAST SQUARES SOLUTION WITH BOUNDS#####
        if wav_enforced is not None:
            wav_f = np.fft.rfft(wav_enforced)  # L
            wav_f_mean = abs(wav_f).mean()

        sg_pad_f = []
        mean_irs = abs(irs_pad_f).mean(axis=(0, -1))  # mean across freqs, mean across speakers -> Mt dimensional
        for i, (ipf, tpf) in enumerate(zip(irs_pad_f, tgs_pad_f)):
            # compute boundary values from target fourier components
            bd = abs(tpf) / (mean_irs + 1e-15)
            bd = bc_tolerance * bd.max() + 1e-15
            bds = bd * np.ones(2 * ntargets)  # same boundaries for all speakers (real and imaginary parts)
            # can favour speakers
            if relative_weights is not None:
                bds = np.tile(bd * np.array(relative_weights), 2) + 1e-15
            # set bounds
            upper = bds
            lower = -bds
            # overwrite some bounds if a certain waveform should be enforced
            if wav_enforced is not None:
                tolerance = 1e-8 * wav_f_mean
                upper[apply_to] = wav_f[i].real + tolerance / 2
                upper[apply_to + ntargets] = wav_f[i].imag + tolerance / 2
                lower[apply_to] = wav_f[i].real - tolerance / 2
                lower[apply_to + ntargets] = wav_f[i].imag - tolerance / 2

            # cast complex N to real space 2N
            ipf = complex_2D_to_real(ipf)
            tpf = complex_to_real(tpf)
            spf = scipy.optimize.lsq_linear(ipf, tpf, bounds=(lower, upper), method='trf')['x']  # trf, bvls
            spf = real_to_complex(spf)
            sg_pad_f.append(spf)
        sg_pad_f = np.array(sg_pad_f)

        # TRANSFORM BACK
        sg_pad = np.fft.irfft(sg_pad_f.T).T  # (K+L) x Mi
        sound = np.dot(sg_pad, sp).T  # Number of connected speakers x (K+L)
        return sound

    def pad2targetsamples(self, sd, targetsamples, axis, only_end=True):
        add_samples = targetsamples - sd.shape[axis]
        sd = self.padsamples(sd, add_samples, axis, only_end)
        return sd

    def padsamples(self, sd, add_samples, axis, only_end=True):
        add_samples = int(add_samples)
        if add_samples <= 0:
            return sd
        npad = [(0, 0)] * sd.ndim
        if only_end:
            npad[axis] = (0, add_samples)
        else:
            npad[axis] = (add_samples, add_samples)
        sd = np.pad(sd, pad_width=npad, mode='constant', constant_values=0)
        return sd

    def run_tests_only_p(self, sounds_cdmap, startx, stopx, nx, starty, stopy, ny, repeats):
        fields = {}
        nsamples = np.max([sd.shape[1] for sd in list(sounds_cdmap.values())])
        Daq = daq.DAQPlaybackMultiStim(conf=self.conf, wavsamples=nsamples, ai_channels=True, accel_channels=False)

        # grid
        St = motor.AdafruitStepper(self.port)
        yscm, xscm = np.meshgrid(np.linspace(starty, stopy, ny), np.linspace(startx, stopx, nx), indexing='ij')
        ysth, xsth = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        xscm[1::2, :] = xscm[1::2, ::-1]  # shorter path
        xsth[1::2, :] = xsth[1::2, ::-1]  # same trafo for index
        xscm = xscm.flatten()
        yscm = yscm.flatten()
        xsth = xsth.flatten()
        ysth = ysth.flatten()

        # data structure
        for k in sounds_cdmap.keys():
            fields[k] = np.zeros(
                (nsamples, ny, nx, repeats))  # for each position each position-dependent conditioned stim.

        # visit all points
        for i, (x, y) in enumerate(zip(xscm, yscm)):
            xth, yth = xsth[i], ysth[i]
            print(f"xth: {xth}, yth: {yth}")
            print(f"{x}cm, {y}cm")
            St.goto_cm(x, y)
            time.sleep(.3)  # come to rest
            # conditioned sounds
            for k, v in sounds_cdmap.items():
                print(f"Play {k}")
                sd = v[:, :, yth, xth]  # only look at target pressure
                sd = self.pad2targetsamples(sd, nsamples, axis=1)
                for j in range(repeats):
                    fields[k][:, yth, xth, j] = self.m2p * np.squeeze(
                        Daq.play_and_record(sd))  # origin top left, image/matrix convention
                    time.sleep(.05)
        Daq.close()
        return fields

    def run_tests_only_accel(self, sounds_cdmap, startx, stopx, nx, starty, stopy, ny, repeats):
        fields = {}
        nsamples = np.max([sd.shape[1] for sd in list(sounds_cdmap.values())])
        Daq = daq.DAQPlaybackMultiStim(conf=self.conf, wavsamples=nsamples, ai_channels=False, accel_channels=True)
        time.sleep(5)  # wait for bias voltage to build up?

        # grid
        St = motor.AdafruitStepper(self.port)
        yscm, xscm = np.meshgrid(np.linspace(starty, stopy, ny), np.linspace(startx, stopx, nx), indexing='ij')
        ysth, xsth = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        xscm[1::2, :] = xscm[1::2, ::-1]  # shorter path
        xsth[1::2, :] = xsth[1::2, ::-1]  # same trafo for index
        xscm = xscm.flatten()
        yscm = yscm.flatten()
        xsth = xsth.flatten()
        ysth = ysth.flatten()

        # data structure
        for k in sounds_cdmap.keys():
            fields[k] = np.zeros(
                (nsamples, ny, nx, repeats,
                 3))  # for each position each position-dependent conditioned stim and 3 accel axes.

        # visit all points
        for i, (x, y) in enumerate(zip(xscm, yscm)):
            xth, yth = xsth[i], ysth[i]
            print(f"xth: {xth}, yth: {yth}")
            print(f"{x}cm, {y}cm")
            St.goto_cm(x, y)
            time.sleep(.3)  # come to rest
            # conditioned sounds
            for k, v in sounds_cdmap.items():
                print(f"Play {k}")
                sd = v[:, :, yth, xth]  # only look at target pressure
                sd = self.pad2targetsamples(sd, nsamples, axis=1)
                for j in range(repeats):
                    fields[k][:, yth, xth, j, :] = self.m2a * np.squeeze(
                        Daq.play_and_record(sd))  # origin top left, image/matrix convention
                    time.sleep(.05)
        Daq.close()
        return fields

    def run_tests(self, sounds_cdmap, wavs, test_static, startx, stopx, nx, starty, stopy, ny, repeats):
        fields = {}
        if test_static:
            # estimate voltage scaling for naive sound from conditioned sound amplitudes at center of sound field
            centerx = nx // 2
            centery = ny // 2
            centersds = [abs(sd[:2, :, centery, centerx]).max() for k, sd in sounds_cdmap.items()]
            voltage_amp = np.mean(centersds)  # mean of max loading for first two channels across stimuli
            sps = [[1, 0, 0, 0], [0, 1, 0, 0]]  # left and right
            print(f"Chose {voltage_amp}V as peak amplitude for static, non-conditioned sound.")

        nsamples = np.max([sd.shape[1] for sd in list(sounds_cdmap.values())])
        Daq = daq.DAQPlaybackMultiStim(conf=self.conf, wavsamples=nsamples)

        # grid
        St = motor.AdafruitStepper(self.port)
        yscm, xscm = np.meshgrid(np.linspace(starty, stopy, ny), np.linspace(startx, stopx, nx), indexing='ij')
        ysth, xsth = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        xscm[1::2, :] = xscm[1::2, ::-1]  # shorter path
        xsth[1::2, :] = xsth[1::2, ::-1]  # same trafo for index
        xscm = xscm.flatten()
        yscm = yscm.flatten()
        xsth = xsth.flatten()
        ysth = ysth.flatten()

        # data structure
        for k in sounds_cdmap.keys():
            fields[k] = np.zeros(
                (nsamples, ny, nx, repeats, ny, nx))  # for each position each position-dependent conditioned stim.
        if test_static:
            for k in wavs.keys():
                fields[k] = np.zeros(
                    (nsamples, ny, nx, repeats, len(sps)))  # same sound for each position from different speakers

        # show planned path
        if 0:
            plt.figure(figsize=(8, 8 * (stopy - starty) // (stopx - startx)))
            plt.plot(xscm, -yscm, 'o-')
            plt.scatter(xscm, -yscm, s=600, marker='x', c='green')
            plt.show()

        # visit all points
        for i, (x, y) in enumerate(zip(xscm, yscm)):
            xth, yth = xsth[i], ysth[i]
            print(f"xth: {xth}, yth: {yth}")
            print(f"{x}cm, {y}cm")
            St.goto_cm(x, y)
            time.sleep(.3)  # come to rest
            # conditioned sounds
            for k, v in sounds_cdmap.items():
                print(f"Play {k}")
                for ii in range(ny):
                    for jj in range(nx):
                        sd = v[:, :, ii, jj]  # use sounds targeted for all positions
                        sd = self.pad2targetsamples(sd, nsamples, axis=1)
                        for j in range(repeats):
                            fields[k][:, yth, xth, j, ii, jj] = self.m2p * np.squeeze(
                                Daq.play_and_record(sd))  # origin top left, image/matrix convention
                            time.sleep(.05)
            # naive sounds
            if test_static:
                for k, v in wavs.items():
                    for hh, sp in enumerate(sps):
                        print(f"Play {k} via {sp}")
                        wav_scaled = voltage_amp * v
                        sd = np.array(sp)[:, None] * wav_scaled[None, :]
                        sd = self.pad2targetsamples(sd, nsamples, axis=1)
                        for j in range(repeats):
                            fields[k][:, yth, xth, j, hh] = self.m2p * np.squeeze(
                                Daq.play_and_record(sd))  # origin top left, image/matrix convention
                            time.sleep(.05)
        Daq.close()
        return fields

    def get_corrected(self, sound_cdmap0, test0, kernelfield, targets, lowcut, highcut, env_time, startx, stopx, nx,
                      starty,
                      stopy, ny):
        sound_cdmap1 = {}
        for k in sound_cdmap0.keys():
            # new conditioned sound map
            sds = sound_cdmap0[k]
            nspeaker = sds.shape[0]

            # same targets as before
            tg = targets[k]['target']
            sp_map = targets[k]['sp_map']

            # pressure fields for all target positions
            pfields = test0[k].mean(3)  # mean across repetitions

            # loop over all targeting positions
            soundmap = np.zeros(sds.shape)
            for i in range(ny):
                for j in range(nx):
                    pfield = pfields[:, :, :, i, j]  # nspeaker, y, x, target-for-y, target-for-x
                    obs = self.pfield2obs(pfield, startx, stopx, nx, starty, stopy, ny, self.params["rho"])
                    ks = {kk: [v[:, i, j, ii] for ii in range(nspeaker)] for kk, v in
                          kernelfield.items()}
                    padsmpls = int(self.sr / lowcut)
                    tg_diff = {q: np.pad(tg[q] - obs[q][:, i, j], padsmpls) for q in tg.keys()}
                    sd_diff = self.find_sound(ks, tg_diff, sp_map)[:, padsmpls:-padsmpls]
                    sdnew = sds[:, :, i, j] - sd_diff  # nspeaker, time, y, x

                    # filter signal
                    sdnew = self.butter_bandpass_filter(sdnew, lowcut=lowcut, highcut=highcut, fs=self.sr, order=4,
                                                        repeat=4)

                    # smooth to zero
                    env_samples = max(2, int(self.sr * env_time))  # at least 2 samples
                    env_start = self.get_env_start(sdnew.shape[1], climb_samples=env_samples)
                    env_end = self.get_env_end(sdnew.shape[1], fall_samples=env_samples)
                    sdnew *= env_start
                    sdnew *= env_end

                    soundmap[:, :, i, j] = sdnew
            sound_cdmap1[k] = soundmap
        return sound_cdmap1

    def pfield2obs(self, pfield, startx, stopx, nx, starty, stopy, ny, rho):
        result = {}
        # turn into observables
        # step sizes
        dx = 1e-2 * (stopx - startx) / (nx - 1)  # start/stop in cm
        dy = 1e-2 * (stopy - starty) / (ny - 1)
        result["pressure"] = pfield  # dim: nsamples, ny, nx, nspeakers
        result["acceleration_y"] = - np.gradient(pfield, axis=1) / dy / rho  # Euler Eq.
        result["acceleration_x"] = - np.gradient(pfield, axis=2) / dx / rho  # Euler Eq.
        return result

    def a_sphere(self, p, r=0.1):
        """
        see e.g. Eq A3 in https://doi.org/10.1242/jeb.093831, v = (1+i/kr)*p/(rho*c)
        :param p:
        :param r:
        :return:
        """
        sr = self.sr
        # radial acceleration from pressure for a spherical wave
        k = 2 * np.pi * np.fft.fftfreq(p.shape[-1])[1:] * sr / self.params[
            "c"]  # exclude DC component (divergence of 1/kr)
        coeff = 1 + 1j / (k * r)
        derivative = -2 * np.pi * 1j * np.fft.fftfreq(p.shape[-1])[1:] * sr
        tmp = np.fft.fft(p)[1:] * coeff * derivative
        a = np.real(np.fft.ifft(np.insert(tmp, 0, 0) / (self.params["rho"] * self.params["c"])))  # insert zero DC
        return a

    def a_sphere_r_profile(self, p_wave, r_profile, scale_p=False, r0=None):
        """
        p: pressure waveform
        r_profile: distance profile that matches length of p
        scale_p: if True, scales p and a. Otherwise only scales a according to r_profile.
        r0: distance that p refers to, only used if scale_p=True
        """
        r = r_profile
        sr = self.sr
        p = p_wave.copy()

        if scale_p:
            p = p * r0 / r  # rescale pressure with 1/r profile

        r = r[:, None]
        f = np.fft.rfftfreq(len(p), d=1 / sr)[None, :]
        p_f = np.fft.rfft(p)[None, :]
        t = (np.arange(len(p)) / sr)[:, None]
        p_wavelets = p_f * np.exp(1j * 2 * np.pi * f * t)

        k = 2 * np.pi * f / self.params["c"]
        a_wavelets = (1 / r - 1j * k) / self.params["rho"] * p_wavelets
        return p, np.real(a_wavelets.mean(1))

    def get_r_profile(self, n, dstart, dend, tstart, tend):
        r = dstart * np.ones(n)
        r[tstart:tend] = np.linspace(dstart, dend, tend - tstart)
        r[tend:] = dend
        return r

    def a_plane(self, p):
        v = p / (self.params["rho"] * self.params["c"])
        a = np.gradient(v) * self.sr
        return a

    def get_first_speakers(self, n=None):
        """
        Default speakers for playback: take first n speakers, given n signals to playback.
        :param n: number of speakers to take, all speakers if None.
        :return: speaker configuration
        """
        nch = len(self.conf["audio_out"]["chStr"])
        speakers = np.identity(nch, dtype=bool)[:n].tolist()
        return speakers

    def get_speakers_custom(self):
        """
        Define manual speaker configuration. Given n observables, one can target n signals.
        A signal can be mapped to a combination of speakers.
        """
        speakers = [[1, 1, 0], [0, 1, 0]]  # n x number of connected speakers
        return speakers


if __name__ == "__main__":
    main()
