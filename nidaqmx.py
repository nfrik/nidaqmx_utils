import nidaqmx
from nidaqmx import stream_writers, stream_readers
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import logging
import json
from utils import plott
from tqdm import tqdm
import pandas as pd
from threading import Thread
from scipy import signal
from scipy.ndimage.interpolation import shift
from scipy.signal import find_peaks
from datetime import datetime
from hyperopt import hp, tpe, fmin
import os

ttables = {}
ttables['xor'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
ttables['xor3'] = [[-1, -1, -1, 0],[-1, -1, 1, 1],[-1, 1, -1, 0], [-1, 1, 1, 1],[1, -1, -1, 1],[1, -1, 1, 0],[1, 1, -1, 1],[1, 1, 1, 0]]
ttables['or'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
ttables['and'] = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

logger = logging.getLogger(__name__)


class Measurement:
    def __init__(self,peak_find_delta=2):
        self.current_folder_name=None
        self.file_index=0
        self.peak_find_delta=peak_find_delta
        self.optim_score=[]

    def perturb_X(self,X,boost=3,var=1):
        # Y=X.copy()
        Y=X*boost
        for idx,x in np.ndenumerate(Y):
            Y[idx]+=(np.random.rand() - 0.5) * var
    #     result = np.array(list(map(lambda t: boost*t + (np.random.rand() - 0.5) * var, X)))
        return Y

    def bool_tt(self,ary=3,op='^'):
        assert ary>1
        exprs=[]
        results=[]
        for i in range(2**ary):
            expr=bin(i)[2:].zfill(ary)
            res=eval(op.join(list(expr)))
            expr=[-1 if c=='0' else 1 for c in list(expr)]
            expr.append(res)
            exprs.append(list(expr))
        return exprs

    def get_tt(self,op='^',nary=3,periods=6,boost=1,var=0.0):
        data=np.array(self.bool_tt(ary=nary,op=op)*periods)
        # X = data[:,:-1]
        # y = data[:,-1]
        return data

    def get_pulse_train(self,amp,samps,npulse,wpulse):

        t = np.linspace(0, 1, samps)
        signals = amp*signal.square(2 * np.pi * npulse * t, wpulse)
        return signals


    def filter_sharps(self,data, peakstarts, peakends):
        # points = np.array([[a, b] for a, b in zip(peakstarts, peakends)]).ravel()
        # pointmap = np.ones_like(data)
        # for point in points:
        #     pointmap[:,point] = 0
        #     pointmap[:,point - 1] = 0
        #     pointmap[:,point + 1] = 0

        # return np.multiply(data, pointmap)

        for p_s in peakstarts:
            try:
                data[:,p_s] = data[:,p_s-self.peak_find_delta]
                data[:, p_s-1] = data[:, p_s - self.peak_find_delta]
                data[:, p_s+1] = data[:, p_s - self.peak_find_delta]
            except:
                pass

        for p_s in peakends:
            try:
                data[:,p_s] = data[:,p_s+self.peak_find_delta]
                data[:, p_s-1] = data[:, p_s + self.peak_find_delta]
                data[:, p_s+1] = data[:, p_s + self.peak_find_delta]
            except:
                pass

        return data

    def remove_baseline(self,data,peakstarts,peakends):

        pre_trend = data[:,peakstarts - self.peak_find_delta]
        pos_trend = data[:,peakends - self.peak_find_delta]

        trend = np.hstack((pre_trend,pos_trend))

        trend = np.mean(trend,axis=1)

        data = np.subtract(data, trend.reshape((-1, 1)))

        return data

    def channel_collision(self,*lists):
        keys=[]
        # for d in dicts:
        #     keys.append(d.keys())
        for l in lists:
            keys.append(set(l))

        unique=set.intersection(*keys)

        if len(list(unique))>0:
            return True
        return False


    def multi_measurement2(self, indata={}, contdata={}, outchans=[], pulse_dur=0.1, samps_pulse=1000, duty=0.5, signal_type='triangle', peak_type='max'):
        # binsteps,inchans=indata.shape
        inchans = len(indata.keys())
        binsteps = len(indata.items()[0])

        if self.channel_collision(indata,contdata):
            raise ValueError('Input and control channels should not intersect')

        freq = round(samps_pulse/pulse_dur)

        samps = binsteps * samps_pulse
        total_time = int(np.ceil(1/freq * samps))
        print("Frequency: {}, Samples: {}, Time required: {}".format(freq,samps,total_time))

        if freq>25000:
            raise ValueError('AO frequency exceeded 25kS/s/ch')

        x_p = np.linspace(0, 1, samps_pulse)

        # if 'square' in signal_type:
        y_p = (signal.square(2*np.pi*x_p+2*np.pi/2,duty)+1)/2

        peak_rise_idx = list(y_p).index(1)
        peak_fall_idx = len(y_p) - list(y_p[::-1]).index(1) - 1

        if 'triang' in signal_type:
            y_p = (signal.sawtooth(2 * np.pi * 2 * x_p, 0.5) + 1) / 2
            y_p = shift(y_p, int(np.floor(samps_pulse / 2)))

        # construct list of peak rising indxs

        peak_start_idx_lst=np.arange(peak_rise_idx, samps, samps_pulse)
        peak_end_idx_lst = np.arange(peak_fall_idx, samps, samps_pulse)


        #prepare input data
        insig=np.zeros((inchans, samps))

        for bincol, chan in zip(indata[sorted(indata.keys())], range(inchans)):
            pulses=[]
            for binval in bincol:
                pulses=np.append(pulses,binval*y_p)
            insig[chan]=pulses

        # Due to 1 step delay in the 1st channel we need to apply a hack to shift data by one step
        insig[0, :] = shift(insig[0, :], -1)

        # prepare input data
        insig = np.zeros((inchans, samps))

        for bincol, chan in zip(indata[sorted(indata.keys())], range(inchans)):
            pulses = []
            for binval in bincol:
                pulses = np.append(pulses, binval * y_p)
            insig[chan] = pulses

        # Due to 1 step delay in the 1st channel we need to apply a hack to shift data by one step
        insig[0, :] = shift(insig[0, :], -1)

        with nidaqmx.Task() as taskin, nidaqmx.Task() as taskout:

            for inchan in indata.keys():
                taskout.ao_channels.add_ao_voltage_chan("AO/ao{}".format(inchan))

            for contchan in contdata.keys():
                    taskout.ao_channels.add_ao_voltage_chan("AO/ao{}".format(contchan))

            for outchan in outchans:
                taskin.ai_channels.add_ai_voltage_chan("AI/ai{}".format(outchan),
                                                           terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)

            taskout.timing.cfg_samp_clk_timing(freq, samps_per_chan=samps)
            taskin.timing.cfg_samp_clk_timing(freq, samps_per_chan=samps, source='ao/SampleClock')

            stream_O = stream_writers.AnalogMultiChannelWriter(taskout.out_stream, auto_start=False)
            stream_I = stream_readers.AnalogMultiChannelReader(taskin.in_stream)

            #Upload signal to hardware buffer
            stream_O.write_many_sample(insig)

            taskin.start()
            taskout.start()

            result = np.zeros((outchans, samps))
            stream_I.read_many_sample(result,timeout=total_time*1.2)
            # taskin.wait_until_done()
            taskout.wait_until_done()


            t=np.arange(samps)/freq

            # print("Looking for peaks")

            result = self.filter_sharps(result, peak_start_idx_lst, peak_end_idx_lst)

            result = self.remove_baseline(result,peak_start_idx_lst, peak_end_idx_lst)

            if 'max' in peak_type:
                real_peaks = []
                for res in result:
                    rp_chan = []
                    for s_p, e_p in zip(peak_start_idx_lst, peak_end_idx_lst):
                        #         peaks,_ = find_peaks(np.abs(sig[s_p:e_p]))
                        peaks = np.argmax(np.abs(res[s_p:e_p]))
                        #         peaks += s_p
                        #         if len(peaks)>0:
                        rp_chan.append((peaks + s_p))
                    real_peaks.append(rp_chan)

            elif 'last' in peak_type:
                real_peaks = []
                for res in result:
                    rp_chan = []
                    for s_p, e_p in zip(peak_start_idx_lst, peak_end_idx_lst):
                        #         peaks,_ = find_peaks(np.abs(sig[s_p:e_p]))
                        # peaks = np.argmax(np.abs(res[s_p:e_p]))
                        #         peaks += s_p
                        #         if len(peaks)>0:
                        rp_chan.append(e_p-2)
                    real_peaks.append(rp_chan)
            # real_peaks

            # a = plt.figure()
            # plt.subplot(211)
            #
            # for r, chan in zip(result,range(outchans)):
            #     # if chan != 1:
            #     plt.plot(t,r, label="out sig {}".format(chan))
            #     plt.plot(t[real_peaks[chan]],r[real_peaks[chan]], 'x',label="out sig {}".format(chan))
            #
            # # for idx in peak_start_idx_lst:
            # #     plt.plot(t[idx],0,'o',color='r')
            # #
            # # for idx in peak_end_idx_lst:
            # #     plt.plot(t[idx], 0, 'o', color='g')
            #
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
            #
            #
            # plt.subplot(212)
            # for s,chan in zip(sig,range(inchans)):
            #     plt.plot(t,s, label="inp sig {}".format(chan))
            #
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
            # # plt.show()
            # a.show()
            #
            #
            # f=plt.figure()
            #
            # plt.subplots_adjust(hspace=0.001)
            #
            # tf = str(result.shape[0])
            #
            # for s, i in zip(result, range(result.shape[0])):
            #     ax1 = plt.subplot(int(tf + '1' + str(i + 1)))
            #     ax1.plot(t,result[i])
            #     plt.plot(t[real_peaks[i]], result[i][real_peaks[i]], 'x')
            #     plt.yticks(np.arange(1.2 * np.min(result[i]), 1.2 * np.max(result[i]), 3))
            #
            # plt.show()
            # # f.show()

            return  {"peak_start_idx":peak_start_idx_lst.tolist(),"peak_end_idx":peak_end_idx_lst.tolist(),"peaks":np.array(real_peaks).tolist(),"result":result.tolist(),"signal":insig.tolist(),"time":t.tolist()}
            #construct trains of pulses for each sample

    def multi_measurement(self, bindata, pulse_dur=0.1, samps_pulse=1000, duty=0.5, signal_type='triangle', peak_type='max', inmask=[], contmask=[], outmask=[]):
        binsteps,inchans = bindata.shape

        if self.channel_collision(inmask,contmask):
            raise ValueError('Input and control channels must have distinct numbers')

        inmask=inmask+contmask

        freq = round(samps_pulse/pulse_dur)

        samps = binsteps * samps_pulse
        total_time = int(np.ceil(1/freq * samps))
        print("Frequency: {}, Samples: {}, Time required: {}".format(freq,samps,total_time))

        if freq>25000:
            raise ValueError('AO frequency exceeded 25kS/s/ch')

        x_p = np.linspace(0, 1, samps_pulse)

        # if 'square' in signal_type:
        y_p = (signal.square(2*np.pi*x_p+2*np.pi/2,duty)+1)/2

        peak_rise_idx = list(y_p).index(1)
        peak_fall_idx = len(y_p) - list(y_p[::-1]).index(1) - 1

        if 'triang' in signal_type:
            y_p = (signal.sawtooth(2 * np.pi * 2 * x_p, 0.5) + 1) / 2
            y_p = shift(y_p, int(np.floor(samps_pulse / 2)))

        # construct list of peak rising indxs

        peak_start_idx_lst=np.arange(peak_rise_idx, samps, samps_pulse)
        peak_end_idx_lst = np.arange(peak_fall_idx, samps, samps_pulse)


        sig=np.zeros((inchans, samps))

        for bincol, chan in zip(bindata.T,range(inchans)):
            pulses=[]
            for binval in bincol:
                pulses=np.append(pulses,binval*y_p)
            sig[chan]=pulses

        # Due to 1 step delay in the 1st channel we need to apply a hack to shift data by one step
        sig[0, :] = shift(sig[0, :], -1)

        with nidaqmx.Task() as taskin, nidaqmx.Task() as taskout:

            for inchan in inmask:
                taskout.ao_channels.add_ao_voltage_chan("AO/ao{}".format(inchan))

            for outchan in outmask:
                taskin.ai_channels.add_ai_voltage_chan("AI/ai{}".format(outchan),terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)

            taskout.timing.cfg_samp_clk_timing(freq, samps_per_chan=samps)
            taskin.timing.cfg_samp_clk_timing(freq, samps_per_chan=samps, source='ao/SampleClock')

            stream_O = stream_writers.AnalogMultiChannelWriter(taskout.out_stream, auto_start=False)
            stream_I = stream_readers.AnalogMultiChannelReader(taskin.in_stream)

            #Upload signal to hardware buffer
            stream_O.write_many_sample(sig)

            taskin.start()
            taskout.start()

            result = np.zeros((len(outmask), samps))
            stream_I.read_many_sample(result,timeout=total_time*1.2)
            # taskin.wait_until_done()
            taskout.wait_until_done()


            t=np.arange(samps)/freq

            # print("Looking for peaks")

            result = self.filter_sharps(result, peak_start_idx_lst, peak_end_idx_lst)

            result = self.remove_baseline(result,peak_start_idx_lst, peak_end_idx_lst)

            if 'max' in peak_type:
                real_peaks = []
                for res in result:
                    rp_chan = []
                    for s_p, e_p in zip(peak_start_idx_lst, peak_end_idx_lst):
                        #         peaks,_ = find_peaks(np.abs(sig[s_p:e_p]))
                        peaks = np.argmax(np.abs(res[s_p:e_p]))
                        #         peaks += s_p
                        #         if len(peaks)>0:
                        rp_chan.append((peaks + s_p))
                    real_peaks.append(rp_chan)

            elif 'last' in peak_type:
                real_peaks = []
                for res in result:
                    rp_chan = []
                    for s_p, e_p in zip(peak_start_idx_lst, peak_end_idx_lst):
                        #         peaks,_ = find_peaks(np.abs(sig[s_p:e_p]))
                        # peaks = np.argmax(np.abs(res[s_p:e_p]))
                        #         peaks += s_p
                        #         if len(peaks)>0:
                        rp_chan.append(e_p-2)
                    real_peaks.append(rp_chan)
            # real_peaks

            return  {"peak_start_idx":peak_start_idx_lst.tolist(),"peak_end_idx":peak_end_idx_lst.tolist(),"peaks":np.array(real_peaks).tolist(),"result":result.tolist(),"signal":sig.tolist(),"time":t.tolist()}
            #construct trains of pulses for each sample

    def analog_measurement(self, andata, freq=1000, inmask=[0],outmask=[0]):
        andata = andata.T.copy()
        inchans,samps = andata.shape

        freq=float(freq)
        total_time = int(np.ceil(1/freq * samps))
        print("Frequency: {}, Samples: {}, Time required: {}".format(freq,samps,total_time))

        if freq>25000:
            raise ValueError('AO frequency exceeded 25kS/s/ch')

        with nidaqmx.Task() as taskin, nidaqmx.Task() as taskout:
            for inchan in inmask:
                taskout.ao_channels.add_ao_voltage_chan("AO/ao{}".format(inchan))
            for outchan in outmask:
                taskin.ai_channels.add_ai_voltage_chan("AI/ai{}".format(outchan),terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)

            taskout.timing.cfg_samp_clk_timing(freq, samps_per_chan=samps,sample_mode=nidaqmx.constants.AcquisitionType.FINITE)
            taskin.timing.cfg_samp_clk_timing(freq, samps_per_chan=samps,sample_mode=nidaqmx.constants.AcquisitionType.FINITE, source='ao/SampleClock')

            stream_O = stream_writers.AnalogMultiChannelWriter(taskout.out_stream, auto_start=False)
            stream_I = stream_readers.AnalogMultiChannelReader(taskin.in_stream)

            #Upload signal to hardware buffer
            stream_O.write_many_sample(andata)

            taskin.start()
            taskout.start()

            result = np.zeros((len(outmask), samps))
            stream_I.read_many_sample(result,timeout=total_time*1.2)
            # taskin.wait_until_done()
            taskout.wait_until_done()

            t=np.arange(samps)/freq

            # print("Looking for peaks")


            return  {"result":result.tolist(),"signal":andata.tolist(),"time":t.tolist()}
            #construct trains of pulses for each sample

    def analog_continuous_measurement(self, andata, freq=1000, inmask=[0],outmask=[0]):
        andata = andata.T.copy()
        inchans,samps = andata.shape

        freq=float(freq)
        total_time = int(np.ceil(1/freq * samps))
        print("Frequency: {}, Samples: {}, Time required: {}".format(freq,samps,total_time))

        if freq>25000:
            raise ValueError('AO frequency exceeded 25kS/s/ch')

        with nidaqmx.Task() as taskin, nidaqmx.Task() as taskout:
            for inchan in inmask:
                taskout.ao_channels.add_ao_voltage_chan("AO/ao{}".format(inchan))
            for outchan in outmask:
                taskin.ai_channels.add_ai_voltage_chan("AI/ai{}".format(outchan),terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)

            taskout.timing.cfg_samp_clk_timing(freq, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
            taskin.timing.cfg_samp_clk_timing(freq, samps_per_chan=samps,sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

            stream_O = stream_writers.AnalogMultiChannelWriter(taskout.out_stream,auto_start=True)
            stream_I = stream_readers.AnalogMultiChannelReader(taskin.in_stream)

            #Upload signal to hardware buffer
            stream_O.write_many_sample(np.append(andata,np.zeros_like(andata[:, 0])))
            try:
                taskout.wait_until_done(timeout=10)
            except:
                pass

            # samples = np.append(5 * np.ones(30), np.zeros(10))
            taskout.stop()
            taskout.close()
            # taskin.start()
            # taskout.start()
            #
            # output = np.zeros

            #
            #
            # result = np.zeros((len(outmask), samps))
            # stream_I.read_many_sample(result,timeout=total_time*1.2)
            # # taskin.wait_until_done()
            # taskout.wait_until_done()
            #
            # t=np.arange(samps)/freq
            #
            # # print("Looking for peaks")
            #
            #
            # return  {"result":result.tolist(),"signal":andata.tolist(),"time":t.tolist()}
            #construct trains of pulses for each sample

    def clear_all_output_chans(self):

        data = np.zeros((1,16))
        replicate = 5
        data = np.repeat(data, replicate, axis=0)

        out = self.multi_measurement(data, pulse_dur=.05, samps_pulse=80, duty=.5, signal_type='square',
                                     peak_type='last', inmask=[0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], contmask=[], outmask=[0, 1, 2, 3, 4, 5, 6, 7])

    def plot_hysteresis(self,results):
        data = np.array(results["result"])
        sig = np.array(results["signal"])
        t = np.array(results['time'])

        a = plt.figure()

        for r, chan in zip(data, range(len(data))):
            # if chan != 1:
            plt.plot(sig[0], r, label="out sig {}".format(chan))

        plt.show()


    def plot_result(self,results,replicate,draw_peaks=False,savepath=''):

        try:
            peaks = np.array(results["peaks"])
            peaks = peaks[:, replicate - 1::replicate]
        except:
            pass
        data = np.array(results["result"])
        sig = np.array(results["signal"])
        t = np.array(results['time'])

        a = plt.figure()
        plt.subplot(211)


        for r, chan in zip(data, range(len(data))):
            # if chan != 1:
            plt.plot(t, r, label="out sig {}".format(chan))
            try:
                if draw_peaks:
                    plt.plot(t[peaks[chan]],r[peaks[chan]], 'x', label="out sig {}".format(chan))
            except:
                pass

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

        plt.subplot(212)
        for s, chan in zip(sig, range(sig.shape[0])):
            plt.plot(t, s, label="inp sig {}".format(chan))

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
        # plt.show()

        plt.tight_layout()
        if savepath!="":
            a.savefig(savepath)
        else:
            a.show()

        f = plt.figure()

        plt.subplots_adjust(hspace=0.001)

        tf = str(len(data))

        for s, i in zip(data, range(len(data))):
            ax1 = plt.subplot(int(tf + '1' + str(i + 1)))
            ax1.plot(t, data[i])
            try:
                plt.plot(t[peaks[i]], data[i][peaks[i]], 'x')
            except:
                pass
            plt.yticks(np.arange(1.2 * np.min(data[i]), 1.2 * np.max(data[i]), 3))

        if savepath=="":
            plt.show()

    def simple_experiment(self):
        with nidaqmx.Task() as taskin, nidaqmx.Task() as taskout:
            taskout.ao_channels.add_ao_voltage_chan("AO/ao0")
            taskout.ao_channels.add_ao_voltage_chan("AO/ao1")
            taskout.ao_channels.add_ao_voltage_chan("AO/ao2")
            taskin.ai_channels.add_ai_voltage_chan("AI/ai0",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
            taskin.ai_channels.add_ai_voltage_chan("AI/ai1",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
            taskin.ai_channels.add_ai_voltage_chan("AI/ai2",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
            # taskin.timing.cfg_samp_clk_timing(100,sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
            samps=10000
            freq=10000
            taskout.timing.cfg_samp_clk_timing(freq,samps_per_chan=samps)
            taskin.timing.cfg_samp_clk_timing(freq,samps_per_chan=samps,source='ao/SampleClock')
            # taskin.timing.ref_clk_src
            taskin.triggers.arm_start_trigger
            # taskout1.timing.cfg_samp_clk_timing(20000,sample_mode=nidaqmx.constants.AcquisitionType.FINITE,samps_per_chan=1000)
            # taskout2.timing.cfg_samp_clk_timing(1000)

            t=np.linspace(0,1,samps)
            # sawtooth = 1 + signal.sawtooth(2 * np.pi * 5 * t, width=0.5)
            # sawtooth = 3 * np.hstack((sawtooth, -1 * sawtooth))
            signal1=1+signal.square(2*np.pi*1*t+2*np.pi/2,0.005)
            signal1+=2+2*signal.square(2*np.pi*1*t+2*np.pi/2+10*0.005,0.005)
            signal1=signal.square(2 * np.pi * 150 * t, 0.01)
            # signal1 = 4*np.sin(2*np.pi*10*t)
            signal2 = 4*np.sin(2*np.pi*100*t+2*np.pi/2)
            signal3 = 4 * np.sin(2 * np.pi * 100 * t)

            stream_O = stream_writers.AnalogMultiChannelWriter(taskout.out_stream, auto_start=False)
            stream_I = stream_readers.AnalogMultiChannelReader(taskin.in_stream)
            # stream_writers.AnalogMultiChannelWriter

            # Thread(target=test_Reader.read_many_sample, args=[result,nidaqmx.constants.READ_ALL_AVAILABLE,10]).start()
            # Thread(target=test_Writer.write_many_sample, args=[np.array([signal1,signal2])]).start()
            stream_O.write_many_sample(np.array([signal1,signal2,signal3]))

            taskin.start()
            taskout.start()

            result = np.zeros((3, samps))
            stream_I.read_many_sample(result)
            # taskin.wait_until_done()
            taskout.wait_until_done()
            # taskout1.close()
            # signal1=sawtooth
            # signal2=sawtooth
            # signal=0*np.ones(100)
            # print(taskout1.write([list(signal1),list(signal2)], auto_start=True))
            # taskout1.stop()
            # print(taskout2.write(list(signal2), auto_start=True))
            # taskout2.stop()
            # taskout1.stop()
            # taskout2.write(signal2, auto_start=True)
            # result=[]
            # for s1,s2 in tqdm(zip(signal1,signal2)):
            #     # time.sleep(0.01)
            #     taskout1.write([s1,s2], auto_start=True)
            #     # taskout2.write(s2, auto_start=True)
            #     result.append(taskin.read(number_of_samples_per_channel=1))
        #
        #     # result = taskin.read(number_of_samples_per_channel=100)
        #     # taskout.write(0, auto_start=True)
        #     print(result)
        #
        # result=np.array(result)
        plt.figure()
        plt.subplot(221)
        plt.plot(result[0],label="aich1")
        plt.plot(result[1],label="aich2")
        plt.plot(result[2],label="aich3")
        plt.legend()
        plt.subplot(222)
        plt.plot(signal1,label="signal1")
        plt.plot(signal2,label="signal2")
        plt.plot(signal3,label="signal2")
        plt.legend()
        plt.subplot(212)
        plt.plot(signal1, result[0], label="hist1")
        plt.plot(signal2, result[1], label="hist2")
        plt.plot(signal1, result[2], label="hist2")
        plt.axhline(0, linewidth=.3, color='k')
        plt.axvline(0, linewidth=.3, color='k')
        plt.legend()
        plt.show()


    def nidaq_single_sim(self,X,y, inputids, outputids, eq_time,av_samps=1):

        logger.debug("Setting up inputs: {} for outputs: {} ".format(X, y))

        with nidaqmx.Task() as taskin, nidaqmx.Task() as taskout:
            for inputid, idnum in zip(inputids, range(len(inputids))):
                taskout.ao_channels.add_ao_voltage_chan("AO/ao{}".format(inputid))
            taskout.write(X, auto_start=True)

            time.sleep(eq_time)

            outvals = []
            # print("Done equilibrating, reading output values")
            logger.info("Done equilibrating, reading output values")
            for outid in outputids:
                taskin.ai_channels.add_ai_voltage_chan("AI/ai{}".format(outid),min_val=-10.0, max_val=10.0,terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)

            result=[]
            for i in range(av_samps):
                result.append(np.array(taskin.read(number_of_samples_per_channel=1)).reshape((1,-1))[0])
                time.sleep(eq_time)
            result=np.array(result).mean(axis=0)
            outvals.append(list(np.array(result).reshape((1,-1))[0]))

            outvals[-1].append(y)
        return outvals[0]


    def nidaq_circuit_eval(self, X, y, inputids, outputids, eq_time, zero_each_step=True,av_samps=10):
        results = []

        for xx,yy in tqdm(zip(X,y)):
            results.append(self.nidaq_single_sim(X=xx,y=yy,inputids=inputids,outputids=outputids,eq_time=eq_time,av_samps=av_samps))

            if zero_each_step:
                with nidaqmx.Task() as taskout:
                    for inputid, idnum in zip(inputids, range(len(inputids))):
                        taskout.ao_channels.add_ao_voltage_chan("AO/ao{}".format(inputid))
                    taskout.write(np.zeros(len(inputids)), auto_start=True)

        return results

    def opt_func(self,args):

        control = args['control']

        mul = args['params']['multiplier']


        # data = np.array(ttables['xor3'] * 4)
        data = self.get_tt(nary=3, periods=2, op='^')
        replicate = 4
        data = np.repeat(data, replicate, axis=0)

        X = data[:, :-1]
        y = data[:, -1]

        data = self.perturb_X(X, boost=mul, var=0.0)

        for c in control:
            data = np.hstack((data, np.array([c * np.ones_like(data[:, 0])]).T))

        out = self.multi_measurement(data,
                                     outmask=4,
                                     pulse_dur=.1,
                                     samps_pulse=50,
                                     duty=.5,
                                     signal_type='square',
                                     peak_type='last')

        yp = np.array(out['result'])[:, np.array(out['peaks'])[0, :]]

        result = np.hstack((yp.T, np.array([y]).T))

        score = plott.calculate_logreg_score(result)
        print('Current score {} for control {} and multiplier {}'.format(-score,control,mul))

        if args['plot']:
            if self.current_folder_name == None:
                self.current_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.current_folder_name = os.path.join('./output/', self.current_folder_name)
                if not os.path.exists(self.current_folder_name):
                    os.makedirs(self.current_folder_name)
            plott.pca_plotter(result,title="Score {0:.2f} {1}".format(score,args['control']),savepath=os.path.join(self.current_folder_name,'pca_{}.png').format(self.file_index))

            plott.plot_line(self.optim_score,title='Optimization Score Trend',savepath=os.path.join(self.current_folder_name,'score.png'))

            np.save(os.path.join(self.current_folder_name,'result_{}.npy').format(self.file_index),result)

            # './output/20190419_2xor4/pca{}.png'.format(datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            # plott.pca_plotter(result, title="Score {0:.2f} {1}".format(score,args['control']))
            self.plot_result(results=out, replicate=replicate, savepath=os.path.join(self.current_folder_name,'wave_{}.png'.format(self.file_index)))
            self.file_index += 1

        self.optim_score.append(score)
        return -score


    def mask_data(self,data,mask=[]):
        #check if dimensions agree

        data_dim=data.shape[1]
        mask_dim=len(mask)
        if data_dim != mask_dim:
            raise ValueError('Mask and data dimensions must agree')

        masked_data={}

        for data_col, maskid in zip(data.T,mask):
            masked_data[maskid]=data_col

        return masked_data

    def unmask_data(self,masked_data):
        masked_data.keys()
        data=[]
        for key in sorted(masked_data.keys()):
            data.append(data[key])

        data=np.array(data)
        return data


def dummy_f(args):
    print(args,-np.sum(args['control']))
    return -np.sum(args['control'])


def main():
    # X=[1,2,3]
    # y=[0]
    # outvals1 = nidaq_single_sim(X,y,inputids=[0,1,2],outputids=[0,1,2],eq_time=0.1)
    # print(outvals1)
    # X = [-1, -2, -3]
    # outvals2 = nidaq_single_sim(X, y, inputids=[0,1,2], outputids=[0, 1,2], eq_time=0.1)
    # print(outvals2)
    # X=[[1,2,3],[-1,-2,-3],[7.23,7.24,-7.25]]
    measurement=Measurement()
    data = np.array(ttables['xor'] * 10)
    X = data[:, :-1]
    y = data[:, -1]
    X = measurement.perturb_X(X, boost=5.0, var=0.00)
    # n=10
    # # X=np.random.random((n,3))*3
    # X=np.ones((n,3))*5.002
    # y=np.ones(n)
    result = measurement.nidaq_circuit_eval(X,y,inputids=[0,1],outputids=[0,1,2],eq_time=0.1,av_samps=5)
    plott.plot3d(result)
    plott.pca_plotter(result)
    print(X)
    print(np.array(result))
    df=pd.DataFrame(np.array(result))
    df.to_csv('result.csv')

    # 2.9996851733847802], [2.9996851733847802], [3.0009987072145923]
    plt.figure()
    plt.subplot(211)
    plt.plot(result)
    plt.show()
    print(result)


def main2():

    measurement = Measurement()
    # data = 1 * np.array([[1., -1,.0],  # R
    #                       [-1., 1,1],  # W
    #                       [1., -1,.0],
    #                       [-1., 0,.0],
    #                       [0, 1.,1],  # R
    #                       [0, -1.,.0],  # W
    #                       [0, 1.,.5],
    #                       [0, -1.,.0]
    #                       ])  # R
    #
    #
    #
    #
    # data = np.tile(data,(3,1))

    # data = np.array(ttables['xor'] * 4)
    data = measurement.get_tt(nary=2, periods=3, op='^')
    replicate = 6
    data = np.repeat(data, replicate, axis=0)

    # np.random.shuffle(data)

    X = data[:, :-1]
    y = data[:, -1]

    data = measurement.perturb_X(X, boost=8, var=0.00)


    # 3.05163959444126, 'c2': 2.474871658179593, 'c1': 2.3949911463760887
    # data = np.hstack((data, np.array([0.39 * np.ones_like(data[:, 0])]).T))
    # data = np.hstack((data, np.array([0.39 * np.ones_like(data[:, 0])]).T))
    # data = np.hstack((data, np.array([0.39 * np.ones_like(data[:, 0])]).T))

    # data = np.hstack((data, np.array([0.47*np.ones_like(data[:,0])]).T))
    #
    # data = np.hstack((data, np.array([0.05 * np.ones_like(data[:, 0])]).T))

    out = measurement.multi_measurement(data, outmask=4, pulse_dur=.1, samps_pulse=30, duty=.5, signal_type='square', peak_type='last')

    replicate=1
    yp = np.array(out['result'])[:, np.array(out['peaks'])[0, replicate-1::replicate]]

    result = np.hstack((yp.T, np.array([y[replicate-1::replicate]]).T))

    print("Got score",plott.calculate_logreg_score(result))

    # plott.plot3d(result)
    plott.plot3d_native(result)
    plott.pca_plotter(result)

    measurement.plot_result(out,replicate)

    with open('data.json', 'w') as f:
        print("writing file")
        json.dump(out, f)

def main_analog():
    meas = Measurement()
    # data = 1 * np.array([[1., -1,.0],  # R
    #                       [-1., 1,1],  # W
    #                       [1., -1,.0],
    #                       [-1., 0,.0],
    #                       [0, 1.,1],  # R
    #                       [0, -1.,.0],  # W
    #                       [0, 1.,.5],
    #                       [0, -1.,.0]
    #                       ])  # R
    # data = measurement.get_tt(nary=2, periods=3, op='^')

    x=np.linspace(0,1,200)
    data = np.zeros((x.shape[0],3))
    data[:, 0] = 6*np.sin(2 * np.pi * x * 7- 2.2)
    data[:, 1] = 4*np.sin(2 * np.pi * x * 7 -1.5)
    data[:, 2] = 5*np.sin(2 * np.pi * x * 9 - 1)

    # data = np.repeat(data,2,axis=1)

    out = meas.analog_measurement(data, freq=50, outmask=[1,2,3,4,5,6,7], inmask=[5,6,7])
    meas.clear_all_output_chans()

    meas.plot_result(out, 1)

    meas.plot_hysteresis(results=out)

def main_analog_continuous():
    measurement = Measurement()
    # data = 1 * np.array([[1., -1,.0],  # R
    #                       [-1., 1,1],  # W
    #                       [1., -1,.0],
    #                       [-1., 0,.0],
    #                       [0, 1.,1],  # R
    #                       [0, -1.,.0],  # W
    #                       [0, 1.,.5],
    #                       [0, -1.,.0]
    #                       ])  # R
    # data = measurement.get_tt(nary=2, periods=3, op='^')

    x=np.linspace(0,1,100)
    data = np.zeros((x.shape[0],1))
    data[:, 0] = 6*np.sin(2 * np.pi * x * 4)
    # data[:, 1] = 4*np.sin(2 * np.pi * x * 7 -1.5)
    # data[:, 2] = 5*np.sin(2 * np.pi * x * 9 - 1)

    # data = np.repeat(data,2,axis=1)

    out = measurement.analog_continuous_measurement(data, freq=50,  inmask=[0], outmask=[0,1,2,3,4,5,6,7])

    # measurement.plot_result(out, 1)
    #
    # measurement.plot_hysteresis(results=out)

def main_opt():
    mul_space=[6.]
    space = {'control':[hp.uniform('c1',-8,8),hp.uniform('c2',-8,8),hp.uniform('c3',-8,8)],
             'params':{
                 'multiplier':hp.choice('m',mul_space)
                },
             'plot':True
             }

    measurement = Measurement()

    best = fmin(measurement.opt_func,space,algo=tpe.suggest,max_evals=20,verbose=True)
    # best = fmin(dummy_f, space, algo=tpe.suggest, max_evals=20, verbose=True)

    bvals={'control': [best['c1'], best['c2'], best['c3']],
     'params': {
         'multiplier': mul_space[best['m']]
     },
     'plot': True
     }

    measurement.opt_func(bvals)

    print(best)

def playground():

    meas = Measurement(peak_find_delta=5)

    # data = meas.get_tt(nary=2, periods=3, op='^')
    data = 6*np.array(
             [[1., 0],  # W1
              [-1.,0],  # E1
              [0., 1],  # W2
              [0.,-1],  # E2
              [1., 0],  # W1
              [0,  1],  # W2
                ])

    data = np.tile(data, (3, 1))

    replicate = 5
    data = np.repeat(data, replicate, axis=0)
    # np.random.shuffle(data)

    # X = data[:, :-1]
    y = data[:, -1]
    # data = meas.perturb_X(X, boost=9, var=0.00)



    # indata = meas.mask_data(data=data,mask=[3,4])



    # contdata = meas.mask_data(data=contdata, mask=[7, 8])

    # print(data)

    # 3.05, 'c2': 2.47, 'c1': 2.39
    # here we append control signal to the back of the array
    # data = np.hstack((data, np.array([0.39 * np.ones_like(data[:, 0])]).T))
    # data = np.hstack((data, np.array([-0.39 * np.ones_like(data[:, 0])]).T))

    out = meas.multi_measurement(data, pulse_dur=.05, samps_pulse=80, duty=.5, signal_type='square',
                                 peak_type='last', inmask=[0,1], contmask=[], outmask=[0,1, 2, 3, 4, 5,6,7])
    meas.clear_all_output_chans()
    replicate = 1
    yp = np.array(out['result'])[:, np.array(out['peaks'])[0, replicate - 1::replicate]]

    result = np.hstack((yp.T, np.array([y[replicate - 1::replicate]]).T))

    # print("Got score", plott.calculate_logreg_score(result))

    # plott.plot3d(result)
    plott.plot3d_native(result)
    # plott.pca_plotter(result)

    meas.plot_result(out, replicate,draw_peaks=False)



if __name__ == "__main__":
    main_analog()
    # main_analog_continuous()
    # main2()
    # main_opt()
    # playground()
    # main()
