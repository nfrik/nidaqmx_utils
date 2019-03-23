import nidaqmx
from nidaqmx import stream_writers, stream_readers
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import logging
from utils import plott
from tqdm import tqdm
import pandas as pd
from threading import Thread
from scipy import signal
from scipy.ndimage.interpolation import shift

ttables = {}
ttables['xor'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
ttables['xor3'] = [[-1, -1, -1, 0],[-1, -1, 1, 1],[-1, 1, -1, 0], [-1, 1, 1, 1],[1, -1, -1, 1],[1, -1, 1, 0],[1, 1, -1, 1],[1, 1, 1, 0]]
ttables['or'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
ttables['and'] = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

logger = logging.getLogger(__name__)

def perturb_X(X,boost=3,var=1):
    # Y=X.copy()
    Y=X*boost
    for idx,x in np.ndenumerate(Y):
        Y[idx]+=(np.random.rand() - 0.5) * var
#     result = np.array(list(map(lambda t: boost*t + (np.random.rand() - 0.5) * var, X)))
    return Y

def get_pulse_train(amp,samps,npulse,wpulse):

    t = np.linspace(0, 1, samps)
    signals = amp*signal.square(2 * np.pi * npulse * t, wpulse)
    return signals

def multi_measurement(bindata, outchans=2, pulse_dur=0.1, samps_pulse=1000,duty=0.5):
    binsteps,inchans=bindata.shape

    freq = round(samps_pulse/pulse_dur)

    samps = binsteps * samps_pulse
    total_time = int(np.ceil(1/freq * samps))
    print("Frequency: {}, Samples: {}, Time required: {}".format(freq,samps,total_time))

    if freq>25000:
        raise ValueError('AO frequency exceeded 25kS/s/ch')

    x_p = np.linspace(0,1,samps_pulse)
    y_p = (signal.square(2*np.pi*x_p+2*np.pi/2,duty)+1)/2

    #construct list of peak rising indxs
    peak_rise_idx=list(y_p).index(1)
    peak_fall_idx = len(y_p)-list(y_p[::-1]).index(1)-1
    peak_start_idx_lst=np.arange(peak_rise_idx, samps, samps_pulse)
    peak_end_idx_lst = np.arange(peak_fall_idx, samps, samps_pulse)


    sig=np.zeros((inchans, samps))

    for bincol, chan in zip(bindata.T,range(inchans)):
        pulses=[]
        for binval in bincol:
            pulses=np.append(pulses,binval*y_p)
        sig[chan]=pulses

    # Due to 1 step delay in the 1st channel have to apply a hack to shift data by one step
    sig[0, :] = shift(sig[0, :], -1)

    with nidaqmx.Task() as taskin, nidaqmx.Task() as taskout:
        for inchan in range(inchans):
            taskout.ao_channels.add_ao_voltage_chan("AO/ao{}".format(inchan))
        for outchan in range(outchans):
            taskin.ai_channels.add_ai_voltage_chan("AI/ai{}".format(outchan),terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)

        taskout.timing.cfg_samp_clk_timing(freq, samps_per_chan=samps)
        taskin.timing.cfg_samp_clk_timing(freq, samps_per_chan=samps, source='ao/SampleClock')

        stream_O = stream_writers.AnalogMultiChannelWriter(taskout.out_stream, auto_start=False)
        stream_I = stream_readers.AnalogMultiChannelReader(taskin.in_stream)

        #Upload signal to hardware buffer
        stream_O.write_many_sample(sig)

        taskin.start()
        taskout.start()

        result = np.zeros((outchans, samps))
        stream_I.read_many_sample(result,timeout=total_time*1.2)
        # taskin.wait_until_done()
        taskout.wait_until_done()

        plt.figure()
        plt.subplot(211)
        t=np.arange(samps)/freq
        for r, chan in zip(result,range(outchans)):
            plt.plot(t,r, label="out sig {}".format(chan))

        for idx in peak_start_idx_lst:
            plt.plot(t[idx],0,'o',color='r')

        for idx in peak_end_idx_lst:
            plt.plot(t[idx], 0, 'o', color='g')

        plt.legend()
        plt.subplot(212)
        for s,chan in zip(sig,range(inchans)):
            plt.plot(t,s, label="inp sig {}".format(chan))

        plt.legend()
        plt.show()

        return  {"peak_idx":peak_start_idx_lst,"result":result}
        #construct trains of pulses for each sample
        # for rw in bindata:


def simple_experiment():
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


def nidaq_single_sim(X,y, inputids, outputids, eq_time,av_samps=1):

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


def nidaq_circuit_eval(X, y, inputids, outputids, eq_time, zero_each_step=True,av_samps=10):
    results = []

    for xx,yy in tqdm(zip(X,y)):
        results.append(nidaq_single_sim(X=xx,y=yy,inputids=inputids,outputids=outputids,eq_time=eq_time,av_samps=av_samps))

        if zero_each_step:
            with nidaqmx.Task() as taskout:
                for inputid, idnum in zip(inputids, range(len(inputids))):
                    taskout.ao_channels.add_ao_voltage_chan("AO/ao{}".format(inputid))
                taskout.write(np.zeros(len(inputids)), auto_start=True)

    return results

def main():
    # X=[1,2,3]
    # y=[0]
    # outvals1 = nidaq_single_sim(X,y,inputids=[0,1,2],outputids=[0,1,2],eq_time=0.1)
    # print(outvals1)
    # X = [-1, -2, -3]
    # outvals2 = nidaq_single_sim(X, y, inputids=[0,1,2], outputids=[0, 1,2], eq_time=0.1)
    # print(outvals2)
    # X=[[1,2,3],[-1,-2,-3],[7.23,7.24,-7.25]]

    data = np.array(ttables['xor'] * 10)
    X = data[:, :-1]
    y = data[:, -1]
    X = perturb_X(X, boost=5.0, var=0.00)
    # n=10
    # # X=np.random.random((n,3))*3
    # X=np.ones((n,3))*5.002
    # y=np.ones(n)
    result = nidaq_circuit_eval(X,y,inputids=[0,1],outputids=[0,1,2],eq_time=0.1,av_samps=5)
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
    # simple_experiment()
    data = 5 * (2 * np.random.rand(100, 16) - 1)
    data = np.repeat(data,6,axis=0)
    multi_measurement(data,outchans=16,pulse_dur=.005,samps_pulse=50,duty=.5)

if __name__ == "__main__":
    main2()
