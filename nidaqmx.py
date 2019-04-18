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

def bool_tt(ary=3,op='^'):
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

def get_tt(op='^',nary=3,periods=6,boost=1,var=0.0):
    data=np.array(bool_tt(ary=nary,op=op)*periods)
    # X = data[:,:-1]
    # y = data[:,-1]
    return data

def get_pulse_train(amp,samps,npulse,wpulse):

    t = np.linspace(0, 1, samps)
    signals = amp*signal.square(2 * np.pi * npulse * t, wpulse)
    return signals


def filter_sharps(data, peakstarts, peakends):
    # points = np.array([[a, b] for a, b in zip(peakstarts, peakends)]).ravel()
    # pointmap = np.ones_like(data)
    # for point in points:
    #     pointmap[:,point] = 0
    #     pointmap[:,point - 1] = 0
    #     pointmap[:,point + 1] = 0

    # return np.multiply(data, pointmap)

    for p_s in peakstarts:
        try:
            data[:,p_s]=data[:,p_s-2]
            data[:, p_s-1] = data[:, p_s - 2]
            data[:, p_s+1] = data[:, p_s - 2]
        except:
            pass

    for p_s in peakends:
        try:
            data[:,p_s] = data[:,p_s+2]
            data[:, p_s-1] = data[:, p_s + 2]
            data[:, p_s+1] = data[:, p_s + 2]
        except:
            pass

    return data

def remove_baseline(data,peakstarts,peakends):

    pre_trend = data[:,peakstarts - 2]
    pos_trend = data[:,peakends - 2]

    trend = np.hstack((pre_trend,pos_trend))

    trend = np.mean(trend,axis=1)

    data = np.subtract(data, trend.reshape((-1, 1)))

    return data

def multi_measurement(bindata, outchans=2, pulse_dur=0.1, samps_pulse=1000,duty=0.5,signal_type='triangle',peak_type='max'):
    binsteps,inchans=bindata.shape

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


        t=np.arange(samps)/freq

        print("Looking for peaks")

        result = filter_sharps(result, peak_start_idx_lst, peak_end_idx_lst)

        result = remove_baseline(result,peak_start_idx_lst, peak_end_idx_lst)

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

        return  {"peak_start_idx":peak_start_idx_lst.tolist(),"peak_end_idx":peak_end_idx_lst.tolist(),"peaks":np.array(real_peaks).tolist(),"result":result.tolist(),"signal":sig.tolist(),"time":t.tolist()}
        #construct trains of pulses for each sample


def plot_result(results,replicate):
    peak_start_idx_lst = results["peak_start_idx"]
    peak_end_idx_lst = results["peak_end_idx"]
    peaks = np.array(results["peaks"])
    peaks = peaks[:,replicate-1::replicate]
    data = np.array(results["result"])
    sig = np.array(results["signal"])
    t = np.array(results['time'])

    a = plt.figure()
    plt.subplot(211)

    for r, chan in zip(data, range(len(data))):
        # if chan != 1:
        plt.plot(t, r, label="out sig {}".format(chan))
        plt.plot(t[peaks[chan]],r[peaks[chan]], 'x', label="out sig {}".format(chan))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    plt.subplot(212)
    for s, chan in zip(sig, range(len(data))):
        plt.plot(t, s, label="inp sig {}".format(chan))

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    # plt.show()
    a.show()

    f = plt.figure()

    plt.subplots_adjust(hspace=0.001)

    tf = str(len(data))

    for s, i in zip(data, range(len(data))):
        ax1 = plt.subplot(int(tf + '1' + str(i + 1)))
        ax1.plot(t, data[i])
        plt.plot(t[peaks[i]], data[i][peaks[i]], 'x')
        plt.yticks(np.arange(1.2 * np.min(data[i]), 1.2 * np.max(data[i]), 3))

    plt.show()

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
    # data = 5 * (2 * np.random.rand(10, 16) - 1)
    # data=10*np.array([[1,0,0],
    #                   [-1,0,0],
    #                   [1,0,0],
    #                   [0,1,0],
    #                   [0,-1,0],
    #                   [0, 1, 0]])
    # data = np.repeat(data,[2,10,2,10],axis=0)

    # data = 10 * np.array([[-1, 0],#R
    #                       [1,0],#W
    #                       [-1, 0],#R
    #                       [0, -1],#R
    #                       [0,1],#W
    #                       [0, -1]])#R


    replicate=1
    data = 1 * np.array([[1., -1,.0],  # R
                          [-1., 1,1],  # W
                          [1., -1,.0],
                          [-1., 0,.0],
                          [0, 1.,1],  # R
                          [0, -1.,.0],  # W
                          [0, 1.,.5],
                          [0, -1.,.0]
                          ])  # R


    # data = np.repeat(data, replicate, axis=0)

    data = np.tile(data,(3,1))

    # data = np.array(ttables['xor'] * 4)
    data = get_tt(nary=2, periods=10, op='^')

    np.random.shuffle(data)

    X = data[:, :-1]
    y = data[:, -1]

    data = perturb_X(X, boost=8.0, var=0.00)


    # data = np.hstack((data, np.array([-1.46 * np.ones_like(data[:, 0])]).T))
    #
    # data = np.hstack((data,np.array([-0.848*np.ones_like(data[:,0])]).T))
    #
    # data = np.hstack((data, np.array([-7.65 * np.ones_like(data[:, 0])]).T))

    out = multi_measurement(data,outchans=4,pulse_dur=.1,samps_pulse=100,duty=.5,signal_type='square',peak_type='last')

    yp = np.array(out['result'])[:, np.array(out['peaks'])[0, :]]

    result = np.hstack((yp.T, np.array([y]).T))

    print("Got score",plott.calculate_logreg_score(result))

    # plott.plot3d(result)
    plott.pca_plotter(result)

    plot_result(out,replicate)

    with open('data.json', 'w') as f:
        print("writing file")
        json.dump(out, f)


def opt_func(args):

    # c1 = args

    control = args['control']

    mul = args['params']['multiplier']


    # data = np.array(ttables['xor3'] * 4)
    data = get_tt(nary=3, periods=2, op='^')

    X = data[:, :-1]
    y = data[:, -1]

    data = perturb_X(X, boost=mul, var=0.0)

    for c in control:
        data = np.hstack((data, np.array([c * np.ones_like(data[:, 0])]).T))

    out = multi_measurement(data, outchans=3, pulse_dur=.1, samps_pulse=100, duty=.5, signal_type='triang',
                            peak_type='max')

    yp = np.array(out['result'])[:, np.array(out['peaks'])[0, :]]

    result = np.hstack((yp.T, np.array([y]).T))

    score = plott.calculate_logreg_score(result)

    if args['plot']:
        plott.pca_plotter(result,title="Score {}".format(score),savepath='./output/{}.png'.format(datetime.now().strftime("%Y%m%d-%H%M%S-%f")))

    return -score

def dummy_f(args):
    print(args)
    return 0

def main_opt():
    space = {'control':[hp.uniform('c1',-8,8),hp.uniform('c2',-8,8),hp.uniform('c3',-8,8)],
             'params':{
                 'multiplier':hp.choice('m',[1.,2.,3.,4.,5.,6.,7.,8.])
                },
             'plot':True
             }

    # best = fmin(opt_func,space,algo=tpe.suggest,max_evals=30,verbose=True)
    best = fmin(opt_func, space, algo=tpe.suggest, max_evals=2, verbose=True)

    bvals={'control': [best['c1'], best['c2'], best['c3']],
     'params': {
         'multiplier': best['m']
     },
     'plot': True
     }

    opt_func(bvals)

    print(best)

if __name__ == "__main__":
    # main2()
    main_opt()