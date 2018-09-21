import nidaqmx
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import logging
from utils import plott
from tqdm import tqdm
from scipy import signal

ttables = {}
ttables['xor'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
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

def simple_experiment():
    with nidaqmx.Task() as taskin, nidaqmx.Task() as taskout1, nidaqmx.Task() as taskout2:
        taskout1.ao_channels.add_ao_voltage_chan("AO/ao0")
        taskout2.ao_channels.add_ao_voltage_chan("AO/ao1")
        taskin.ai_channels.add_ai_voltage_chan("AI/ai0",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
        taskin.ai_channels.add_ai_voltage_chan("AI/ai1",terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
        taskin.ai_channels.add_ai_voltage_chan("AI/ai3", terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
        # taskin.timing.cfg_samp_clk_timing(1000,sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        t=np.linspace(0,1,60)
        sawtooth = 1 + signal.sawtooth(2 * np.pi * 5 * t, width=0.5)
        sawtooth = 3 * np.hstack((sawtooth, -1 * sawtooth))
        signal1=1.3*np.sin(2*np.pi*3*t)
        signal2 = 1.3*np.sin(2 * np.pi * 3 * t)
        # signal1=sawtooth
        # signal2=sawtooth
        # signal=0*np.ones(100)
        taskout1.write(signal1, auto_start=True)
        result=[]
        for s1,s2 in zip(signal1,signal2):
            # time.sleep(0.01)
            taskout1.write(s1, auto_start=True)
            # taskout2.write(sawtooth, auto_start=True)
            result.append(taskin.read(number_of_samples_per_channel=1))

        # result = taskin.read(number_of_samples_per_channel=100)
        # taskout.write(0, auto_start=True)
        print(result)

    result=np.array(result)
    plt.figure()
    plt.subplot(221)
    plt.plot(result[:,0],label="1")
    plt.plot(result[:,1],label="2")
    plt.plot(result[:,2],label="3")
    plt.legend()
    plt.subplot(222)
    plt.plot(signal1,label="signal1")
    # plt.plot(signal2, label="signal2")
    plt.legend()
    plt.subplot(212)
    plt.plot(signal1, result[:,0], label="hist1")
    plt.plot(signal1, result[:,1], label="hist2")
    plt.plot(signal1, result[:,2], label="hist2")
    plt.axhline(0, linewidth=.3, color='k')
    plt.axvline(0, linewidth=.3, color='k')
    plt.legend()
    plt.show()


def nidaq_single_sim(X,y, inputids, outputids, eq_time):

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
        result = taskin.read(number_of_samples_per_channel=1)
        outvals.append(list(np.array(result).reshape((1,-1))[0]))

        outvals[-1].append(y)
    return outvals[0]


def nidaq_circuit_eval(X, y, inputids, outputids, eq_time, zero_each_step=True):
    results = []

    for xx,yy in tqdm(zip(X,y)):
        results.append(nidaq_single_sim(X=xx,y=yy,inputids=inputids,outputids=outputids,eq_time=eq_time))

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

    data = np.array(ttables['xor'] * 20)
    X = data[:, :-1]
    y = data[:, -1]
    X = perturb_X(X, boost=10., var=0.00)
    # n=10
    # # X=np.random.random((n,3))*3
    # X=np.ones((n,3))*5.002
    # y=np.ones(n)
    result = nidaq_circuit_eval(X,y,inputids=[0,1],outputids=[0,1,2],eq_time=0.05)
    plott.plot3d(result)
    print(X)
    print(np.array(result))
    # 2.9996851733847802], [2.9996851733847802], [3.0009987072145923]
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(outvals1)
    # plt.show()
    # print(outvals2)

def main2():
    simple_experiment()

if __name__ == "__main__":
    main()
