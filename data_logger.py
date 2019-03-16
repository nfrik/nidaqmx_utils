import matplotlib
matplotlib.use('QT5Agg')
import pprint
import nidaqmx
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import pandas as pd
import numpy as np
import time

pp = pprint.PrettyPrinter(indent=4)

def measure_for_time(srate=1000,duration=20):
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("AI/ai0:1", terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)

        task.timing.cfg_samp_clk_timing(srate, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        # Python 2.X does not have nonlocal keyword.
        non_local_var = {'samples': []}

        def callback(task_handle, every_n_samples_event_type,
                     number_of_samples, callback_data):
            print('Every N Samples callback invoked.')

            samples = task.read(number_of_samples_per_channel=int(srate/2))
            non_local_var['samples'].extend([samples])

            return 0

        task.register_every_n_samples_acquired_into_buffer_event(int(srate/2), callback)

        task.start()

        # val = raw_input('Running task. Press Enter to stop and see number of accumulated samples.')
        time.sleep(duration)

        print(len(non_local_var['samples']))

        with open('log.csv', 'w') as f:
            f.write("%s\n" % "{}, {}".format("ch0", "ch1"))
            for record in non_local_var['samples']:
                for ch0, ch1 in zip(record[0], record[1]):
                    f.write("%s\n" % "{}, {}".format(ch0, ch1))

        # df = pd.read_csv('./log.csv', skiprows=1, header=None)
        # y = df[0].as_matrix()
        # x = df[1].as_matrix()
        #
        # f, axarr = plt.subplots(2, figsize=(8, 5))
        # # df = pd.read_csv(fname, skiprows=1, header=None)
        # # y = df[0].as_matrix()
        # # y=y/1e5*1e6
        # # x = df[1].as_matrix()
        # lth = len(x)
        # t = np.arange(lth)  # /srate
        # axarr[0].plot(t, x)
        # axarr[1].plot(t, y)
        # plt.show()

        # plt.figure()
        # plt.plot(x)
        # plt.plot(y)
        # plt.show()

def measure_continously():
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("AI/ai0:1", terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)

        task.timing.cfg_samp_clk_timing(1000,sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        # Python 2.X does not have nonlocal keyword.
        non_local_var = {'samples': []}

        def callback(task_handle, every_n_samples_event_type,
                     number_of_samples, callback_data):
            print('Every N Samples callback invoked.')

            samples = task.read(number_of_samples_per_channel=500)
            non_local_var['samples'].extend([samples])

            return 0

        task.register_every_n_samples_acquired_into_buffer_event(
            200, callback)

        task.start()

        val = raw_input('Running task. Press Enter to stop and see number of accumulated samples.')

        print(len(non_local_var['samples']))

        with open('log.csv', 'w') as f:
            f.write("%s\n" % "{}, {}".format("ch0", "ch1"))
            for record in non_local_var['samples']:
                for ch0,ch1 in zip(record[0],record[1]):
                    f.write("%s\n" % "{}, {}".format(ch0,ch1))

        df = pd.read_csv('./log.csv', skiprows=1, header=None)
        y = df[0].as_matrix()
        x = df[1].as_matrix()

        # plt.figure()
        # plt.plot(x)
        # plt.plot(y)
        # plt.show()

if __name__ == '__main__':
    """"""
    # measure_continously()