import pprint
import nidaqmx
import matplotlib.pyplot as plt
import pandas as pd

pp = pprint.PrettyPrinter(indent=4)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("AI/ai0:1", terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)

    # print('1 Channel 1 Sample Read Raw: ')
    # # data = task.read_raw()
    #
    # data = task.read(number_of_samples_per_channel=5000)

    # plt.figure()
    # plt.plot(data[0])
    # plt.plot(data[1])
    # plt.show()


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

    plt.figure()
    plt.plot(x)
    plt.plot(y)
    plt.show()
