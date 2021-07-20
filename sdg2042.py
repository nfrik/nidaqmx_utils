#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('QT5Agg')
import visa
import time
import binascii
import time
import data_logger as dl
from threading import Thread
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
#USB resource of Device
device_resource = "USB0::0xF4EC::0xEE38::SDG2XCAQ2R2647::INSTR"


#Little endian, 16-bit 2's complement
wave_points = [0x0010, 0x0020, 0x0030, 0x0040, 0x0050, 0x0060, 0x0070, 0xff7f]
def create_wave_file():
    """create a file"""
    f = open("wave1.bin", "wb")
    for a in wave_points:
        b = hex(a)
        b = b[2:]
        len_b = len(b)
        if (0 == len_b):
            b = '0000'
        elif (1 == len_b):
            b = '000' + b
        elif (2 == len_b):
            b = '00' + b
        elif (3 == len_b):
            b = '0' + b
        c = binascii.a2b_hex(b.encode("ascii", "ignore"))  # Hexadecimal integer to ASCii encoded string
        f.write(c)
    f.close()


def send_wawe_data(dev):
    """send wave1.bin to the device"""
    f = open("wave1.bin", "rb")  # wave1.bin is the waveform to be sent
    data = f.read()
    print 'write bytes:', len(data)
    # dev.write("C1:WVDTWVNM, wave1, FREQ, 2000.0, AMPL, 4.0, OFST, 0.0, PHASE, 0.0, WAVEDATA, % s" % (data))
    dev.write_ascii_values("C1:WVDTWVNM, wave1, FREQ, 2000.0, AMPL, 4.0, OFST, 0.0, PHASE, 0.0, WAVEDATA, my_new_wave",[1,2,3,4,5,6])
    # "X" series (SDG1000X/SDG2000X/SDG6000X/X-E)
    # dev.write("C1:ARWV NAME,wave1")
    dev.write_ascii_values("C1:ARWV NAME,wave1")
    f.close()


def get_wave_data(dev):
    """get wave from the devide"""
    f = open("wave2.bin", "wb")  # save the waveform as wave2.bin
    dev.write("WVDT? user,wave1")  # "X" series (SDG1000X/SDG2000X/SDG6000X/X-E)
    time.sleep(1)
    data = dev.read()
    data_pos = data.find("WAVEDATA,") + len("WAVEDATA,")
    print data[0:data_pos]
    wave_data = data[data_pos:]
    print 'read bytes:', len(wave_data)
    f.write(wave_data)
    f.close()

def pulse(dev, width, amp, offset, freq, hold,inv=True):
    dev.write("C1:OUTP OFF")
    dev.write("C1:BSWV WVTP,PULSE,WIDTH,{:.2E},AMP,{},OFST,{},FRQ,{:.2E}".format(width,amp,offset,freq))
    if inv:
        dev.write("C1:INVT ON")
    else:
        dev.write("C1:INVT OFF")
    dev.write("C1:OUTP ON")
    time.sleep(hold)
    dev.write("C1:OUTP OFF")

def basic_wave(dev,wave, width, amp, offset, freq, hold,dly=0,inv=True,chan='C1'):
    dev.write("{}:OUTP OFF".format(chan))
    dev.write("{}:OUTP LOAD, HZ".format(chan))
    dev.write("{}:BSWV WVTP,{},WIDTH,{:.2E},AMP,{},OFST,{},FRQ,{:.2E},DLY,{:.2E}".format(chan,wave,width,amp,offset,freq,dly))
    # dev.write("{}:BSWV WVTP,{},WIDTH,{:.2E},AMP,{},OFST,{},FRQ,{:.2E}".format(chan, wave, width, amp, offset, freq))
    if inv:
        dev.write("{}:INVT ON".format(chan))
    else:
        dev.write("{}:INVT OFF".format(chan))
    dev.write("{}:OUTP ON".format(chan))

    time.sleep(hold)
    dev.write("{}:BSWV OFST,0,AMP,0")
    dev.write("{}:OUTP OFF".format(chan))
    dev.write("{}:INVT OFF".format(chan))

def basic_wave_dual_ch(dev,wave1, width1, amp1, offset1, freq1, dly1,  \
                       wave2, width2, amp2, offset2, freq2, dly2, hold, inv1=True, inv2=True):
    dev.write("C1:OUTP OFF")
    dev.write("C2:OUTP OFF")

    dev.write("C1:BSWV WVTP,{},WIDTH,{:.2E},AMP,{},OFST,{},FRQ,{:.2E},DLY,{:.2E}".format(wave1,width1,amp1,offset1,freq1,dly1))
    if inv1:
        dev.write("C1:INVT ON")
    else:
        dev.write("C1:INVT OFF")

    dev.write("C2:BSWV WVTP,{},WIDTH,{:.2E},AMP,{},OFST,{},FRQ,{:.2E},DLY,{:.2E}".format(wave2, width2, amp2, offset2, freq2, dly2))
    if inv1:
        dev.write("C2:INVT ON")
    else:
        dev.write("C2:INVT OFF")

    dev.write("C1:OUTP ON")
    dev.write("C2:OUTP ON")
    time.sleep(hold)
    dev.write("C1:OUTP OFF")
    dev.write("C2:OUTP OFF")
    dev.write("C1:INVT OFF")
    dev.write("C2:INVT OFF")

def npulses(dev, width, amp, offset, freq, N,inv=True):
    dev.write("C1:OUTP OFF")
    dev.write("C1:BSWV WVTP,PULSE,WIDTH,{:.2E},AMP,{},OFST,{},FRQ,{:.2E}".format(width,amp,offset,freq))
    if inv:
        dev.write("C1:INVT ON")
    else:
        dev.write("C1:INVT OFF")
    dev.write("C1:OUTP ON")
    hold=1./freq*N
    time.sleep(hold)
    dev.write("C1:OUTP OFF")
    dev.write("C1:INVT OFF")

def set_dc(dev,offset,hold):
    dev.write("C1:BSWV WVTP,DC,OFST,{}".format(offset))
    dev.write("C1:OUTP ON")
    time.sleep(hold)
    dev.write("C1:OUTP OFF")


waves={'sin':'SINE', 'square':'SQUARE', 'ramp':'RAMP', 'pulse':'PULSE', 'noise':'NOISE'}

if __name__ == '__main__':
    """"""
    # device = visa.instrument(device_resource, timeout=5000, chunk_size=40 * 1024)
    rm = visa.ResourceManager()
    device = rm.open_resource(device_resource,open_timeout=4000)

    print(device.query('*IDN?'))

    # dev.write("C1:INVT OFF")
    # dev.write("C1:OUTP ON")
    # # dev.write("C1:BSWV WVTP,PULSE,WIDTH,0.003,AMP,5,OFST,0,FRQ,5")
    # dev.write("C1:BSWV WVTP,PULSE,WIDTH,{:.2E},AMP,{},OFST,{},FRQ,{:.2E}".format(0.1, 5, 2.5, 0.6))

    # npulses(device,width=0.03,amp=5,offset=2.5,freq=10,N=10,inv=False)
    # npulses(device, width=0.002, amp=5, offset=2.5, freq=50, N=50, inv=False)

    # dl.measure_for_time(1000,10)

    srate=4000
    Thread(target=dl.measure_for_time,args=[srate,5]).start()

    # basic_wave(device, 'RAMP', width=0.001, amp=4, offset=2.0, freq=5, hold=2, inv=False, chan='C1', dly=0)
    # basic_wave(device, 'RAMP', width=0.001, amp=4, offset=2.0, freq=5, hold=10, inv=True, chan='C1', dly=0)

    # basic_wave(device, 'PULSE', width=0.1, amp=5., offset=2.5, freq=5, hold=5, inv=True, chan='C1', dly=0)
    # basic_wave(device, 'PULSE', width=0.1, amp=5., offset=2.5, freq=5, hold=5, inv=False, chan='C1', dly=0)
    # basic_wave(device, 'PULSE', width=0.01, amp=4., offset=2., freq=2, hold=5,  inv=False, chan='C1', dly=0)
    # basic_wave(device, 'PULSE', width=0.01, amp=4., offset=2., freq=25, hold=1, inv=False, chan='C1', dly=0)
    # basic_wave(device, 'PULSE', width=0.01, amp=4., offset=2., freq=1, hold=10, inv=False, chan='C1', dly=0)
    # basic_wave(device, 'PULSE', width=0.01, amp=4., offset=2., freq=10, hold=2, inv=False, chan='C1', dly=0)

    basic_wave(device, 'PULSE', width=0.01, amp=2., offset=1, freq=20, hold=1, inv=False, chan='C1', dly=0)
    basic_wave(device, 'PULSE', width=0.01, amp=2., offset=1, freq=30, hold=1, inv=True, chan='C1', dly=0)
    basic_wave(device, 'PULSE', width=0.01, amp=2., offset=1, freq=40, hold=1, inv=False, chan='C1', dly=0)
    basic_wave(device, 'PULSE', width=0.01, amp=2., offset=1, freq=50, hold=1, inv=True, chan='C1', dly=0)

    # #Wei Lu's Protocol
    # basic_wave(device, 'PULSE', width=0.001, amp=2, offset=1, freq=40, hold=10, inv=False, chan='C1', dly=0)
    # basic_wave(device, 'PULSE', width=0.001, amp=2, offset=1, freq=10, hold=10, inv=False, chan='C1', dly=0)
    # basic_wave(device, 'PULSE', width=0.001, amp=2, offset=1, freq=2, hold=10, inv=False, chan='C1', dly=0)
    # basic_wave(device, 'PULSE', width=0.001, amp=2, offset=1, freq=25, hold=10, inv=False, chan='C1', dly=0)
    # basic_wave(device, 'PULSE', width=0.001, amp=2, offset=1, freq=1, hold=10, inv=False, chan='C1', dly=0)
    # basic_wave(device, 'PULSE', width=0.001, amp=2, offset=1, freq=10, hold=10, inv=False, chan='C1', dly=0)

    # basic_wave(device, 'SQUARE', width=0.001, amp=4, offset=2.0, freq=5, hold=10, inv=True, chan='C1', dly=0)
    # basic_wave(device, 'SQUARE', width=0.001, amp=4, offset=2.0, freq=5, hold=10, inv=False, chan='C1', dly=0)
    # basic_wave(device, 'RAMP', width=0.001, amp=4, offset=2.0, freq=5, hold=10, inv=True, chan='C1', dly=0)
    # time.sleep(10)
    # basic_wave(device, 'RAMP', width=0.001, amp=4, offset=2.0, freq=5, hold=10, inv=False, chan='C1', dly=0)
    # basic_wave(device, 'PULSE', width=0.001, amp=1, offset=0.5, freq=0.5, hold=200, inv=True, chan='C1', dly=0)
    # basic_wave(device, 'RAMP', width=0.001, amp=4, offset=2.0, freq=5, hold=8, inv=False, chan='C1', dly=0)

    # basic_wave(device, 'RAMP', width=0.001, amp=4, offset=2.0, freq=5, hold=1, inv=True,chan='C1',dly=0)
    # basic_wave(device, 'RAMP', width=0.001, amp=4., offset=2., freq=5, hold=1, inv=False,chan='C1',dly=0)
    # time.sleep(1)
    # basic_wave(device, 'RAMP', width=0.001, amp=4, offset=2.0, freq=5, hold=2, inv=True, chan='C1', dly=0)
    # basic_wave(device, 'RAMP', width=0.001, amp=4., offset=2., freq=5, hold=2, inv=False, chan='C1', dly=0)
    # # time.sleep(2)
    # basic_wave(device, 'RAMP', width=0.001, amp=4, offset=2.0, freq=5, hold=4, inv=True, chan='C1', dly=0)
    # basic_wave(device, 'RAMP', width=0.001, amp=4., offset=2.0, freq=5, hold=4, inv=False, chan='C1', dly=0)
    # time.sleep(4)
    # basic_wave(device, 'RAMP', width=0.001, amp=4, offset=2.0, freq=5, hold=8, inv=True, chan='C1', dly=0)
    # basic_wave(device, 'RAMP', width=0.001, amp=4., offset=2., freq=5, hold=8, inv=False, chan='C1', dly=0)
    # time.sleep(4)
    # basic_wave(device, 'RAMP', width=0.001, amp=4, offset=2.0, freq=5, hold=2, inv=True, chan='C1', dly=0)
    # basic_wave(device, 'RAMP', width=0.001, amp=4., offset=2., freq=5, hold=2, inv=False, chan='C1', dly=0)
    # # time.sleep(8)
    # basic_wave(device, 'RAMP', width=0.001, amp=4, offset=0.0, freq=5, hold=4, inv=True, chan='C1', dly=0)
    # basic_wave(device, 'PULSE', width=0.5, amp=5, offset=2.5, freq=0.0001, hold=3, inv=True,chan='C1',dly=0)

    # basic_wave(device, 'PULSE', width=0.1, amp=2, offset=1., freq=100, hold=10.5, inv=False)
    # basic_wave(device, 'PULSE', width=0.0015, amp=2, offset=1, freq=100, hold=0.5, inv=True)
    # basic_wave(device, 'PULSE', width=0.003, amp=4, offset=2, freq=100, hold=0.5, inv=False)
    # basic_wave(device, 'SINE', width=0.003, amp=4, offset=0, freq=10, hold=1, inv=False)
    # basic_wave(device, 'SINE', width=0.003, amp=4, offset=0, freq=20, hold=1, inv=False)
    # basic_wave(device, 'SINE', width=0.003, amp=4, offset=0, freq=40, hold=1, inv=False)
    # basic_wave(device, 'SINE', width=0.003, amp=4, offset=0, freq=80, hold=1, inv=False)
    # device.write("C1:INVT OFF")



    import pandas as pd
    import numpy as np
    time.sleep(13)
    fname = './log.csv'

    f, axarr = plt.subplots(2, figsize=(8, 5))
    df = pd.read_csv(fname, skiprows=1, header=None)
    y=df[0].as_matrix()
    # y=y/1e5*1e6
    x=df[1].as_matrix()
    lth = len(x)
    t = np.arange(lth)/srate
    axarr[0].plot(t, x)
    axarr[0].set_title('input')
    axarr[1].plot(t, y)
    axarr[1].set_title('output')
    plt.show()

    # plt.figure(figsize=(10,5))

    # visa.visa_main()
    # create_wave_file()
    # send_wawe_data(device)
    # get_wave_data(device)
