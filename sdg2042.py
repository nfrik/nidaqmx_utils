#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import visa
import time
import binascii
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

def pulse(dev, width, period,inv=True):
    # device.write("C1:BSWV WVTP,PULSE,WIDTH,5e-3,AMP,8V,OFST,4V,FRQ,10")
    # device.write("C1:OUTP ON")
    #
    # device.write("C1:INVT OFF")
    # device.write("C1:OUTP OFF")

    dev.write("C1:BSWV WVTP,PULSE,2,PULSE,WIDTH,5E-2,AMP,2V")
    if inv:
        dev.write("C1:INVT ON")
    else:
        dev.write("C1:INVT OFF")


if __name__ == '__main__':
    """"""
    # device = visa.instrument(device_resource, timeout=5000, chunk_size=40 * 1024)
    rm = visa.ResourceManager()
    device = rm.open_resource(device_resource,open_timeout=5000)

    print(device.query('*IDN?'))

    device.write("C1:BSWV WVTP,PULSE,2,PULSE,WIDTH,5E-2,AMP,2V")
    device.write("C1:INVT OFF")

    # visa.visa_main()
    # create_wave_file()
    # send_wawe_data(device)
    # get_wave_data(device)
