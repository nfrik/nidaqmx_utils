from PyDAQmx.DAQmxCallBack import *
from PyDAQmx import *
from numpy import zeros

# Class of the data object
# one cannot create a weakref to a list directly
# but the following works well
class MyList(list):
    pass

# list where the data are stored
data = MyList()
id_data = create_callbackdata_id(data)

def EveryNCallback_py(taskHandle, everyNsamplesEventType, nSamples, callbackData_ptr):
    callbackdata = get_callbackdata_from_id(callbackData_ptr)
    read = int32()
    data = zeros(1000)
    DAQmxReadAnalogF64(taskHandle,1000,10.0,DAQmx_Val_GroupByScanNumber,data,1000,byref(read),None)
    callbackdata.extend(data.tolist())
    print("Acquired total %d samples"%len(data))
    return 0 # The function should return an integer

# Convert the python function to a C function callback
EveryNCallback = DAQmxEveryNSamplesEventCallbackPtr(EveryNCallback_py)

DAQmxRegisterEveryNSamplesEvent(taskHandle,DAQmx_Val_Acquired_Into_Buffer,1000,0,EveryNCallback,id_data)
