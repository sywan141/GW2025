import numpy as np
import numpy as np
import multiprocessing
import os
from functools import partial  

def DCF_generate(lightcurve, index):

    times = lightcurve[:, 0].copy()
    amps = lightcurve[:, 1].copy()
    np.random.shuffle(amps)
    shuffled = np.column_stack((times, amps))
    
    np.savetxt(Parent_dir + 'shuffled_LC/arr3/' + 'arr3_%s.txt'%index, shuffled)

if __name__ == '__main__':
    Parent_dir = '/data/nas_data/wsy/GW2025/hw1/'
    filename = 'lightcurve_0.1msbin.txt'
    File_all = np.loadtxt(Parent_dir + filename)
    times = File_all[0 , :]
    amps = File_all[1 , :]

    # divide the signal into three segments
    breaking_p1 = 24.85
    breaking_p2 = 24.90
    times1 = times[times < breaking_p1]
    times2 = times[(times >= breaking_p1) & (times <= breaking_p2)]
    times3 = times[times > breaking_p2]
    amps1 = amps[times < breaking_p1]
    amps2 = amps[(times >= breaking_p1) & (times <= breaking_p2)]
    amps3 = amps[times > breaking_p2]
    arr1 = np.column_stack((times1, amps1)) # background 1
    arr2 = np.column_stack((times2, amps2)) # background + burst 
    arr3 = np.column_stack((times3, amps3)) # background 2
    
    
    pool = multiprocessing.Pool(64) 
    n_samples = 1000
    
    task_indices = range(n_samples)
    func = partial(DCF_generate, arr3)
    
    pool.map(func, task_indices)
    
    pool.close()
    pool.join()
    print("Finished !")
    