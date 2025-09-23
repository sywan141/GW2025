import numpy as np
import os
import multiprocessing

Parent_dir = '/data/nas_data/wsy/GW2025/hw1/'

shuffled_dir = 'shuffled_LC/arr3/'
shuffled_dir2 = 'shuffled_LC/arr2/'

result_file = "arr3arr2_dcf_results.txt"
taos = np.linspace(-1, 1, 201)  
n_bins = len(taos) - 1  
n_samples = 1000
bin_width = 1e-4


def DCF_cal(arr_a, arr_b, taos, bin_width):

    if arr_a.ndim != 2 or arr_b.ndim != 2:
        raise ValueError("输入数组必须是二维数组 (形状为[N,2]和[M,2])")
    

    t_a, amp_a = arr_a[:, 0], arr_a[:, 1]
    t_b, amp_b = arr_b[:, 0], arr_b[:, 1]
    

    ma, mb = np.mean(amp_a), np.mean(amp_b)
    sa, sb = np.std(amp_a, ddof=1), np.std(amp_b, ddof=1)
    
    dt = t_a[:, np.newaxis] - t_b  # broadcast

    UDCF = ( (amp_a - ma)[:, np.newaxis] * (amp_b - mb) ) / (sa * sb)
    
    dt_flat = dt.ravel()
    UDCF_flat = UDCF.ravel()
    
    DCF = np.zeros(n_bins)
    DCF_err = np.zeros(n_bins)
    M_counts = np.zeros(n_bins, dtype=int)  
    #t_mid = 0.5 * (taos[1:] + taos[:-1]) 
    
    for k in range(n_bins):
        tao_min = taos[k]
        tao_max = tao_min + bin_width
        zero_lag_bin = 0.01 * bin_width
        
        if abs(tao_min) < 1e-9:  # 0 time lag
            tao_max = tao_min + zero_lag_bin # use a smaller time width
        else:
            tao_max = tao_min + bin_width 
            
        mask = (dt_flat >= tao_min) & (dt_flat < tao_max)
        M_counts[k] = np.sum(mask)  

        
        UDCF_bin = UDCF_flat[mask]
        DCF[k] = np.sum(UDCF_bin) / M_counts[k]
        DCF_err[k] = np.sqrt(np.sum((UDCF_bin-DCF[k])**2))/(M_counts[k]-1)
        
    return DCF


def process_k(k):
    # 假设文件名格式：shuffled_0.txt 到 shuffled_999.txt
    filename = f"arr3_{k}.txt"
    filename2 = f"arr2_{k}.txt"
    arr3 = np.loadtxt(os.path.join(Parent_dir + shuffled_dir, filename))
    arr2 = np.loadtxt(os.path.join(Parent_dir + shuffled_dir2, filename2))
    return DCF_cal(arr3, arr2, taos, bin_width)



if __name__ == "__main__":
    
    with multiprocessing.Pool(64) as pool:
        results = pool.map(process_k, range(n_samples))  # 用k控制每个样本
    

    np.savetxt(
        Parent_dir + 'Signi/' + result_file,
        np.array(results),
        fmt="%.6f"
    )
    
    print(f"Results saved !")
