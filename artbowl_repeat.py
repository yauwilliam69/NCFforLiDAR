import math
import numpy as np
import imageio.v3 as iio
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate
from scipy.optimize import minimize
import time
import cv2
from scipy import ndimage
import os
import warnings
warnings.filterwarnings("ignore")

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def reflectivity_MSE_db(estimate, truth):
    # method to calculate the Mean Squared Error of the estimated reflectivity compared to the true reflectivity in dB
    row_num = len(truth)
    col_num = len(truth[0])
    mse_sum = 0
    for i in range(row_num):
        for j in range(col_num):
            mse_sum += (truth[i][j] - estimate[i][j])**2
    return 10*np.log10(mse_sum/(row_num*col_num))


def depth_RMSE(estimate, truth):
    # method to calculate the Root Mean Square Error of the estimated depth compared to the true depth in meters
    return np.mean(np.square(truth - estimate))

def cap_scale(estimate, high_cap=1.0, low_cap=0.0, offset=0.0, range=1.0):
    cutoff = np.maximum(np.minimum(estimate, high_cap), low_cap)
    return range*(cutoff-low_cap)/(high_cap - low_cap) + offset

def find_B(alpha_scene_avg, SBR): # a function to calculate B
    return eta*alpha_scene_avg*S/SBR

def generate_timestamp(dist_image, reflect_image, N, B):
    row_num = len(dist_image)
    col_num = len(dist_image[0])

    timestamps_matrix = np.empty((row_num, col_num, N)) # N pulses

    P0 = np.exp(-(eta*reflect_image*S + B)) # probability of detecting 0 photons
    P1 = 1 - P0 # probability of detecting 1 photon

    P_bgd = B/(eta*reflect_image*S + B) # given that 1 photon is detected, the probability that it is due to background
    
    for i in range(N):
        success_m = np.random.binomial(1, P1, size=(row_num, col_num)) # successfully detecting 1 photon during that single-pulse period

        bgd_m = np.random.binomial(1, P_bgd, size=(row_num, col_num)) # detecting background

        bgd_ts = np.random.uniform(low=T_r/2, high=3*T_r/2, size=(row_num, col_num)) # background timestamps, uniformly distributed over [T_r/2, 3T_r/2]
        sgl_ts = np.random.normal(loc=T_r/2 + 2*dist_image/c, scale=T_p, size=(row_num, col_num)) # signal timestamps, normal distribution with mean = T_r/2 + 2*dist_image/c, STD = T_p
        ts_m = (1-bgd_m)*sgl_ts + bgd_m*bgd_ts # combining the timestamps

        hist_m = ts_m*success_m # combining timestamps with no detections
        hist_m[hist_m==0] = np.nan # imputing 0 as np.NaN

        timestamps_matrix[:,:,i] = hist_m
    
    return timestamps_matrix

def load_csv_sbr(scene, sppp, sbr):
    if scene == "art":
        foldpath = f"art_sppp{str(int(sppp)).zfill(1)}_sbr{str(int(sbr*1000)).zfill(4)}/"
        ts_m = np.empty((555, 695, int(sppp*500)))
    else:
        foldpath = f"bowl_sppp{str(int(sppp)).zfill(1)}_sbr{str(int(sbr*1000)).zfill(4)}/"
        ts_m = np.empty((555, 626, int(sppp*500)))
    if scene == "bowl" and sppp == 2:
        for k in range(int(sppp*500)):
            framepath = f"bowl_sppp{str(int(sppp)).zfill(1)}_sbr{str(int(sbr*1000)).zfill(4)}_fr{str(k).zfill(4)}"
            ts_m[:,:,k] = np.genfromtxt(foldpath + framepath, delimiter=",")
    else:
        for k in range(int(sppp*500)):
            framepath = f"sbr{str(int(sbr*1000)).zfill(4)}_fr{str(k).zfill(4)}.csv"
            ts_m[:,:,k] = np.genfromtxt(foldpath + framepath, delimiter=",")
    return ts_m, foldpath[:-1]

def load_csv_sppp(scene, sppp, sbr):
    if scene == "art":
        foldpath = f"art_sbr{str(int(sbr*1000)).zfill(4)}_sppp{str(int(sppp*1000)).zfill(4)}/"
        ts_m = np.empty((555, 695, int(sppp*500)))
    else:
        foldpath = f"bowl_sbr{str(int(sbr*1000)).zfill(4)}_sppp{str(int(sppp*1000)).zfill(4)}/"
        ts_m = np.empty((555, 626, int(sppp*500)))

    for k in range(int(sppp*500)):
        framepath = f"sppp{str(int(sppp*1000)).zfill(4)}_fr{str(k).zfill(4)}.csv"
        ts_m[:,:,k] = np.genfromtxt(foldpath + framepath, delimiter=",")
    return ts_m, foldpath[:-1]

def load_max(scene, sppp):
    if scene == "art":
        foldpath = f"art_sbrmax_sppp{str(int(sppp*1000)).zfill(4)}/"
        ts_m = np.empty((555, 695, int(sppp*500)))
    else:
        foldpath = f"bowl_sbrmax_sppp{str(int(sppp*1000)).zfill(4)}/"
        ts_m = np.empty((555, 626, int(sppp*500)))
    for k in range(int(sppp*500)):
        framepath = f"sppp3000_fr{str(k).zfill(4)}.csv"
        ts_m[:,:,k] = np.genfromtxt(foldpath + framepath, delimiter=",")
    return ts_m, foldpath

def find_k_matrix(ts_m):
    k_matrix = np.empty((len(ts_m), len(ts_m[0])))
    for i in range(len(ts_m)):
        for j in range(len(ts_m[0])):
            k_matrix[i, j] = np.count_nonzero(~np.isnan(ts_m[i, j])) # reject the NaNs recorded
    return k_matrix

def NPC(k_m, sppp):
    N = sppp*500
    # normalized photon count
    return k_m/(N*eta*S)

def shorten_ts(ts_m, k_m):
    row_num = len(ts_m)
    col_num = len(ts_m[0])
    max_k = k_m.max()
    ts_short = np.empty((row_num, col_num, int(max_k)))
    for i in range(row_num):
        for j in range(col_num):
            valid_v = ts_m[i, j][~np.isnan(ts_m[i, j])]
            ts_short[i, j] = np.pad(valid_v, (0, int(max_k - k_m[i, j])), "constant", constant_values=np.nan)
    
    return ts_short

def compute_median(ts):
    start_time = time.time() # start timing
    ts_padded = np.pad(ts, [(1, 1), (1, 1), (0, 0)], mode='constant', constant_values=np.nan)
    row_num = len(ts_padded)
    col_num = len(ts_padded[0])
    ts_median = np.empty((row_num - 2, col_num - 2))
    ts_median.fill(np.inf)
    for i in range(1, row_num - 1):
        for j in range(1, col_num - 1):
            neighbor = np.concatenate((ts_padded[i-1][j-1:j+2], 
                                        ts_padded[i][j-1:j], 
                                        ts_padded[i][j+1:j+2], 
                                        ts_padded[i+1][j-1:j+2]))
            median = np.nanmedian(neighbor)
            if median == None:
                median = np.inf
            ts_median[i - 1, j - 1] = median
    T = time.time() - start_time # calculate the time required for convergence
    print("ROM, Computing used = ", T)
    return ts_median, T

def reject_ts(ts_m, ts_ROM, alpha_PML, sbr):
    start_time = time.time() # start timing
    N = len(ts_m[0][0])

    if scene == "art":
        B = find_B(art_alpha_avg, sbr)
    else:
        B = find_B(bowl_alpha_avg, sbr)

    return_ts_cen = np.empty_like(ts_m)

    for k in range(N):
        diff = np.abs(ts_m[:,:,k] - ts_ROM) # |t - tROM|
        criterion = 2*(T_p*2)*(B/(eta*alpha_PML*S+B)) # see note below for T_p*2
        return_ts_cen[:,:,k] = np.where(diff < criterion, ts_m[:,:,k], np.nan)
    T = time.time() - start_time # calculate the time required for convergence
    print("ROM, Censoring used = ", T)
    return return_ts_cen, T


def pen_TV_z(z, gamma_z): # same thing as total variation penalty for reflectivity
    term1 = np.pad(np.square(np.diff(z)), ((0, 0), (0, 1)), "constant", constant_values=0)
    term2 = np.pad(np.square(np.diff(z.T).T), ((0, 1), (0, 0)), "constant", constant_values=0)
    return np.sum(np.sqrt(term1 + term2)) + gamma_z

def loss_z_PML(z_m, ts_m, beta_z, gamma_z):
    row_num = len(ts_m)
    col_num = len(ts_m[0])
    N = len(ts_m[0, 0]) # number of pulses sent

    z_m = z_m.reshape((row_num, col_num))
    tot_neg = 0

    for k in range(N):
        tot_neg += np.nansum(np.square(ts_m[:,:,k] - (2/c)*z_m - T_r/2))
    tot_neg /= 2*(T_p)**2

    pen_z = pen_TV_z(z_m, gamma_z)
    value = beta_z*pen_z+tot_neg

    return value  

def compute_jac_z(z, *argv):
    # args=(ts_m, beta_z, gamma_z)
    ts_m = argv[0]
    row_num = len(ts_m)
    col_num = len(ts_m[0])
    N = len(ts_m[0, 0]) # number of pulses sent
    
    beta_z = argv[1]

    z = z.reshape((row_num, col_num))

    # compute grad(neg_log_likelihood)
    
    grad_logl = np.zeros_like(z)

    for k in range(N):
        add_term = ts_m[:,:,k] - (2/c)*z - T_r/2
        grad_logl += np.where(np.isnan(add_term), 0, add_term)
    grad_logl *= -2/(c*(T_p**2))

    # compute grad(pen(z))

    ij = np.pad(z, ((1, 1), (1, 1)), "edge")

    i_add_1 = np.roll(ij, -1, axis=0)
    i_min_1 = np.roll(ij, 1, axis=0)
    j_add_1 = np.roll(ij, -1, axis=1)
    j_min_1 = np.roll(ij, 1, axis=1)
    i_min_1_j_add_1 = np.roll(ij, (1, -1), axis=(0, 1))
    i_add_1_j_min_1 = np.roll(ij, (-1, 1), axis=(0, 1))

    term1 = np.square(i_add_1 - ij) + np.square(j_add_1 - ij)
    term1 = np.where(term1>1E-9, term1, 1E-9)

    term2 = np.square(ij - i_min_1) + np.square(i_min_1_j_add_1-i_min_1)
    term2 = np.where(term2>1E-9, term2, 1E-9)

    term3 = np.square(ij - j_min_1) + np.square(i_add_1_j_min_1 - j_min_1)
    term3 = np.where(term3>1E-9, term3, 1E-9)

    grad_TV = np.power(term1, -0.5)*(2*ij - i_add_1 - j_add_1) 
    + np.power(term2, -0.5)*(ij - i_min_1) 
    + np.power(term3, -0.5)*(ij - j_min_1)

    return (grad_logl+beta_z*grad_TV[1:-1,1:-1]).flatten()

def show_calculateRMSE(result, 
                       loss_funct_value, 
                       gradient_norm, 
                       row_num,
                       col_num,
                       foldpath, 
                       method,
                       beta_z=10000,
                       high_cap=2.5,
                       low_cap=1.5
                       ):

    image = result.x.reshape(row_num, col_num)

    fig, ax = plt.subplots(2, 2, dpi=200)
    fig.subplots_adjust(right=1, left=0, hspace=0.4)
    if scene == "art":
        test_z = art_z
    else:
        test_z = bowl_z

    fig.suptitle(f"{foldpath}, {method}, Cnvrgd = {result.success}, RMSE = {depth_RMSE(image, test_z)*100:.3f} cm")
    
    im = ax[0, 0].imshow(image, vmin=np.min(test_z), vmax = np.max(test_z), cmap=cmap, interpolation="nearest", interpolation_stage="data")
    ax[0, 0].grid()
    ax[0, 0].set_title("Estimated depth")
    
    im = ax[0, 1].imshow(test_z, vmin=np.min(test_z), vmax = np.max(test_z), cmap=cmap, interpolation="nearest", interpolation_stage="data")
    ax[0, 1].grid()
    ax[0, 1].set_title("True depth")

    cbar_ax = fig.add_axes([0.42, 0.55, 0.02, 0.35])
    fig.colorbar(im, cax=cbar_ax)

    ax[1, 0].plot(loss_funct_value)
    ax[1, 0].set_title("Objective")
    ax[1, 0].set_yscale('log')
    ax[1, 0].grid()

    ax[1, 1].plot(gradient_norm)
    ax[1, 1].set_title("Penalty Term")
    ax[1, 1].grid()

    fig.patch.set_facecolor('white')
    plt.savefig(f'{foldpath}dPMLcurve_{method}.png', bbox_inches='tight')
    plt.close()
    

def calculate_initial_guess(ts_pred):
    pred_z = (ts_pred - T_r/2)*c/2
    
    return ndimage.gaussian_filter(np.where(np.isnan(pred_z), 0.5*np.random.random() + 1.75, pred_z), sigma=4, mode="reflect") + (np.random.random(ts_pred.shape) - 0.5)*0.1

def trial_z(ts, ig, foldpath, method, gamma_z=1, beta_z=10000, ftol=1E-3):
    
    def log_loss_value(intermediate_result):
        nonlocal loss_funct_value
        nonlocal gradient_norm
        loss_funct_value.append(intermediate_result.fun)
        gradient_norm.append(pen_TV_z(intermediate_result.x.reshape((row_num, col_num)), gamma_z))

    row_num = len(ts)
    col_num = len(ts[0])

    start_time = time.time()
    loss_funct_value = []
    gradient_norm = []

    result = minimize(loss_z_PML,
                    ig.flatten(), 
                    args=(ts, beta_z, gamma_z), 
                    method='L-BFGS-B', 
                    tol=ftol,
                    jac=compute_jac_z,
                    callback=log_loss_value, 
                    options={'eps':1E-5, 'maxiter':1E5})
    T = time.time() - start_time

    show_calculateRMSE(result, 
                       loss_funct_value, 
                       gradient_norm, 
                       row_num,
                       col_num,
                       foldpath, 
                       method,
                       beta_z=beta_z,
                       high_cap=2.5,
                       low_cap=1.5
                       )
    return result

def mode1(x):
    values, counts = np.unique(x[~np.isnan(x)], return_counts=True, equal_nan=True)
    if len(values) == 0:
        return np.nan
    m = np.flatnonzero(counts == np.max(counts))
    if len(m) > 1:
        m = np.random.choice(m, 1)
    return values[m]

vmode1 = np.vectorize(mode1, signature='(n)->()')

def compute_mode(ts):
    start_time = time.time()
    ts_padded = np.pad(ts, [(1, 1), (1, 1), (0, 0)], mode='constant', constant_values=np.nan)

    i_min_1_j_min_1 = np.roll(ts_padded, (1, 1), axis=(0, 1))
    i_min_1 = np.roll(ts_padded, (1, 0), axis=(0, 1))
    i_min_1_j_add_1 = np.roll(ts_padded, (1, -1), axis=(0, 1))
    j_min_1 = np.roll(ts_padded, (0, 1), axis=(0, 1))
    # iij = np.roll(arr, (1, 0), axis=(0, 1))
    j_add_1 = np.roll(ts_padded, (0, -1), axis=(0, 1))
    i_add_1_j_min_1 = np.roll(ts_padded, (-1, 1), axis=(0, 1))
    i_add_1 = np.roll(ts_padded, (-1, 0), axis=(0, 1))
    i_add_1_j_add_1 = np.roll(ts_padded, (-1, -1), axis=(0, 1))
    concated = np.round(np.concatenate((i_min_1_j_min_1, i_min_1, i_min_1_j_add_1, 
                                    j_min_1, ts_padded, j_add_1, 
                                    i_add_1_j_min_1, i_add_1, i_add_1_j_add_1), axis=-1), decimals=11)


    ts_mode = vmode1(concated)

    T = time.time() - start_time # calculate the time required for convergence
    print("Mode, Computing used = ", T)
    
    return ts_mode[1:-1, 1:-1], T

def reject_ts_mode(ts_m, ts_ROM, alpha_PML, sbr):
    N = len(ts_m[0][0])
    B = find_B(alpha_PML, sbr)
    start_time = time.time() # start timing
    

    return_ts_cen = np.empty_like(ts_m)
    # return_ts_cen.fill(np.nan)

    for k in range(N):
        diff = np.abs(ts_m[:,:,k] - ts_ROM) # |t - tROM|
        criterion = 2*(T_p*2) # see note below for T_p*2
        return_ts_cen[:,:,k]  = np.where(np.logical_or(diff > criterion, np.isnan(diff)),  np.nan, ts_m[:,:,k])
    T = time.time() - start_time # calculate the time required for convergence
    print("Mode, Censoring used = ", T)
    return return_ts_cen, T

def find_superpixel_size(sppp):
    # target: 4 signal photons for -1.75 SD reflectivity
    # which is about 1/4 of average alpha
    return int(np.ceil((np.sqrt(16/sppp) - 1)/2.0))

def smallest_diff_1d(difference):

    convolved = np.convolve(difference, [1/4, 1/2, 1/4], 'valid')
    len_nonnan = np.count_nonzero(~np.isnan(convolved))
    if len_nonnan == 0 or np.nanmin(convolved) > 2*2*T_p:
        return -1
    else:
        return np.nanargmin(convolved) + 2

vsmallest_diff = np.vectorize(smallest_diff_1d, signature='(n)->()')

def compute_tightest_censor(ts, sppp):
    start_time = time.time()
    ts_padded = np.pad(ts, [(1, 1), (1, 1), (0, 1)], mode='constant', constant_values=np.nan)

    superPlength = find_superpixel_size(sppp)

    concated = np.concatenate([np.roll(ts_padded, (i, j), axis=(0, 1)) for i in range(-superPlength, superPlength+1) for j in range(-superPlength, superPlength+1)], axis=-1)
    concated_sorted = np.sort(concated)
    concated_diff = np.diff(concated_sorted, axis=-1, append=np.nan)
    ts_tightest_idx = vsmallest_diff(concated_diff)
    ts_tightest = concated_sorted[row_idx, col_idx, ts_tightest_idx][1:-1, 1:-1]

    concated = concated[1:-1, 1:-1, :]

    computeT = time.time() - start_time

    N = len(concated[0][0])

    for k in range(N):
        diff = np.abs(concated[:,:,k] - ts_tightest) # |t - tROM|
        criterion = (4*T_p) # see note below for T_p*2
        concated[:,:,k] = np.where(np.logical_or(diff > criterion, np.isnan(diff)),  np.nan, concated[:,:,k])

    T = time.time() - start_time - computeT

    print("Difference, Computering & Censoring used = ", T + computeT)
    
    return ts_tightest, concated, computeT, T

def find_vmin(data):
    return (np.nanmin(data) + np.nanmean(data))/2.0

def find_vmax(data):
    return (np.nanmax(data) + np.nanmean(data))/2.0

if __name__ == "__main__":
    mpl.rcParams['font.size'] = 12
    plt.rcParams["font.family"] = "Times New Roman"

    cmap = plt.get_cmap('winter')
    cmap.set_bad('grey',1.)

    cwcmap = plt.get_cmap('coolwarm')
    cwcmap.set_bad('grey',1.)

    rcmap = plt.get_cmap('Greens')
    rcmap.set_bad('grey',1.)

    # defining the constants
    T_p = 270E-12/2
    T_r = 100E-9
    eta = 0.35
    c=3E8
    S = 0.0114

    art_z = iio.imread('ArtnBowl/art_disp1.png')
    art_z = 1.5 + (- art_z + 255)/255
    art_z = cv2.resize(art_z, (0,0), fx=0.5, fy=0.5) 
    
    art_alpha = iio.imread('ArtnBowl/art_view1.png')
    art_alpha = rgb2gray(art_alpha)/255
    art_alpha = cv2.resize(art_alpha, (0,0), fx=0.5, fy=0.5) 

    bowl_z = iio.imread('ArtnBowl/bowl_disp1.png')
    bowl_z = 1.5 + (- bowl_z + 255)/255
    bowl_z = cv2.resize(bowl_z, (0,0), fx=0.5, fy=0.5) 
    
    bowl_alpha = iio.imread('ArtnBowl/bowl_view1.png')
    bowl_alpha = rgb2gray(bowl_alpha)/255
    bowl_alpha = cv2.resize(bowl_alpha, (0,0), fx=0.5, fy=0.5) 

    bowl_alpha_avg = np.mean(bowl_alpha)
    art_alpha_avg = np.mean(art_alpha)
    

    SBR_test_arr = [1.0]*7

    sppp_test_arr =  [0.1, 0.2, 0.35, 0.6, 1.0, 2.0, 3.5]


    for scene in ["art"]:
        
        print("#######################################################################")
        
        for sbr, sppp in zip(SBR_test_arr, sppp_test_arr):
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            computeT_ROM_arr = []
            rejectT_ROM_arr = []
            RMSE_ROM_arr = []

            computeT_mode_arr = []
            rejectT_mode_arr = []
            RMSE_mode_arr = []

            computeT_diff_arr = []
            rejectT_diff_arr = []
            
            RMSE_diff_arr = []
            
            filterT_arr = []
            RMSE_diff_fil_arr = []
            
            sbr_txt = sbr
            if sbr == 1E12:
                sbr_txt = "max"
                RMSE_oracle_arr = []

                for exp_iter in range(1, 11):
                    print("------------------------------------------------------------------")

                    print(f"Generating ts, scene = {scene}, sbr = {sbr_txt}, sppp = {sppp}, trial #{exp_iter}")
                    
                    folderpath = f"repeat/{scene}_sbrmax_sppp{(str(int(sppp*1000)).zfill(4))}_trial{exp_iter}/"
                    
                    if not os.path.exists(folderpath):
                        os.makedirs(folderpath)

                    if scene == "art":
                        B = find_B(art_alpha_avg, sbr)
                        ts = generate_timestamp(art_z, art_alpha, int(sppp*500), B)
                        scene_shape = art_alpha.shape
                        real_scene_alpha = art_alpha
                        real_scene_z = art_z
                    else:
                        B = find_B(bowl_alpha_avg, sbr)
                        ts = generate_timestamp(bowl_z, bowl_alpha, int(sppp*500), B)
                        scene_shape = bowl_alpha.shape
                        real_scene_alpha = bowl_alpha
                        real_scene_z = bowl_z
                    scene_dim = scene_shape[0]*scene_shape[1]
                    true_sig_ts = T_r/2 + 2*real_scene_z/c

                    k_m = find_k_matrix(ts)

                    ts = shorten_ts(ts, k_m)
                    print("Ts shape:", ts.shape)
            #         # np.savetxt(f"{folderpath}ts.csv", ts.flatten(), delimiter=",")

            #         image = np.nanmean(ts, axis=-1)
            #         masked_array = np.ma.array(image, mask=np.isnan(image))
            #         plt.imshow(masked_array, cmap=cmap, 
            #                     vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
            #                     interpolation="nearest", interpolation_stage="data")
                        
            #         plt.title(f"Mean Photon Arrival Time \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.grid()
            #         plt.savefig(f"{folderpath}mean_ts.png", bbox_inches="tight")
            #         plt.close()

            #         plt.imshow(k_m,
            #                     vmin=0, vmax=find_vmax(k_m), 
            #                     cmap=plt.get_cmap("gray"), interpolation="nearest", interpolation_stage="data")
            #         plt.title(f"Number of Timestamps \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                        
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.grid()
            #         plt.savefig(f"{folderpath}num_ts.png", bbox_inches="tight")
            #         plt.close()

            #         print("Starting Oracle")

            #         arr_show = np.nanmean(ts, axis=-1) - true_sig_ts
            #         masked_array = np.ma.array(arr_show, mask=np.isnan(arr_show))
            #         plt.imshow(masked_array, vmax=0.5*np.nanmax(masked_array), vmin=-0.5*np.nanmax(masked_array), cmap=cwcmap, interpolation="nearest", interpolation_stage="data")
                        
            #         plt.title(f"Abs. Err. of Detection Time, Oracle Mean, \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.grid()
            #         plt.savefig(f"{folderpath}mean_abserr_oracle.png", bbox_inches="tight")
            #         plt.close()

            #         ig_oracle = calculate_initial_guess(np.nanmean(ts, axis=-1))

            #         image = ig_oracle
            #         masked_array = np.ma.array(image, mask=np.isnan(image))
            #         plt.imshow(masked_array, cmap=cmap, 
            #                     vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
            #                     interpolation="nearest", interpolation_stage="data")
                        
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.grid()
            #         plt.title(f"Initial Guess, Oracle mean \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.savefig(f"{folderpath}ig_oracle.png", bbox_inches="tight")
            #         plt.close()

            #         print("Oracle, PML")
            #         dPMLresult_oracle = trial_z(ts, ig_oracle, folderpath, "Oracle", beta_z=0, ftol=1E-3)
            #         dPMLim_oracle = dPMLresult_oracle.x.reshape(scene_shape)
                    
            #         np.savetxt(f"{folderpath}dPML_oracle.csv", dPMLim_oracle, delimiter=",")
            #         RMSE_oracle = depth_RMSE(dPMLim_oracle, real_scene_z)
                    
            #         print(f"RMSE, oracle = {RMSE_oracle:.6f}")

            #         RMSE_oracle_arr.append(RMSE_oracle)


            #         plt.imshow(dPMLim_oracle, cmap=cmap, 
            #                     vmin=find_vmin(dPMLim_oracle), vmax=find_vmax(dPMLim_oracle),
            #                     interpolation="nearest", interpolation_stage="data")
            #         plt.title(f"PML Estimation, Oracle \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.grid()
            #         plt.savefig(f"{folderpath}dPMLim_oracle.png", bbox_inches="tight")
            #         plt.close()
                
            #         arr_show = dPMLim_oracle - real_scene_z
            #         masked_array = np.ma.array(arr_show, mask=np.isnan(arr_show))
            #         plt.imshow(masked_array, vmax=0.5*np.nanmax(masked_array), vmin=-0.5*np.nanmax(masked_array), cmap=cwcmap, interpolation="nearest", interpolation_stage="data")
            #         plt.title(f"Abs. Err., Oracle \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.grid()
            #         plt.colorbar()
            #         plt.savefig(f"{folderpath}dPMLim_abserr_oracle.png", bbox_inches="tight")
            #         plt.close()

            #         continue
            #     datapath = f"repeat/{scene}_sbrmax_sppp{(str(int(sppp*1000)).zfill(4))}_data/"
                
            #     if not os.path.exists(datapath):
            #         os.makedirs(datapath)

            #     np.savetxt(f"{datapath}RMSE_oracle.csv", RMSE_oracle_arr, delimiter=",")

                continue
            else:

                for exp_iter in range(1, 11):
                    print("------------------------------------------------------------------")

                    print(f"Generating ts, scene = {scene}, sbr = {sbr_txt}, sppp = {sppp}, trial #{exp_iter}")
                    
                    folderpath = f"repeat/{scene}_sbr{str(int(sbr*1000)).zfill(4)}_sppp{(str(int(sppp*1000)).zfill(4))}_trial{exp_iter}/"

                    
                    if not os.path.exists(folderpath):
                        os.makedirs(folderpath)

                    if scene == "art":
                        B = find_B(art_alpha_avg, sbr)
                        ts = generate_timestamp(art_z, art_alpha, int(sppp*500), B)
                        scene_shape = art_alpha.shape
                        real_scene_alpha = art_alpha
                        real_scene_z = art_z
                    else:
                        B = find_B(bowl_alpha_avg, sbr)
                        ts = generate_timestamp(bowl_z, bowl_alpha, int(sppp*500), B)
                        scene_shape = bowl_alpha.shape
                        real_scene_alpha = bowl_alpha
                        real_scene_z = bowl_z
                    scene_dim = scene_shape[0]*scene_shape[1]
                    true_sig_ts = T_r/2 + 2*real_scene_z/c

                    k_m = find_k_matrix(ts)

                    ts = shorten_ts(ts, k_m)
                    print("Ts shape:", ts.shape)
                    
            #         image = np.nanmean(ts, axis=-1)
            #         masked_array = np.ma.array(image, mask=np.isnan(image))
            #         plt.imshow(masked_array, cmap=cmap, 
            #                     vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
            #                     interpolation="nearest", interpolation_stage="data")
                        
            #         plt.title(f"Mean Photon Arrival Time \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.savefig(f"{folderpath}mean_ts.png", bbox_inches="tight")
            #         plt.close()

            #         plt.imshow(k_m,
            #                     vmin=0, vmax=find_vmax(k_m), 
            #                     cmap=plt.get_cmap("gray"), interpolation="nearest", interpolation_stage="data")
            #         plt.title(f"Number of Timestamps \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                        
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.savefig(f"{folderpath}num_ts.png", bbox_inches="tight")
            #         plt.close()

            #         # ROM

            #         print("Starting ROM")
                    
            #         ts_median, computeT_ROM = compute_median(ts)
                    
            #         image = ts_median
            #         masked_array = np.ma.array(image, mask=np.isnan(image))
            #         plt.imshow(masked_array, cmap=cmap, 
            #                     vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
            #                     interpolation="nearest", interpolation_stage="data")
                        
            #         plt.title(f"Median Timestamp \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                 left=False, 
            #                 top=False, 
            #                 right=False, 
            #                 bottom=False, 
            #                 labelleft=False, 
            #                 labeltop=False, 
            #                 labelright=False, 
            #                 labelbottom=False)
            #         plt.savefig(f"{folderpath}oraest_ROM.png", bbox_inches="tight")
            #         plt.close()

            #         ts_cen_ROM, rejectT_ROM = reject_ts(ts, ts_median, real_scene_alpha, sbr)

            #         k_cen_ROM = find_k_matrix(ts_cen_ROM)

            #         ts_cen_ROM = shorten_ts(ts_cen_ROM, k_cen_ROM)

            #         image = np.nanmean(ts_cen_ROM, axis=-1)
            #         masked_array = np.ma.array(image, mask=np.isnan(image))
            #         plt.imshow(masked_array, cmap=cmap, 
            #                     vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
            #                     interpolation="nearest", interpolation_stage="data")
                        
            #         plt.title(f"Mean Photon Arrival Time, ROM Censored \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.savefig(f"{folderpath}cen_mean_ROM.png", bbox_inches="tight")
            #         plt.close()

            #         plt.imshow(k_cen_ROM, 
            #                     vmin=0, vmax=find_vmax(k_cen_ROM),
            #                     cmap=plt.get_cmap("gray"))
            #         plt.title(f"Number of Timestamps, ROM Censored \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.savefig(f"{folderpath}cen_num_ROM.png", bbox_inches="tight")
            #         plt.close()

                    
            #         arr_show = np.nanmean(ts_cen_ROM, axis=-1) - true_sig_ts
            #         masked_array = np.ma.array(arr_show, mask=np.isnan(arr_show))
            #         plt.imshow(masked_array, vmax=0.5*np.nanmax(masked_array), vmin=-0.5*np.nanmax(masked_array), cmap=cwcmap, interpolation="nearest", interpolation_stage="data")
                        
            #         plt.title(f"Abs. Err. of Detection Time, ROM Censored, \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.savefig(f"{folderpath}cen_mean_abserr_ROM.png", bbox_inches="tight")
            #         plt.close()

            #         ig_ROM = calculate_initial_guess(ts_median)

            #         image = ig_ROM
            #         masked_array = np.ma.array(image, mask=np.isnan(image))
            #         plt.imshow(masked_array, cmap=cmap, 
            #                     vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
            #                     interpolation="nearest", interpolation_stage="data")
                        
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.title(f"Initial Guess, ROM Censored \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.savefig(f"{folderpath}ig_ROM.png", bbox_inches="tight")
            #         plt.close()

            #         print("PML, ROM")
            #         dPMLresult_ROM = trial_z(ts_cen_ROM, ig_ROM, folderpath, "ROM", beta_z=10000, ftol=1E-3)
            #         dPMLim_ROM = dPMLresult_ROM.x.reshape(scene_shape)
                    
            #         np.savetxt(f"{folderpath}dPML_ROM.csv", dPMLim_ROM, delimiter=",")
            #         RMSE_ROM = depth_RMSE(dPMLim_ROM, real_scene_z)
                    
            #         print(f"RMSE, ROM = {RMSE_ROM:.6f}")

            #         RMSE_ROM_arr.append(RMSE_ROM)


            #         plt.imshow(dPMLim_ROM, cmap=cmap, 
            #                     vmin=find_vmin(dPMLim_ROM), vmax=find_vmax(dPMLim_ROM),
            #                     interpolation="nearest", interpolation_stage="data")
            #         plt.title(f"PML Estimation, ROM + PML \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.savefig(f"{folderpath}dPMLim_ROM.png", bbox_inches="tight")
            #         plt.close()
                    
            #         arr_show = dPMLim_ROM - real_scene_z
            #         masked_array = np.ma.array(arr_show, mask=np.isnan(arr_show))
            #         plt.imshow(masked_array, vmax=0.5*np.nanmax(masked_array), vmin=-0.5*np.nanmax(masked_array), cmap=cwcmap, interpolation="nearest", interpolation_stage="data")
            #         plt.title(f"Abs. Err., ROM + PML \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.colorbar()
            #         plt.savefig(f"{folderpath}dPMLim_abserr_ROM.png", bbox_inches="tight")
            #         plt.close()

            #         # Mode

            #         print("Starting Mode")

            #         ts_mode, computeT_mode = compute_mode(ts)

            #         image = ts_mode
            #         masked_array = np.ma.array(image, mask=np.isnan(image))
            #         plt.imshow(masked_array, cmap=cmap, 
            #                     vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
            #                     interpolation="nearest", interpolation_stage="data")
                        
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.colorbar()
            #         plt.title(f"Mode Timestamp \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.savefig(f"{folderpath}oraest_mode.png", bbox_inches="tight")
            #         plt.close()

            #         cen_mode, rejectT_mode = reject_ts_mode(ts, ts_mode, real_scene_alpha, sbr)
                    
            #         k_cen_mode = find_k_matrix(cen_mode)

            #         cen_mode = shorten_ts(cen_mode, k_cen_mode)
                    
            #         image = k_cen_mode
            #         masked_array = np.ma.array(image, mask=np.isnan(image))
            #         plt.imshow(masked_array, cmap=plt.get_cmap("gray"), 
            #                     vmin=0, vmax=find_vmax(masked_array),
            #                     interpolation="nearest", interpolation_stage="data")
                        
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.title(f"Number of Detections, Mode Censored \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.savefig(f"{folderpath}cen_num_mode.png", bbox_inches="tight")
            #         plt.close()

            #         image = np.nanmean(cen_mode, axis=-1)
            #         masked_array = np.ma.array(image, mask=np.isnan(image))
            #         plt.imshow(masked_array, 
            #                     vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
            #                     interpolation='nearest', interpolation_stage="data", cmap=cmap)
                        
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.title(f"Mean Arrival Time, Mode Censored \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.savefig(f"{folderpath}cen_mean_mode.png", bbox_inches="tight")
            #         plt.close()

            #         image = np.nanmean(cen_mode, axis=-1) - true_sig_ts
            #         masked_array = np.ma.array(image, mask=np.isnan(image))
            #         plt.imshow(masked_array, vmax=0.5*np.nanmax(masked_array), vmin=-0.5*np.nanmax(masked_array), cmap=cwcmap, interpolation="nearest", interpolation_stage="data")
                        
            #         plt.title(f"Abs. Err. of Detection Time, Mode Censored, \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.savefig(f"{folderpath}cen_mean_abserr_mode.png", bbox_inches="tight")
            #         plt.close()

            #         ig_mode = calculate_initial_guess(ts_mode)

            #         image = ig_mode
            #         masked_array = np.ma.array(image, mask=np.isnan(image))
            #         plt.imshow(masked_array, cmap=cmap, 
            #                     vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
            #                     interpolation="nearest", interpolation_stage="data")
                        
            #         plt.colorbar()
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.title(f"Initial Guess, Mode Censored \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.savefig(f"{folderpath}ig_mode.png", bbox_inches="tight")
            #         plt.close()

            #         print("PML, Mode")
            #         dPMLresult_mode = trial_z(cen_mode, ig_mode, folderpath, "Mode", beta_z=10000, ftol=1E-3)

            #         dPMLim_mode = dPMLresult_mode.x.reshape(scene_shape)
                    
            #         np.savetxt(f"{folderpath}dPML_mode.csv", dPMLim_mode, delimiter=",")

            #         RMSE_mode = depth_RMSE(dPMLim_mode, real_scene_z)
            #         print(f"RMSE, Mode = {RMSE_mode:.6f}")

            #         RMSE_mode_arr.append(RMSE_mode)

            
            #         image = dPMLim_mode
            #         masked_array = np.ma.array(image, mask=np.isnan(image))
            #         plt.imshow(masked_array, cmap=cmap, 
            #                     vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
            #                     interpolation="nearest", interpolation_stage="data")
            #         plt.colorbar()
                        
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.title(f"PML Estimation, Mode Censored \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.savefig(f"{folderpath}dPMLim_mode.png", bbox_inches="tight")
            #         plt.close()

            #         image = dPMLim_mode - real_scene_z
            #         masked_array = np.ma.array(image, mask=np.isnan(image))
            #         plt.imshow(masked_array, vmax=0.5*np.nanmax(masked_array), vmin=-0.5*np.nanmax(masked_array), cmap=cwcmap, interpolation="nearest", interpolation_stage="data")
            #         plt.colorbar()
                        
            #         plt.tick_params(axis='both', 
            #                         left=False, 
            #                         top=False, 
            #                         right=False, 
            #                         bottom=False, 
            #                         labelleft=False, 
            #                         labeltop=False, 
            #                         labelright=False, 
            #                         labelbottom=False)
            #         plt.title(f"Abs. Err. of PML Estimation, Mode Censored \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
            #         plt.savefig(f"{folderpath}dPMLim_abserr_mode.png", bbox_inches="tight")
            #         plt.close()

                    print('Starting Difference')

                    col_idx = np.outer(np.ones((scene_shape[0] + 2)), np.arange(scene_shape[1] + 2)).astype(np.intp)
                    row_idx = np.outer(np.arange((scene_shape[0] + 2)), np.ones(scene_shape[1] + 2)).astype(np.intp)
                    
                    ts_diff, ts_cen_diff, computeT_diff, rejectT_diff = compute_tightest_censor(ts, sppp)
                    k_cen_diff = find_k_matrix(ts_cen_diff)
                    ts_cen_diff = shorten_ts(ts_cen_diff, k_cen_diff)

                    image = ts_diff
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array, cmap=cmap, 
                                vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
                                interpolation="nearest", interpolation_stage="data")
                    plt.colorbar()
                        
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.title(f"Difference Timestamp \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.savefig(f"{folderpath}oraest_diff.png", bbox_inches="tight")
                    plt.close()

                    image = np.nanmean(ts_cen_diff, axis=-1)
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array, cmap=cmap, 
                                vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
                                interpolation="nearest", interpolation_stage="data")
                    plt.colorbar()
                        
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.title(f"Mean Arrival Time, Difference Censored \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.savefig(f"{folderpath}cen_mean_diff.png", bbox_inches="tight")
                    plt.close()

                    image = np.nanmean(ts_cen_diff, axis=-1) - true_sig_ts
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array, vmax=0.5*np.nanmax(masked_array), vmin=-0.5*np.nanmax(masked_array), cmap=cwcmap, interpolation="nearest", interpolation_stage="data")
                        
                    plt.title(f"Abs. Err. of Detection Time, Difference Censored, \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.colorbar()
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.savefig(f"{folderpath}cen_mean_abserr_diff.png", bbox_inches="tight")
                    plt.close()

                    
                    image = k_cen_diff
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array, 
                                vmin=0, vmax=find_vmax(masked_array),
                                cmap=plt.get_cmap("gray"), interpolation="nearest", interpolation_stage="data")
                    plt.colorbar()
                        
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.title(f"Number of Detections, Difference Censored \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.savefig(f"{folderpath}cen_num_diff.png", bbox_inches="tight")
                    plt.close()

                    ig_diff = calculate_initial_guess(ts_diff)

                    image = ig_diff
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array, cmap=cmap, 
                                vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
                                interpolation="nearest", interpolation_stage="data")
                        
                    plt.colorbar()
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.title(f"Initial Guess, Difference Censored \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.savefig(f"{folderpath}ig_diff.png", bbox_inches="tight")
                    plt.close()

                    print("PML, Difference")
                    dPMLresult_diff = trial_z(ts_cen_diff, ig_diff, folderpath, "Difference", beta_z=10000, ftol=1E-3)
                    dPMLim_diff = dPMLresult_diff.x.reshape(scene_shape)

                    np.savetxt(f"{folderpath}dPML_diff.csv", dPMLim_diff, delimiter=",")

                    RMSE_diff = depth_RMSE(dPMLim_diff, real_scene_z)
                    print(f"RMSE, Difference = {RMSE_diff:.6f}")
                    RMSE_diff_arr.append(RMSE_diff)

                
                    image = dPMLim_diff
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array, cmap=cmap, 
                                vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
                                interpolation="nearest", interpolation_stage="data")
                    plt.colorbar()
                        
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.title(f"PML Estimation, Difference Censored \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.savefig(f"{folderpath}dPMLim_diff.png", bbox_inches="tight")
                    plt.close()

                    image = dPMLim_diff - real_scene_z
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array, vmax=0.5*np.nanmax(masked_array), vmin=-0.5*np.nanmax(masked_array), cmap=cwcmap, interpolation="nearest", interpolation_stage="data")
                    plt.colorbar()
                        
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.title(f"Abs. Err. of PML Estimation, Difference Censored \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.savefig(f"{folderpath}dPMLim_abserr_diff.png", bbox_inches="tight")
                    plt.close()

                    image = np.nanstd(ts_cen_diff, axis=-1)
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array, cmap=rcmap, interpolation="nearest", interpolation_stage="data")
                    plt.colorbar()
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.grid()
                    plt.title(f"Detecn. Time Std. Dev., Difference Cnsrd. \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.savefig(f"{folderpath}cen_std_diff.png", bbox_inches="tight")
                    plt.close()

                    # filtering
                    print("missing data, before filter", np.count_nonzero(k_cen_diff == 0))

                    start_time = time.time()
                    scene_mean = np.nanmean(ts_cen_diff)
                    scene_std = np.nanstd(ts_cen_diff)
                    print("mean", scene_mean)
                    print("std", scene_std)
                    print("out of bound pixel num = ", np.count_nonzero(np.abs(ts_cen_diff - scene_mean) > 2.0*scene_std))

                    ts_cen_diff = np.where(np.logical_or(np.isnan(ts_cen_diff), np.abs(ts_cen_diff - scene_mean) > 2.0*scene_std), np.nan, ts_cen_diff)
                    
                    print("missing data, filter", np.count_nonzero(np.count_nonzero(~np.isnan(ts_cen_diff), axis=-1) == 0))

                    filterT = time.time() - start_time
                    print("Difference, filtering used = ", filterT)
                    filterT_arr.append(filterT)

                    image = np.nanstd(ts_cen_diff, axis=-1)
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array,  cmap=rcmap, interpolation="nearest", interpolation_stage="data")
                    plt.colorbar()
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.grid()
                    plt.title(f"Detecn. Time Std. Dev., Difference Cnsrd., Fltrd. \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.savefig(f"{folderpath}cen_std_diff_fil.png", bbox_inches="tight")
                    plt.close()

                    image = np.nanmean(ts_cen_diff, axis=-1)
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array, cmap=cmap, interpolation="nearest", interpolation_stage="data")
                    plt.colorbar()
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.grid()
                    plt.title(f"Mean Detecn. Time, Difference Cnsrd., Fltrd. \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.savefig(f"{folderpath}cen_mean_diff_fil.png", bbox_inches="tight")
                    plt.close()

                    image = np.nanmean(ts_cen_diff, axis=-1) - true_sig_ts
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array, vmax=0.5*np.nanmax(masked_array), vmin=-0.5*np.nanmax(masked_array), cmap=cwcmap, interpolation="nearest", interpolation_stage="data")
                    plt.title(f"Abs. Err. of Detecn. Time, Difference Cnsrd., Fltrd. \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.colorbar()
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.grid()
                    plt.savefig(f"{folderpath}cen_mean_abserr_diff_fil.png", bbox_inches="tight")
                    plt.close()

                    
                    image = np.count_nonzero(~np.isnan(ts_cen_diff), axis=-1)
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array, 
                                vmin=0, vmax=find_vmax(masked_array),
                                cmap=plt.get_cmap("gray"), interpolation="nearest", interpolation_stage="data")
                    plt.colorbar()
                        
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.grid()
                    plt.title(f"Num. of Detecns., Difference Cnsrd., Fltrd. \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.savefig(f"{folderpath}cen_num_diff_fil.png", bbox_inches="tight")
                    plt.close()


                    ig_diff_fil = calculate_initial_guess(np.nanmean(ts_cen_diff, axis=-1))

                    image = ig_diff_fil
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array, cmap=cmap, 
                                vmin=find_vmin(masked_array), vmax=find_vmax(masked_array),
                                interpolation="nearest", interpolation_stage="data")
                        
                    plt.colorbar()
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.grid()
                    plt.title(f"Initial Guess, Difference Cnsrd., Fltrd. \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.savefig(f"{folderpath}ig_diff_fil.png", bbox_inches="tight")
                    plt.close()

                    print("PML, Difference, Filtered")
                    dPMLresult_diff_fil = trial_z(ts_cen_diff, ig_diff_fil, folderpath, "Diff_Fltrd", beta_z=10000, ftol=1E-3)
                    dPMLim_diff_fil = dPMLresult_diff_fil.x.reshape(scene_shape)

                    np.savetxt(f"{folderpath}dPML_diff_fil.csv", dPMLim_diff_fil, delimiter=",")

                    RMSE_diff_fil = depth_RMSE(dPMLim_diff_fil, real_scene_z)
                    print(f"RMSE, Difference, Filtered = {RMSE_diff_fil:.6f}")
                    RMSE_diff_fil_arr.append(RMSE_diff_fil)

                    image = dPMLim_diff_fil
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array, cmap=cmap, 
                                vmin=np.nanmin(masked_array), vmax=np.nanmax(masked_array),
                                interpolation="nearest", interpolation_stage="data")
                    plt.colorbar()
                        
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.grid()
                    plt.title(f"PML Estimation, Difference, Filtered \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.savefig(f"{folderpath}dPMLim_diff_fil_new.png", bbox_inches="tight")
                    plt.close()

                    image = dPMLim_diff_fil - real_scene_z
                    masked_array = np.ma.array(image, mask=np.isnan(image))
                    plt.imshow(masked_array, vmax=0.5*np.nanmax(masked_array), vmin=-0.5*np.nanmax(masked_array), cmap=cwcmap, interpolation="nearest", interpolation_stage="data")
                    plt.colorbar()
                        
                    plt.tick_params(axis='both', 
                                    left=False, 
                                    top=False, 
                                    right=False, 
                                    bottom=False, 
                                    labelleft=False, 
                                    labeltop=False, 
                                    labelright=False, 
                                    labelbottom=False)
                    plt.grid()
                    plt.title(f"Abs. Err. of PML Estimation, Difference, Filtered \n SBR = {sbr_txt}, SPPP = {sppp}, Trial # {exp_iter}")
                    plt.savefig(f"{folderpath}dPMLim_abserr_diff_fil_new.png", bbox_inches="tight")
                    plt.close()

                    # computeT_ROM_arr.append(computeT_ROM)
                    # rejectT_ROM_arr.append(rejectT_ROM)

                    # computeT_mode_arr.append(computeT_mode)
                    # rejectT_mode_arr.append(rejectT_mode)

                    computeT_diff_arr.append(computeT_diff)
                    rejectT_diff_arr.append(rejectT_diff)
                
                datapath = f"repeat/{scene}_sbr{str(int(sbr*1000)).zfill(4)}_sppp{(str(int(sppp*1000)).zfill(4))}_data/"
                    
                if not os.path.exists(datapath):
                    os.makedirs(datapath)

                # np.savetxt(f"{datapath}computeT_ROM.csv", computeT_ROM_arr, delimiter=",")
                # np.savetxt(f"{datapath}rejectT_ROM.csv", rejectT_ROM_arr, delimiter=",")
                # np.savetxt(f"{datapath}computeT_mode.csv", computeT_mode_arr, delimiter=",")
                # np.savetxt(f"{datapath}rejectT_mode.csv", rejectT_mode_arr, delimiter=",")
                np.savetxt(f"{datapath}computeT_diff.csv", computeT_diff_arr, delimiter=",")
                np.savetxt(f"{datapath}rejectT_diff.csv", rejectT_diff_arr, delimiter=",")
                np.savetxt(f"{datapath}filterT_diff.csv", filterT_arr, delimiter=",")

                # np.savetxt(f"{datapath}RMSE_ROM.csv", RMSE_ROM_arr, delimiter=",")
                # np.savetxt(f"{datapath}RMSE_mode.csv", RMSE_mode_arr, delimiter=",")
                np.savetxt(f"{datapath}RMSE_diff.csv", RMSE_diff_arr, delimiter=",")
                np.savetxt(f"{datapath}RMSE_diff_fil_new.csv", RMSE_diff_fil_arr, delimiter=",")
                
                