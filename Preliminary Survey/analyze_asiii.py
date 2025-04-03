# functions for helping analyze absorbance data from an AS III Assay

# imports
import numpy as np
import re
from scipy.optimize import curve_fit
from math import nan
from scipy import interpolate
import matplotlib.pyplot as plt

import pandas as pd
#import seaborn as sns
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
#import tensorflow_docs as tfdocs
#import tensorflow_docs.plots
#import tensorflow_docs.modeling
import pickle

# function that reads plate reader absorbance data text file
def read_file(file_name, num_cols, nm_start, nm_end, nm_step):
    all_wells = {}
    for col_reading in range(1,num_cols + 1):
        file = open(file_name)
        nm_match = re.compile(r".*\((\d+)\snm\).*")
        col_match = re.compile(r"(?:\s+\d+){" + str(col_reading - 1) + r"}\s+(\d+)(?:\s+\d+){" + str(num_cols - col_reading) + r"}")
        data_match = re.compile(r"\w(?:\s+\d+[.]\d+){" + str(col_reading - 1) + \
                                r"}\s+(\d+[.]\d+)(?:\s+\d+[.]\d+){" + str(num_cols - col_reading) + r"}")
        nm = 0
        col = 0

        absorbs = np.zeros((8,(nm_end - nm_start)//nm_step + 1))

        cur = 0
        for line in file:
            res = re.match(nm_match, line)
            if res:
                nm = int((int(res.group(1)) - nm_start) / nm_step)
                cur = 0
            res = re.match(col_match, line)
            if res:
                col = int(res.group(1))
            res = re.match(data_match, line)
            if res:
                if col == col_reading:
                    absorbs[cur, nm] = float(res.group(1))
                cur += 1
        
        for r in range(8):
            all_wells[(r+1,col_reading)] = absorbs[r,:]
    return all_wells

# functions for fitting REE content to a single wavelength

def get_absorbance_1fit(ln_conc, *opt):
    return get_absorbance_1(ln_conc, opt)

def get_absorbance_1(ln_conc, opt):
    A, B, Kd, dye_conc = opt
    b = dye_conc + ln_conc + Kd
    compound_conc = (b - (b*b - 4 * ln_conc * dye_conc)**(0.5))/2
    return A * dye_conc + (B - A) * compound_conc

def get_ree_conc_1(absorbance, opt):
    A, B, Kd, dye = opt
    n1 = (absorbance - A * dye)/(B-A)
    return n1 * (Kd + dye - n1) / (dye - n1)

def get_ree_conc_1fit(ln_conc, *opt):
    return get_ree_conc_1(ln_conc, opt)


# ln_conc should be in ascending order and the first value should be 0
def get_p0_guess_1(ln_conc, abses, dye_conc):
    A0 = abses[0] / dye_conc
    abses = np.array(abses)
    idx = np.argmax(abses[1:] - abses[:len(abses)-1])
    x1 = abses[idx+1] - dye_conc * A0
    x2 = abses[idx] - dye_conc * A0
    a = dye_conc * (ln_conc[idx+1] / x1 - ln_conc[idx] / x2)
    b = -(ln_conc[idx+1] - ln_conc[idx])
    c = x1 - x2
    if b**2 < 4*a*c:
        B1 = A0 + -b / 2 / a
        B2 = A0 + -b / 2 / a
    else:
        B1 = A0 + (-b + np.sqrt(b**2 - 4*a*c)) / 2 / a
        B2 = A0 + (-b - np.sqrt(b**2 - 4*a*c)) / 2 / a
    Kd1 = (dye_conc - x1 / (B1-A0)) * (ln_conc[idx+1] - x1 / (B1-A0)) / (x1 / (B1-A0))
    Kd2 = (dye_conc - x1 / (B2-A0)) * (ln_conc[idx+1] - x1 / (B2-A0)) / (x1 / (B2-A0))
    if Kd2 > 0:
        B0 =B1
        Kd0 = Kd1
    else:
        B0 = B2
        Kd0 = Kd2
    return (min(max(0,A0),1), min(max(0,B0),1), min(1200,max(Kd0, .00001)), dye_conc)


def get_bounds_1(ln_conc, abses, dye_conc):
    A0 = max(.000001,abses[0] / dye_conc)
    A_low = 0.8 * A0
    A_high = 1.2 * A0

    B_low = 0
    B_high = 1
    Kd_low = 0
    Kd_high = 1200
    dye_conc_low = dye_conc * 0.9
    dye_conc_high = dye_conc * 1.1
    lower = (A_low, B_low, Kd_low, dye_conc_low)
    upper = (A_high, B_high, Kd_high, dye_conc_high)
    return (lower, upper)


#-------------Fits REE conc using assumption two AS III bind to one REE----------------#

def alt_get_absorbance_1fit(ln_conc, *opt):
    return alt_get_absorbance_1(ln_conc, opt)

def alt_get_absorbance_1(ln_conc, opt):
    A, B, Kd, dye_conc = opt
    comp_concs = np.zeros(len(ln_conc))
    for i in range(len(ln_conc)):
        n3 = 4
        n2 = -4*(ln_conc[i] + dye_conc)
        n1 = Kd + 4*dye_conc*ln_conc[i] + dye_conc**2
        n0 = -dye_conc**2 * ln_conc[i]
        poss_compound_concs = np.roots(np.array([n3, n2, n1, n0]))
        poss = []
        for r in poss_compound_concs:
            if r <= ln_conc[i] and r <= dye_conc / 2 and r >= 0 and np.isreal(r):
                poss.append(r)
        if len(poss) > 0:
            comp_concs[i] = max(poss)
    return A * dye_conc + (B - A) * comp_concs

def alt_get_ree_conc_1(absorbance, opt):
    A, B, Kd, dye = opt
    bound = (absorbance - A * dye)/(B-A)
    return bound * Kd / (dye - 2*bound)**2 + bound

def alt_get_ree_conc_1fit(ln_conc, *opt):
    return alt_get_ree_conc_1(ln_conc, opt)


# ln_conc should be in ascending order and the first value should be 0
def alt_get_p0_guess_1(ln_conc, abses, dye_conc, starting_ad):
    A0 = abses[0] / dye_conc
    abses = np.array(abses)
    # assumes starts with 80% adsorption
    B0 = (abses[1] - abses[0]) / (starting_ad * ln_conc[1]) + A0
    Kd0 = (dye_conc - 2*starting_ad*ln_conc[1])**2 * ((1-starting_ad) * ln_conc[1]) / (starting_ad * ln_conc[1])
    return (A0, B0, Kd0, dye_conc)


def alt_get_bounds_1(ln_conc, abses, dye_conc):
    A0 =abses[0] / dye_conc
    A_low = 0.8 * A0
    A_high = 1.2 * A0

    B_low = 0
    B_high = 1
    Kd_low = 0.000000001
    Kd_high = 1000000
    dye_conc_low = dye_conc * 0.95
    dye_conc_high = dye_conc * 1.05
    lower = (A_low, B_low, Kd_low, dye_conc_low)
    upper = (A_high, B_high, Kd_high, dye_conc_high)
    return (lower, upper)

#-------------Functions for Fitting REE Conc with multiple wavelengths----------------#

# function for fitting REE content to multiple wavelengths

def get_absorbance_nfit(ln_conc, *opt):
    start = ln_conc[0]
    m = 1
    for i in range(1,len(ln_conc)):
        if ln_conc[i] == start:
            break
        m += 1
    n = len(ln_conc) // m
    return get_absorbance_n(ln_conc[:m], opt)

# returns n*m length vector [REE1_w1, REE2_w1 ... REEm_wn]
def get_absorbance_n(ln_conc, opt):
    m = len(ln_conc)
    n = (len(opt) - 2 * m - 1) // 2
    As = np.array(opt[:n]).reshape((n,1))
    Bs = np.array(opt[n:2*n]).reshape((n,1))
    Kd = np.array(opt[2*n])
    if n > 2:
        dyes = np.array(opt[2*n+1:2*n+1+m])
        offsets = np.array(opt[2*n+1+m:])
    else:
        dyes = np.array(opt[2*n+1:])
        offsets = np.zeros(m)
    b = dyes + ln_conc + Kd
    compound_conc = ((b - (b*b - 4 * ln_conc * dyes)**0.5) / 2).reshape(1,m)
    return (As * dyes + (Bs - As) * compound_conc + offsets).reshape(n*m) 

def get_ree_conc_n(absorbances, opt, no_noise=True, dye_conc=None):
    n = len(absorbances)
    As = np.array(opt[:n]).reshape((n,1))
    Bs = np.array(opt[n:2*n]).reshape((n,1))
    Kd = np.array(opt[2*n])
    if dye_conc == None:
        mat = np.hstack((As.reshape((n,1)), (Bs - As).reshape((n,1))))
        if n > 2 and not no_noise:
            mat = np.hstack((mat, np.ones((n,1))))
        x = np.linalg.inv(mat.T.dot(mat)).dot(mat.T).dot(absorbances.reshape(n,1))
        return (x[1] * Kd - x[1]*x[1] + x[0]*x[1]) / (x[0] - x[1])
    else:
        mat = (Bs - As).reshape((n,1))
        if n > 2 and not no_noise:
            mat = np.hstack(mat, np.ones((n,1)))
        ab = absorbances.reshape(n,1) - As.reshape(n,1) * dye_conc
        x = np.linalg.inv(mat.T.dot(mat)).dot(mat.T).dot(ab)
        return (x[0] * Kd - x[0]*x[0] + dye_conc*x[0]) / (dye_conc - x[0])


def get_2ree_conc_n(absorbances, opt1, opt2, no_noise=True, dye_conc=None):
    n = len(absorbances)
    As = np.array(opt1[:n]).reshape((n,1))
    Bs1 = np.array(opt1[n:2*n]).reshape((n,1))
    Kd1 = np.array(opt1[2*n])
    Bs2 = np.array(opt2[n:2*n]).reshape((n,1))
    Kd2 = np.array(opt2[2*n])
    mat = np.hstack((As.reshape((n,1)), (Bs1-As).reshape((n,1)), (Bs2-As).reshape((n,1))))
    x = np.linalg.inv(mat.T.dot(mat)).dot(mat.T).dot(absorbances.reshape(n,1))
    conc1 = x[1] * (Kd1 + x[0] - x[1] - x[2]) / (x[0] - x[1] - x[2])
    conc2 = x[2] * (Kd2 + x[0] - x[1] - x[2]) / (x[0] - x[1] - x[2])
    return conc1, conc2

def get_fit_n(absorbances, opt, no_noise=True):
    n = len(absorbances)
    As = np.array(opt[:n]).reshape((n,1))
    Bs = np.array(opt[n:2*n]).reshape((n,1))
    Kd = np.array(opt[2*n])
    mat = np.hstack((As.reshape((n,1)), (Bs - As).reshape((n,1))))
    if n > 2 and not no_noise:
        mat = np.hstack((mat, np.ones((n,1))))
    return np.linalg.inv(mat.T.dot(mat)).dot(mat.T).dot(absorbances.reshape(n,1))


# assumes first REE concentration is 0
def get_p0_guess_n(ln_conc, abses, dye_conc, m_idx=6, n_idx=5):
    m = len(ln_conc)
    n = len(abses) // m
    A0s = abses[::m] / dye_conc
    # original way of guessing B
    #n_idx_abses = abses[n_idx*m:(n_idx+1)*m]    
    #p0 = get_p0_guess_1(ln_conc, n_idx_abses, dye_conc)
    #bounds = get_bounds_1(ln_conc, n_idx_abses, dye_conc)
    #popt1,pcov = curve_fit(get_absorbance_1fit, ln_conc, 
    #    n_idx_abses, p0=p0, bounds = bounds)
    #est_bound_conc = (abses[n_idx*m + m_idx] - A0s[n_idx] * dye_conc
    #    ) / (popt1[1] - popt1[0])
    #B0s = (abses[m_idx::m] - abses[::m]) / est_bound_conc + A0s
    B0s = np.zeros(len(A0s))
    errs = np.zeros(len(A0s))
    Kd0s = np.zeros(len(A0s))
    for i in range(len(B0s)):
        n_idx_abses = abses[i*m:(i+1)*m]
        p0 = get_p0_guess_1(ln_conc, n_idx_abses, dye_conc)
        bounds = get_bounds_1(ln_conc, n_idx_abses, dye_conc)
        popt,pcov = curve_fit(get_absorbance_1fit, ln_conc,
            n_idx_abses, p0=p0, bounds=bounds)
        synth_abs = get_absorbance_1(ln_conc, popt)
        err = sum(((synth_abs - n_idx_abses) / n_idx_abses)**2) 
        B0s[i] = popt[1]
        Kd0s[i] = popt[2]
    for i in range(len(B0s)):
        B0s[i] = max(min(B0s[i], 1),0)
    best_idx = np.argmin(errs)
    Kd0 = Kd0s[best_idx]
    Kd0 = max(min(Kd0, 100), .1)
    if n > 2:
        return tuple(list(A0s) + list(B0s) + [Kd0] + [dye_conc]*m + [0]*m)
    else:
        return tuple(list(A0s) + list(B0s) + [Kd0] + [dye_conc]*m)

def get_bounds_n(ln_conc, abses, dye_conc, noise_low=-.000001, noise_high=.000001):
    m = len(ln_conc)
    n = len(abses) // m
    p0 = get_p0_guess_n(ln_conc, abses, dye_conc, 0, 0)
    A_lows = 0.9 * np.array(p0[:n])
    A_highs = 1.1 * np.array(p0[:n])
    for i in range(len(A_lows)):
        A_lows[i] = min(p0[i] - .000001, A_lows[i])
        A_highs[i] = max(A_highs[i], p0[i] + .000001)
    B_lows = [0] * n
    B_highs = [1] * n
    Kd_low = [0]
    Kd_high = [200]
    dye_lows = [dye_conc * 0.98]*m
    dye_highs = [dye_conc * 1.02]*m
    offset_lows = [noise_low]*m
    offset_highs = [noise_high]*m
    lower = tuple(list(A_lows) + list(B_lows) + Kd_low + dye_lows + offset_lows)
    upper = tuple(list(A_highs) + list(B_highs) + Kd_high + dye_highs + offset_highs)
    if m == 2:
        lower = lower[:len(lower)-m]
        upper = upper[:len(upper)-m]
    return (lower, upper)

def refit_ln_conc(ln_conc, num_wavelengths):
    ln_conc = list(ln_conc)
    final = []
    for i in range(num_wavelengths):
        final += ln_conc
    return np.array(final)


#------functions for finding 2 REE concentrations in a single assay with 2 ASIII Adds--------------#

# Function that finds absorbance if two REEs are present in the solution
def get_2ree_abs(AS_T, REE1_conc, REE2_conc, Kd1, Kd2, A, B1, B2):
    if REE1_conc == 0 and REE2_conc == 0:
        return AS_T * A
    elif REE1_conc == 0:
        return get_absorbance_1(REE2_conc, (A,B2,Kd2,AS_T))
    elif REE2_conc == 0:
        return get_absorbance_1(REE1_conc, (A,B1,Kd1,AS_T))
    coeffs = [Kd2 * AS_T * REE1_conc**2, 
              Kd1 * (AS_T * REE1_conc - REE1_conc * REE2_conc) - Kd2 * REE1_conc * (2*AS_T + REE1_conc + Kd1),
              Kd2 * (2*REE1_conc + Kd1 + AS_T) - Kd1 * (REE1_conc + AS_T + Kd1 - REE2_conc),
              Kd1 - Kd2]
    coeffs = np.array(coeffs[::-1])
    bound1_ops = np.roots(coeffs)
    bound2_ops = AS_T - bound1_ops - Kd1 * (bound1_ops) / (REE1_conc - bound1_ops)
    bound1 = 0
    bound2 = 0
    for poss1, poss2 in zip(bound1_ops, bound2_ops):
        if poss1 + poss2 <= AS_T and np.isreal(poss1) and np.isreal(poss2) and poss1 >= 0 and poss2 >= 0:
            bound1 = poss1
            bound2 = poss2
            break
    return A * AS_T + (B1 - A) * bound1 + (B2 - A) * bound2

# Finds total REE concs if bound REE concentrations are known
# Returns nan if calculates that conc < 0 or conc > cap
def get_2ree_from_bound(AS_T, Kd1, Kd2, bound1, bound2, cap):
    ln1 = (Kd1 + AS_T - bound1 - bound2) * bound1 / (AS_T - bound1 - bound2)
    ln2 = (Kd2 + AS_T - bound1 - bound2) * bound2 / (AS_T - bound1 - bound2)
    for i in range(len(bound1)):
        if ln1[i] < 0 or ln1[i] > cap or ln2[i] > cap or ln2[i] < 0: 
            ln1[i] = nan
            ln2[i] = nan
    return ln1, ln2

# Gets reasonable range of possible bound REE concentrations
def get_possible_bounds(dye_conc, A, B1, B2, absorb):
    min_bound1 = max(0, (absorb - A * dye_conc - (B2 - A) * dye_conc) / (B1 - A))
    max_bound1 = min(dye_conc, (absorb - (A * dye_conc)) / (B1 - A))
    dye_bound1s = np.arange(min_bound1, max_bound1, 0.01)
    dye_bound2s = (absorb - A * dye_conc - (B1 - A) * dye_bound1s) / (B2 - A)
    
    return dye_bound1s, dye_bound2s

# Returns measured REE concentration
def measure_2ree(abs1, abs2, A, B1, B2, Kd1, Kd2, dye_conc1, 
    perc_dye_conc2, added_dye_conc, max_REE_conc):
    
    dye_conc2 = added_dye_conc * perc_dye_conc2 + (1 - perc_dye_conc2) * dye_conc1

    dye1_bound1s, dye1_bound2s = get_possible_bounds(dye_conc1, A, B1, B2, abs1)
    dye1_ree1s, dye1_ree2s = get_2ree_from_bound(dye_conc1, Kd1, Kd2, dye1_bound1s,
        dye1_bound2s, max_REE_conc)

    dye2_bound1s, dye2_bound2s = get_possible_bounds(dye_conc2, A, B1, B2, abs2)
    dye2_ree1s, dye2_ree2s = get_2ree_from_bound(dye_conc2, Kd1, Kd2, dye2_bound1s,
        dye2_bound2s, max_REE_conc * (1 - perc_dye_conc2))

    dye2_ree1s /= (1 - perc_dye_conc2)
    dye2_ree2s /= (1 - perc_dye_conc2)

    best_ree1 = 0
    best_ree2 = 0
    best_err = 100000000
    for dye1_ree1, dye1_ree2 in zip(dye1_ree1s, dye1_ree2s):
        for dye2_ree1, dye2_ree2 in zip(dye2_ree1s, dye2_ree2s):
            err = (dye1_ree1 - dye2_ree1)**2 + (dye1_ree2 - dye2_ree2)**2
            if err < best_err:
                best_err = err
                best_ree1 = (dye1_ree1 + dye2_ree1) / 2
                best_ree2 = (dye1_ree2 + dye2_ree2) / 2
    return best_ree1,best_ree2



# -------functions for doing spline fit on multiple wavelengths for single REE-------------#

# does best fit of REE from single wavelength
def get_ree_from_spline(spline, ab, small, big, step):
    poss = np.arange(small, big, step)
    best = small
    best_err = abs(ab -  spline(small))
    for i in range(len(poss)):
        test_val = spline(poss[i])
        if abs(ab - test_val) < best_err:
            best_err = abs(ab - test_val)
            best = poss[i]
    return best

# each entry in map corresponds to ree value in np.arange(small,big,step)
def get_ree_from_spline_fast(spline, ab, small, big, step, abs_map):
    poss = np.arange(small, big, step)
    best = small
    best_err = abs(ab - abs_map[0])
    for i in range(len(poss)):
        test_val = abs_map[i]
        if abs(ab - test_val) < best_err:
            best_err = abs(ab - test_val)
            best = poss[i]
    return best

# returns splines and weights for each wavelength
# abses = [abs_conc1, abs_conc2, ...]
def get_spline_n(ln_conc, abses, nms_idxes, grain=100):
    splines = []
    weights = []
    for idx in nms_idxes:
        ab = []
        for i in range(len(abses)):
            ab.append(abses[i][idx])
        spline = interpolate.UnivariateSpline(ln_conc, ab)
        errs = []
        small = min(ln_conc)
        big = max(ln_conc)
        step = (big - small) / grain
        for i in range(len(ln_conc)):
            errs.append(ln_conc[i] - get_ree_from_spline(spline, ab[i], small, big, step))
        weights.append(1 / np.std(errs))
        splines.append(spline)
    return splines, weights

def get_spline_n_map(splines, small, big, step):
    spline_maps = []
    poss = np.arange(small, big, step)
    for i in range(len(splines)):
        cur_map = np.zeros(len(poss))
        for j in range(len(poss)):
            cur_map[j] = splines[i](poss[j])
        spline_maps.append(cur_map)
    return spline_maps


def get_ree_from_spline_n(splines, weights, abses, small, big, step):
    poss = np.arange(small, big, step)
    best = 0
    best_err = 0
    for i in range(len(splines)):
        best_err += (splines[i](0) - abses[i])**2 * weights[i]**2

    for j in range(len(poss)):
        test_err = 0
        for i in range(len(splines)):
            test_err += (splines[i](poss[j]) - abses[i])**2 * weights[i]**2
        if test_err < best_err:
            best_err = test_err
            best = poss[j]
    return best


def get_ree_from_spline_n_fast(splines, weights, abses, small, big, step, mapping):
    poss = np.arange(small, big, step)
    best = 0
    best_err = 0
    for i in range(len(splines)):
        best_err += (mapping[i][0] - abses[i])**2 * weights[i]**2

    for j in range(len(poss)):
        test_err = 0
        for i in range(len(splines)):
            test_err += (mapping[i][j] - abses[i])**2 * weights[i]**2
        if test_err < best_err:
            best_err = test_err
            best = poss[j]
    return best


#-------functions for doing splines fit on multiple wavelengths for two REEs-------------#

# each abses[i] is an array with the ith wavelength measurements for each REE pair
def get_2ree_spline(ree1s, ree2s, abses, grain=50):
    splines = []
    weights = []
    for i in range(len(abses)):
        splines.append(interpolate.SmoothBivariateSpline(ree1s,ree2s,abses[i]))
        min_ab = min(abses[i])
        max_ab = max(abses[i])
        ab_range = max_ab - min_ab
        errs = []
        for j in range(len(ree1s)):
            ab = abses[i][j]
            ab_guess = splines[-1](ree1s[j], ree2s[j])
            errs.append(((ab - ab_guess) / ab_range)**2)
        weights.append(1 / np.sqrt(np.mean(errs)))
    return splines,weights

def get_2ree_spline_maps(splines, small1, big1, step1, small2, big2, step2):
    maps = []
    for i in range(len(splines)):
        cur_map = {}
        for ree1 in np.arange(small1,big1,step1):
            for ree2 in np.arange(small2,big2,step2):
                cur_map[(ree1,ree2)] = splines[i](ree1,ree2)
        maps.append(cur_map)
    return maps

def get_2ree_from_spline(splines, weights, abses, small1, big1, step1, small2, big2, step2):
    poss1 = np.arange(small1, big1, step1)
    poss2 = np.arange(small2, big2, step2)
    best1 = small1
    best2 = small2
    best_err = 0
    for i in range(len(splines)):
        best_err += abs(abses[i] - splines[i](small1, small2))**2 * weights[i]**2
    for ree1 in poss1:
        for ree2 in poss2:
            err = 0
            for i in range(len(splines)):
                err += abs(abses[i] - splines[i](ree1, ree2))**2 * weights[i]**2
            if err < best_err:
                best_err = err
                best1 = ree1
                best2 = ree2
    return best1, best2

def get_2ree_from_spline_fast(splines, weights, abses, small1, big1, step1, 
    small2, big2, step2, maps):
    poss1 = np.arange(small1, big1, step1)
    poss2 = np.arange(small2, big2, step2)
    best1 = small1
    best2 = small2
    best_err = 0
    for i in range(len(splines)):
        best_err += abs(abses[i] - maps[i][(small1, small2)])**2 * weights[i]**2
    for ree1 in poss1:
        for ree2 in poss2:
            err = 0
            for i in range(len(splines)):
                err += abs(abses[i] - maps[i][(ree1, ree2)])**2 * weights[i]**2
            if err < best_err:
                best_err = err
                best1 = ree1
                best2 = ree2
    return best1, best2


#------neural net fitting for a single ree---------------------------------------------#

def get_tensorflow_1ree_predictor(dataset):

    # splits dataset
    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    all_dataset = dataset.copy()

    # sets labels to REE
    train_labels = train_dataset.pop('REE')
    test_labels = test_dataset.pop('REE')
    all_labels = all_dataset.pop('REE')

    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    normed_all_data = norm(all_dataset)

    # sets up model
    def build_model():
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model

    model = build_model()
    model.summary()

    # train the model
    EPOCHS = 2000

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    history = model.fit(
      normed_train_data, train_labels,
      epochs=EPOCHS, validation_split = 0.2, verbose=0,
      callbacks=[early_stop, tfdocs.modeling.EpochDots()])

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    print("Testing set Mean Abs Error: {:5.2f} uM REE".format(mae))

    # plots test predictions vs actual
    test_predictions = model.predict(normed_test_data).flatten()

    plt.figure()
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [REE]')
    plt.ylabel('Predictions [REEs]')
    lims = [0, 40]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)

    error = test_predictions - test_labels
    plt.figure()
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [REE]")
    _ = plt.ylabel("Count")

    # plots all predictions vs actual
    all_predictions = model.predict(normed_all_data).flatten()

    plt.figure()
    a = plt.axes(aspect='equal')
    plt.scatter(all_labels, all_predictions)
    plt.xlabel('True Values [REE]')
    plt.ylabel('Predictions [REE]')
    lims = [0, 40]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)

    error = all_predictions - all_labels
    plt.figure()
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [REE]")
    _ = plt.ylabel("Count")

    return model


def predict_1ree(dataset, model):
    return model.predict(dataset).flatten()

#-----neural net fitting for two rees--------------------------------------------------#


