# patch note #
# 0. prototype  11032025
# 1. patch      11052025    5" boundary argmin leftest -> rightest
# 2. patch      11052025    detached array merging by ampliconic save and postprocessing by k-mer dist overlap: for array, check left[0:mode] has any overlap with ampliconic -> No. If do this, this break the definition of array.... 1Kbp LINE isertion make these kind of array breaking. n + 1. so rest 1 not be caught. So, Anyway, It should be found by ampliconic array identification. K-mer dist not proper for this: erosion between long-ragne dist. So do ampliconic identification, and linking them within, between using k-mer connection
# 3. patch      11052025    0 division for meta data cal - remove 0 in nparray for metadata calculation
# 4. patch      11052025    default min(jacu) for recognition 0.9 -> 0.8
# 5. patch      11082025    unit by minimap
# 6. patch      11092025    unit terminal adjust + last end estimation

import argparse
import pandas as pd
import numpy as np
import subprocess
import pysam
import os
import tempfile
import sys
from collections import Counter
import edlib

print = lambda *a, **k: __builtins__.print(*a, file=sys.stderr, **k)

ALL_COLUMNS = [
    "SeqName", "Start", "End", "KmerFreq", "Kmer", "RCKmer",
    "FKPos", "FKDist", "FKDistSqrt", "FKDistLog",
    "RKPos", "RKDist", "RKDistSqrt", "RKDistLog",
    "minKDist", "maxKDist", "H", "RC", "G", "HN", "HNMean", "HNVar", "maxJacu", "minJacu"
]

USECOLS = [
    "SeqName", "Start", "End", "KmerFreq",
    "FKPos", "FKDist", "RKPos", "RKDist",
    "minKDist", "maxKDist", "RC", "HN", "maxJacu", "minJacu"
]

DTYPES = {
    "SeqName": "category",
    "Start": "int32",
    "End": "int32",
    "KmerFreq": "int32",
    "FKPos": "int32",
    "FKDist": "float32",
    "RKPos": "int32",
    "RKDist": "float32",
    "minKDist": "float32",
    "maxKDist": "float32",
    "RC": "int16",
    "HN": "float32",
    "maxJacu": "float32",
    "minJacu": "float32",
}


def detect_signal(hn, rc, jacu, hn_detect, jacu_detect): # hndetect: 50, jacudetect
    if hn <= hn_detect and jacu >= jacu_detect:
        return True
    else:
        return False

def elon_signal(hn, rc, jacu, hn_elon, jacu_elon): # hndetect: 50, jacudetect
    if hn > hn_elon or jacu < jacu_elon: 
        return False
    else:
        return True

def termination_signal(S, E, J, candidate, win_ter, len_ter_cut, jacu_ter_soft, jacu_ter_hard, eps_up): # termination accepted gap, hn, jacu
    pos0 = int(candidate[1])
    n = J.shape[0]
    n_valid = n - (win_ter * 2)
    win_s = pos0
    win_e = min(pos0 + int(win_ter), n_valid)

    if win_e - win_s < 2: # only >2 would be run for below for phrase
        bstart, bend = _exact_boundary(S, E, J, win_ter, candidate, eps_up)
        return [True, bstart, bend]

    pre_win_len = max(1, int(round(win_ter / 2, 0)))
    pre_s = max(0, win_s - pre_win_len)
    pre_e = win_s
    if pre_s < pre_e:
        baseline = float(np.median(J[pre_s:pre_e]))
    else:
        baseline = 1.0
    if win_e - win_s < int(round(win_ter / 2, 0)): baseline = 1.0

    
    cur_len = 0
    drop_max = 0.0
    len_drop_max = 0
    prev_jacu = float(J[win_s])         # jacu of start
    min_jacu = prev_jacu
    for i in range(win_s + 1, win_e):   
        jacu = float(J[i])                 # jacu i from pos_0 + 1
        d = jacu - prev_jacu                    # delta jacu
        if jacu < min_jacu: 
            min_jacu = jacu # check min jacu
      
        if cur_len == 0: # initiation
            if d < 0:                   # -> decreasing start
                cur_trough = jacu
                cur_len = 1
        else:
            if d > float(eps_up): # drop period termination, save drop length with maximization
                drop = baseline - cur_trough
                if cur_len > len_drop_max:
                    len_drop_max = cur_len
                    drop_max = drop
                elif cur_len == len_drop_max and drop > drop_max:
                    drop_max = drop
                cur_len = 0
            else: # dropping
                cur_len += 1
                if jacu < cur_trough:
                    cur_trough = jacu
        
        prev_jacu = jacu

    if cur_len > 0:
        drop = baseline - cur_trough
        if cur_len > len_drop_max:
            len_drop_max = cur_len
            drop_max = drop
        elif cur_len == len_drop_max and drop > drop_max:
            drop_max = drop
    
    terminate = ((len_drop_max >= len_ter_cut and drop_max >= jacu_ter_soft) or (min_jacu <= jacu_ter_hard))
    if (len_drop_max >= len_ter_cut and drop_max >= jacu_ter_soft):
        term_type = "soft"
    elif (min_jacu <= jacu_ter_hard):
        term_type = "hard"
    else:
        term_type = "none"

    print(f"[TERMINATION] pos0={pos0}, drop_max={drop_max:.3f}, len_drop_max={len_drop_max}, min_jacu={min_jacu:.3f}, type={term_type}, terminate={terminate}")

    if terminate == True: # termination
        print(f"[TERMINATION : BOUNDARY] Triggered at pos0={pos0}")
        bstart, bend  = _exact_boundary(S, E, J, win_ter, candidate, eps_up)
        return [True, bstart, bend]
    else: # elongation
        return [False, None, win_e]


def _exact_boundary(S, E, J, win_ter, candidate, eps_up): # exact boundary identification by jacu
    # get 5' boundary and 3' boundary
    # 1. find zero 
    n = J.shape[0]
    n_valid = n - (win_ter * 2)

    can_start, can_end = candidate[0], candidate[1]
    f_start, f_end = can_start - win_ter, can_start
    r_start, r_end = can_end, can_end + win_ter

    if f_start <0: # no minima in 5". so need to find maxima
        baseline_w = int(round(win_ter / 2, 0))
        baseline = float(np.median(J[can_start : min(can_start + baseline_w, can_end)]))
        maxima, max_pos = 0, 0
        for i in range(0, can_start + baseline_w):
            if J[i] > maxima + eps_up:
                maxima, max_pos = J[i], i
        b_start = max_pos
    else:
        left_slice  = J[f_start:f_end]
        #left_idx_rel  = int(np.argmin(left_slice))  if left_slice.size  > 0 else 0
        left_idx_rel = len(left_slice) - 1 - np.argmin(left_slice[::-1]) if left_slice.size > 0 else 0 # rightest minimum in 5" boudnary
        b_start = f_start + left_idx_rel + win_ter

    if r_end > n_valid: # no minima in 3". so need to find maxima
        baseline_w = int(round(win_ter / 2, 0))
        baseline = float(np.median(J[max(can_start, can_end - baseline_w) : can_end]))
        maxima, max_pos = 0, 0
        for i in range(can_end - baseline_w, n_valid):
            if J[i] > maxima + eps_up:
                maxima, max_pos = J[i], i
        if max_pos == 0: max_pos = can_end - baseline_w
        b_end = max_pos + (win_ter * 2)
    else:
        right_slice = J[r_start:r_end]
        right_idx_rel = int(np.argmin(right_slice)) if right_slice.size > 0 else 0
        b_end = r_start + right_idx_rel + win_ter
    
    #left_slice  = J[f_start:f_end]
    #right_slice = J[r_start:r_end]

    #left_idx_rel  = int(np.argmin(left_slice))  if left_slice.size  > 0 else 0
    #right_idx_rel = int(np.argmin(right_slice)) if right_slice.size > 0 else 0
    
    #b_start = f_start + left_idx_rel + win_ter
    #b_end = r_start + right_idx_rel + win_ter
    print(
        f"[BOUNDARY] can=({can_start},{can_end}) "
        f"-> b_start={b_start}, b_end={b_end} | "
        f"left_min=(idx:{f_start + left_idx_rel if f_start >= 0 else 'N/A'}, "
        f"val:{(left_slice.min() if f_start >= 0 and left_slice.size>0 else float('nan')):.3f}) "
        f"right_min=(idx:{r_start + right_idx_rel if r_end <= n_valid else 'N/A'}, "
        f"val:{(right_slice.min() if r_end <= n_valid and right_slice.size>0 else float('nan')):.3f})"
        )
    return [b_start, b_end]

def _array_class(ter, KD):
    start, end = ter[0], ter[1]
    subKD = KD[start:end]
    subKD = subKD[subKD != 0]
    q1, q3 = np.percentile(subKD, [25, 75])
    dists = subKD[(subKD >= q1) & (subKD <= q3)]
    if dists.size == 0: # GUARD: bug can be happen when n = 2
        return "ampliconic", end-start, 0, 0, 0, 0
    mean_dist = dists.mean()
    
    vals, counts = np.unique(dists, return_counts=True)
    mode_dist = vals[np.argmax(counts)]
    mode_prop = counts.max() / counts.sum()

    if mean_dist * 1.2 <= end - start:
         
        return "array", end-start, round(mean_dist, 4), round((end-start) / mode_dist, 4), mode_dist, round(mode_prop, 4)
    else:
        return "ampliconic", end-start, round(mean_dist, 4), round((end-start) / mode_dist, 4), mode_dist, round(mode_prop, 4)


def _array_separation(elon, MJ, jacu_sep_hard, win_ter):
    elon_start, elon_end = elon[0], elon[1]
    subJ = MJ[elon_start:elon_end]
    min_MJ = np.min(subJ)
    if min_MJ < jacu_sep_hard:
        min_idx_rel = np.argmin(subJ)
        min_idx_abs = elon_start + min_idx_rel + win_ter
        return min_idx_abs # first position of separated array
    else:
        return None


def load_kmer_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        comment="#",
        names=ALL_COLUMNS,      
        usecols=USECOLS,        
        na_values=["", "NA", "NaN", "nan"],
        keep_default_na=True,
        engine="c",        
        )
    int_cols   = ["Start", "End", "KmerFreq", "FKPos", "RKPos", "RC"]
    float_cols = ["FKDist", "RKDist", "minKDist", "maxKDist", "HN", "maxJacu", "minJacu"]

    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int32")
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float32")

    df["SeqName"] = df["SeqName"].astype("category")
    return df


def segregate_arrays_by_fkp(arrays, ampliconics, FKP, KD, arr_seg_cut):
    MIN_UNIT_SIZE   = 100
    if not arrays and not ampliconics:
        return arrays, ampliconics
    
    segmented = {}
    candidate = []

    for (s, e), meta in arrays.items():
        length, mean_dist, arr_ratio, mode_dist, mode_prop = meta
        mode = int(round(mode_dist))

        if mode <= MIN_UNIT_SIZE:
            print(f"[INFO] ARRAY-SEG SKIP small mode region {s}-{e}, len={e-s}, mode={mode}")
            arr_type, m_len, m_mean, m_ratio, m_mode, m_prop = _array_class([s, e], KD)
            meta2 = [m_len, m_mean, m_ratio, m_mode, m_prop]
            segmented[(s, e)] = meta2
            continue

        candidate.append([s, e, meta])
        while len(candidate) > 0:
            s, e, meta = candidate[0]
            length, mean_dist, arr_ratio, mode_dist, mode_prop = meta
            mode = int(round(mode_dist))
            sub = KD[s:s+mode]
            sub = sub[sub != 0]

            if sub.size > 0:
                vals, counts = np.unique(sub, return_counts=True)
                init_mode = vals[np.argmax(counts)]
                init_prop = counts.max() / counts.sum()
            else:
                init_mode = 0
                init_prop = 0.0

            if init_mode != 0 and init_prop > 0.2:
                mode = int(init_mode)
                print(f"[INFO] ARRAY-SEG INITMODE {s}-{e} init_mode={init_mode}, init_prop={init_prop:.3f}, meta_mode={mode_dist}")
            
            seg_len = e - s 
            if seg_len < 2.5 * mode: # not enough window
                print(f"[WARNING] ARRAY-SEG short region {s}-{e}, len={seg_len}, mode={mode}")
                arr_type, m_len, m_mean, m_ratio, m_mode, m_prop = _array_class([s, e], KD)
                meta2 = [m_len, m_mean, m_ratio, m_mode, m_prop]
                
                segmented[(s, e)] = meta2
                del candidate[0]
                continue

            print(f"[INFO] ARRAY-SEG start {s}-{e}, len={seg_len}, mode={mode},  cut={arr_seg_cut}")
            f_win_start = s + mode
            f_win_end = min(s + 2 * mode, e)
            con_len = 0
            con_sum = 0
            # first window cal
            for i in range(f_win_start, f_win_end): 
                con_fpos = FKP[i]
                if con_fpos == 0: pass
                elif s <= con_fpos:
                    con_len += 1
                    con_sum += 1
                else:
                    con_len += 1
            if con_len == 0:
                score = 0
            else:
                score = round(con_sum / con_len, 4)
            
            pos = f_win_start
            # inclumental update
            while pos < e - mode:
                pcon_fpos = FKP[pos]
                if pcon_fpos == 0: pass
                elif s <= pcon_fpos:
                    con_len -= 1
                    con_sum -= 1
                else:
                    con_len -= 1
                
                win_end = pos + mode
                pos += 1
                
                ncon_fpos = FKP[win_end]
                if ncon_fpos == 0: pass
                elif s <= ncon_fpos:
                    con_len += 1
                    con_sum += 1
                else:
                    con_len += 1

                if con_len == 0:
                    score = 0
                else:
                    score = round(con_sum / con_len, 4)
                
                if score < arr_seg_cut:
                    print(f"[INFO] SEG-CUT at {pos} in {s}-{e}, score={score}, cut={arr_seg_cut}, scroe={score}")
                    sub_left = KD[s:pos]
                    if np.count_nonzero(sub_left) == 0:
                        print(f"[INFO] ARRAY-SEG DROP-LEFT-ZERO {s}-{pos}, len={pos-s}")
                    else:
                        #print(f"[DEBUG] sub_left KD[{s}:{pos}] = {KD[s:pos]}, nonzero={np.count_nonzero(KD[s:pos])}")
                        arr_type, m_len, m_mean, m_ratio, m_mode, m_prop = _array_class([s, pos], KD)
                        meta_left = [m_len, m_mean, m_ratio, m_mode, m_prop]
                        segmented[(s, pos)] = meta_left
                    
                    sub_right = KD[pos:e]
                    if np.count_nonzero(sub_right) == 0:
                        print(f"[INFO] ARRAY-SEG DROP-RIGHT-ZERO {pos}-{e}, len={e-pos}, score={score}")
                        del candidate[0]
                        break
                    else:
                        #print(f"[DEBUG] sub_right KD[{pos}:{e}] = {KD[pos:e][:20]}, nonzero={np.count_nonzero(KD[pos:e])}")
                        arr_type, m_len, m_mean, m_ratio, m_mode, m_prop = _array_class([pos, e], KD)
                        meta_right = [m_len, m_mean, m_ratio, m_mode, m_prop]
                        candidate[0] = [pos, e, meta_right] 
                        break
            else:
                sub_all = KD[s:e]
                if np.count_nonzero(sub_all) == 0:
                    print(f"[INFO] ARRAY-SEG DROP-NOCUT-ZERO {s}-{e}, len={e-s}")
                    del candidate[0]
                else:
                    arr_type, m_len, m_mean, m_ratio, m_mode, m_prop = _array_class([s, e], KD)
                    meta2 = [m_len, m_mean, m_ratio, m_mode, m_prop]
                    segmented[(s, e)] = meta2
                    del candidate[0]
                    print(f"[INFO] ARRAY-SEG-NOCUT final={s}-{e}, len={e-s}, mode={mode}, score={score}")

    new_arrays = {}
    new_ampliconics = dict(ampliconics)
    for (s, e), meta in segmented.items():
        arr_type, m_len, m_mean, m_ratio, m_mode, m_prop = _array_class([s, e], KD)
        meta2 = [m_len, m_mean, m_ratio, m_mode, m_prop]
        print(f"[INFO] ARRAY-SEG-CLASSIFY {s}-{e} -> {arr_type}  (len={e-s}, mode={m_mode}, ratio={m_ratio:.3f})")
        if arr_type == "array":
            new_arrays[(s, e)] = meta2
        else:
            new_ampliconics[(s, e)] = meta2

    return new_arrays, new_ampliconics



def merge_arrays_by_rkp(arrays, ampliconics, RKP, KD, arr_mer_cut):
    if not arrays and not ampliconics:
        return arrays

    segments = []
    for (s, e), meta in arrays.items():
        segments.append((s, e, meta, "arr"))
    for (s, e), meta in ampliconics.items():
        segments.append((s, e, meta, "amp"))

    segments.sort(key=lambda x: x[0])  # sort by start

    print(f"[INFO] ARRAY-MERGE initial segments: arrays={len(arrays)}, ampliconics={len(ampliconics)}")
    i = 0
    while i < len(segments) - 1:
        cur_start, cur_end, cur_meta, Type = segments[i]
        length, mean_dist, arr_ratio, mode_dist, mode_prop = cur_meta
        mode = int(round(mode_dist))
        j = i + 1


        next_start, next_end, next_meta, next_Type = segments[j]
        anchor = cur_end + (mode * 0.5)
        if not (next_start <= anchor < next_end):
            i += 1
            print(f"[ANCHOR-SKIP] L={cur_start}-{cur_end}:{Type}, R={next_start}-{next_end}:{next_Type}, mode={mode}, anchor={anchor:.1f}")
            continue
        
        print(f"[ANCHOR-PASS] L={cur_start}-{cur_end}:{Type}, R={next_start}-{next_end}:{next_Type}, mode={mode}, anchor={anchor:.1f}")
        win_start = max(cur_start, cur_end - mode)
        win_end = cur_end
        hit = 0
        for pos in range(win_start, win_end):
            next_pos = RKP[pos]
            if next_pos == 0:
                continue
            if next_start <= next_pos < next_end:
                hit += 1

        print(f"[WINSTAT] win={win_start}-{win_end}, winlen={win_end-win_start}, nextlen={next_end-next_start}, hit={hit}") 
        #ratio = (hit / mode) patch by PRAME last tail not merged
        ratio = (hit / (min(next_end - next_start, win_end - win_start)))
        print(f"[INFO] ARRAY-MERGE-CHECK left={cur_start}-{cur_end}:{Type}, right={next_start}-{next_end}:{next_Type}, W={mode}, hit={hit}, ratio={ratio:.3f}, anchor={anchor}")
        if ratio >= arr_mer_cut:
            merged_start = cur_start
            merged_end = next_end
            print(f"[INFO] ARRAY-MERGE {cur_start}-{cur_end}:{Type} and {next_start}-{next_end}:{next_Type} -> {merged_start}-{merged_end}")
            arr_type, m_len, m_mean, m_ratio, m_mode, m_prop = _array_class([merged_start, merged_end], KD)
            merged_meta = [m_len, m_mean, m_ratio, m_mode, m_prop]
            segments[i] = merged_start, merged_end, merged_meta, "arr"
            del segments[j]
        else:
            print(f"[INFO] ARRAY-MERGE-PASS {cur_start}-{cur_end}:{Type} and {next_start}-{next_end}:{next_Type} ratio:{ratio} # kmer:{hit} cutoff:{arr_mer_cut}")
            i += 1

    merged_arrays = {}
    for cur_start, cur_end, cur_meta, Type in segments:
        if Type != "arr":
            continue
        merged_arrays[(cur_start, cur_end)] = cur_meta

    return merged_arrays


def whole_screening(df, hn_detect, jacu_detect, hn_elon, jacu_elon, win_ter, len_ter_cut, jacu_ter_soft, jacu_ter_hard, eps_up, min_size, jacu_sep_hard, arr_mer_cut,  arr_seg_cut):
    df = df.astype({
        "SeqName":  "category",
        "Start":    "int32",
        "End":      "int32",
        "KmerFreq": "int32",
        "FKPos":    "int32",
        "FKDist":   "float32",
        "RKPos":    "int32",
        "RKDist":   "float32",
        "minKDist": "float32",
        "maxKDist": "float32",
        "RC":       "int16",
        "HN":       "float32",
        "maxJacu":  "float32",
        "minJacu":  "float32",
    }).sort_values(["SeqName","Start","End"], kind="mergesort")
    
    arrays = {}
    ampliconics = {}
    state = "free"
    candidate = [] # start, end
    current_pos = 0 # pos after processing

    start_arr = df["Start"].to_numpy(np.int32,    copy=False)
    end_arr   = df["End"].to_numpy(np.int32,      copy=False)
    kfreq_arr = df["KmerFreq"].to_numpy(np.int32, copy=False)
    fkpos_arr = df["FKPos"].to_numpy(np.int32,    copy=False)
    fkdis_arr = df["FKDist"].to_numpy(np.float32, copy=False)
    rkpos_arr = df["RKPos"].to_numpy(np.int32,    copy=False)
    rkdis_arr = df["RKDist"].to_numpy(np.float32, copy=False)
    mink_arr  = df["minKDist"].to_numpy(np.float32, copy=False)
    maxk_arr  = df["maxKDist"].to_numpy(np.float32, copy=False)
    rc_arr    = df["RC"].to_numpy(np.int16,       copy=False)
    hn_arr    = df["HN"].to_numpy(np.float32,     copy=False)
    Mjacu_arr  = df["maxJacu"].to_numpy(np.float32,copy=False)
    mjacu_arr = df["minJacu"].to_numpy(np.float32,copy=False)

    n = start_arr.shape[0]
    n_valid = n - (win_ter * 2)
    seq_static = str(df["SeqName"].iloc[0])
    S, E, J, KD, MJ = start_arr, end_arr, mjacu_arr, mink_arr, Mjacu_arr
    pos = 0 # pos in loop
    while pos < n_valid: 
        start = start_arr[pos]
        if start < current_pos:
            pos += 1
            continue

        end   = end_arr[pos]
        kfreq = kfreq_arr[pos]
        fkpos = fkpos_arr[pos]
        fkdis = fkdis_arr[pos]
        rkpos = rkpos_arr[pos]
        rkdis = rkdis_arr[pos]
        mink  = mink_arr[pos]
        maxk  = maxk_arr[pos]
        rc    = rc_arr[pos]
        hn    = hn_arr[pos]
        jacu  = mjacu_arr[pos]

        if state == "free":
            if detect_signal(hn, rc, jacu, hn_detect, jacu_detect) == True:
                print(f"[DETECT] Detected at pos={pos}, HN={hn:.2f}, Jacu={jacu:.2f}")
                candidate = [start, start + 1]
                state = "elon"
            else: 
                pass

        elif state == "elon":
            if elon_signal(hn, rc, jacu, hn_elon, jacu_elon) == True:
                candidate[1] = end # elongation DONE
            else: # termination check
                print(f"[ELON-END] Ended at pos={pos}, len={candidate[1]-candidate[0]}, region={candidate[0]}-{candidate[1]}")
                elon_start, elon_end = candidate[0], candidate[1]
                state = "termination_check" # enter to termination process
                if (elon_end - elon_start) - min_size > 0:
                    ter_dec, ter_start, ter_end = termination_signal(S, E, J, candidate, win_ter, len_ter_cut, jacu_ter_soft, jacu_ter_hard, eps_up)
                    print(f"[TERMINATION-CHECK] pos={pos}, dec={ter_dec}, ter_start={ter_start}, ter_end={ter_end}")
                    if ter_dec == True: # termination and exact boundary
                        state = "termination"
                        arr_check = _array_class([ter_start, ter_end], KD)
                        meta = arr_check[1:]
                        if arr_check[0] == "array":
                            if meta[3] < 2.5: # meta[3] = number of unit. 2-unit seperation pass -> save
                                arrays[(ter_start, ter_end)] = meta
                                print(f"{state}: array - SAVED {ter_start}-{ter_end}\n")

                            else: # seperation check
                                print(f"[SEPARATION-CHECK] pos={pos}, nunit={meta[3]}, ter_start={ter_start}, ter_end={ter_end}")
                                sep_pos = _array_separation([elon_start, elon_end], MJ, jacu_sep_hard, win_ter)
                                if sep_pos == None: # no max(FRjacu) drop = same array -> save
                                    arrays[(ter_start, ter_end)] = meta
                                    print(f"{state}: array - SAVED {ter_start}-{ter_end}\n")

                                else: # max(FRjacu) drop found = different array -> seperation
                                    print(f"[SEPARATION-FOUND] pos={pos}, sep_pos={sep_pos}, ter_start={ter_start}, ter_end={ter_end}")
                                    arr_check1 = _array_class([ter_start, sep_pos], KD)
                                    arr_check2 = _array_class([sep_pos, ter_end], KD)
                                    meta1 = arr_check1[1:]
                                    meta2 = arr_check2[1:]
                                    if arr_check1[0] == "array":
                                        arrays[(ter_start, sep_pos)] = meta1
                                        print(f"{state}: array - SAVED {ter_start}-{sep_pos}\n")
                                    else:
                                        ampliconics[(ter_start, sep_pos)] = meta
                                        print(f"{state}: ampliconic - PASS {ter_start}-{sep_pos}\n")

                                    if arr_check2[0] == "array":
                                        arrays[(sep_pos, ter_end)] = meta2
                                        print(f"{state}: array - SAVED {sep_pos}-{ter_end}\n")
                                    else:
                                        ampliconics[(sep_pos, ter_end)] = meta
                                        print(f"{state}: ampliconic - PASS {sep_pos}-{ter_end}\n")
                            
                        else:
                            ampliconics[(ter_start, ter_end)] = meta
                            print(f"{state}: ampliconic - PASS  {ter_start}-{ter_end}\n")
                        
                        state = "free"
                        current_pos = ter_end
                        candidate = []
                    else: # jump noise 
                        state = "elon"
                        current_pos = ter_end
                        candidate[1] = ter_end
                else: # filter-out by least length
                    print(f"length filtering: {(elon_end - elon_start)} < {min_size}\n")
                    state = "free"
                    current_pos = elon_end
                    candidate = []
        else: 
            pass
        pos += 1

    if state == "elon":
        elon_start, elon_end = candidate[0], candidate[1]
        state = "termination_check"
        if (elon_end - elon_start) - min_size > 0:
            ter_dec, ter_start, ter_end = termination_signal(S, E, J, candidate, win_ter, len_ter_cut, jacu_ter_soft, jacu_ter_hard, eps_up)
            print(f"[TERMINATION-CHECK] pos={pos}, dec={ter_dec}, ter_start={ter_start}, ter_end={ter_end}")
            if ter_dec == True: # termination and exact boundary
                state = "termination"
                arr_check = _array_class([ter_start, ter_end], KD)
                meta = arr_check[1:]
                if arr_check[0] == "array":
                    if meta[3] < 2.5: # meta[3] = number of unit. 2-unit seperation pass -> save
                        arrays[(ter_start, ter_end)] = meta
                        print(f"{state}: array - SAVED {ter_start}-{ter_end}\n")

                    else: # seperation check
                        print(f"[SEPARATION-CHECK] pos={pos}, nunit={meta[3]}, ter_start={ter_start}, ter_end={ter_end}")
                        sep_pos = _array_separation([elon_start, elon_end], MJ, jacu_sep_hard, win_ter)
                        if sep_pos == None: # no max(FRjacu) drop = same array -> save
                            arrays[(ter_start, ter_end)] = meta
                            print(f"{state}: array - SAVED {ter_start}-{ter_end}\n")

                        else: # max(FRjacu) drop found = different array -> seperation
                            print(f"[SEPARATION-FOUND] pos={pos}, sep_pos={sep_pos}, ter_start={ter_start}, ter_end={ter_end}")
                            arr_check1 = _array_class([ter_start, sep_pos], KD)
                            arr_check2 = _array_class([sep_pos, ter_end], KD)
                            meta1 = arr_check1[1:]
                            meta2 = arr_check2[1:]
                            if arr_check1[0] == "array":
                                arrays[(ter_start, sep_pos)] = meta1
                                print(f"{state}: array - SAVED {ter_start}-{sep_pos}\n")
                            else:
                                ampliconics[(ter_start, sep_pos)] = meta
                                print(f"{state}: ampliconic - PASS {ter_start}-{sep_pos}\n")

                            if arr_check2[0] == "array":
                                arrays[(sep_pos, ter_end)] = meta2
                                print(f"{state}: array - SAVED {sep_pos}-{ter_end}\n")
                            else:
                                ampliconics[(sep_pos, ter_end)] = meta
                                print(f"{state}: ampliconic - PASS {sep_pos}-{ter_end}\n")
                                
                else:
                    ampliconics[(ter_start, ter_end)] = meta
                    print(f"{state}: ampliconic - PASS  {ter_start}-{ter_end}\n")
        else:
            print(f"length filtering: {(elon_end - elon_start)} < {min_size}\n")

    elif state == "free":
        print("Finish")
        pass  

    arrays, ampliconics = segregate_arrays_by_fkp(arrays, ampliconics, fkpos_arr, mink_arr, arr_seg_cut)
    arrays = merge_arrays_by_rkp(arrays, ampliconics, rkpos_arr, mink_arr, arr_mer_cut)

    return arrays


def filter_nested_hit_info(hit_info):
    items = sorted(hit_info.items(), key=lambda kv: (kv[0], -(kv[1][3] - kv[0])))
    new_hit_info = {}

    for gstart, (alen, qlen, eff_qlen, gend, cov, identity) in items:
        nested = False
        for s0, (_, _, _, gend0, _, _) in new_hit_info.items():
            if gstart >= s0 and gend <= gend0:
                print(f"[INFO] nested skip: new {gstart}-{gend} is contained in existing {s0}-{gend0}")
                nested = True
                break
        if nested:
            continue
        new_hit_info[gstart] = (alen, qlen, eff_qlen, gend, cov, identity)

    return new_hit_info

'''
def filter_bridge_hit_info(hit_info):
    items = sorted(hit_info.items(), key=lambda kv: kv[0])  # (gstart, (..gend..))
    keep = [True] * len(items)
    for i in range(1, len(items) - 1):
        gstart_prev, (_, _, _, gend_prev, _, _) = items[i-1]
        gstart_cur,  (_, _, _, gend_cur,  _, _) = items[i]
        gstart_next, (_, _, _, gend_next, _, _) = items[i+1]

        if gstart_cur < gend_prev and gstart_next < gend_cur:
            keep[i] = False

    new_hit_info = {}
    for (gstart, val), k in zip(items, keep):
        if k:
            new_hit_info[gstart] = val
    return new_hit_info
'''

def Last_tail_alignment(fa, chrom, first_unit, last_unit, array_end):
    f_start, f_end = first_unit[0], first_unit[1]
    l_start, l_end = last_unit[0], last_unit[1]
    
    qseq = fa.fetch(chrom, f_start, f_end).upper()
    tseq = fa.fetch(chrom, l_start, array_end).upper()
    
    result = edlib.align(qseq, tseq, mode="HW", task="path")
    if result['editDistance'] == -1:
        print(f"[Warning] Last_tail alignment failed")
        return None, None
   
    if len(result['locations']) > 1:
        best_loc = max(result['locations'], key=lambda x: x[1])
        print(f"[Info] Multiple hits found ({len(result['locations'])}), using rightmost")
    else:
        best_loc = result['locations'][0]

    real_tend = l_start + best_loc[1] + 1
    identity = 1 - (result['editDistance'] / len(qseq))
    print(f"[3-Align-END] query={f_start}-{f_end}, target={l_start}-{array_end}, real_tend={real_tend}, identity={identity:.4f}")
    
    return real_tend, identity


def Forgap_alignment(fa, chrom, paf_line, offset, TINY_GAP_LEN):

    fields = paf_line.strip().split('\t')
    
    qname, qlen, qstart, qend = fields[0], int(fields[1]), int(fields[2]), int(fields[3])
    tname, tlen, tstart, tend = fields[5], int(fields[6]), int(fields[7]), int(fields[8])
    
    qpos = int(qname.split('_')[-1])
    chrom = tname.split(':')[0]
    tpos = int(tname.split(':')[1].split('-')[0])
    
    qabs = qpos + qstart
    tabs = tpos + tstart
    
    qgap = qabs - qpos
    tgap = tabs - qgap    
    if qgap == 0:
        return tpos, 1.0
   
    if 0 < qgap <= TINY_GAP_LEN:
        real_tstart = tabs - qgap
        print(f"[Tiny-shift] {qname}: qgap={qgap}bp (threshold={TINY_GAP_LEN}bp) | query={qpos}-{qabs}, tstart={tabs} -> real_tstart={real_tstart}")
        return real_tstart, None

    if qgap > 1000:
        print(f"[Warning] qgap too large ({qgap}bp) for {qname}, skipping alignment")
        return None, None
    
    qseq = fa.fetch(chrom, qpos, qabs).upper()
    tseq = fa.fetch(chrom, max(0, tgap - offset), tabs).upper()
    
    result = edlib.align(qseq, tseq, mode="HW", task="path")
    if result['editDistance'] == -1:
        print(f"[Warning] alignment failed for {qname}")
        return None, None
    
    real_tstart = max(0, tgap - offset) + result['locations'][0][0]
    real_tend = max(0, tgap - offset) + result['locations'][0][1] + 1
    identity = 1 - (result['editDistance'] / len(qseq))

    print(f"[5-Align] {qname}: qgap={qgap}bp | query={qpos}-{qabs}, target_before={tabs}-{tpos+tend}, target_gap_estimated={max(0, tgap - offset)}-{tabs}")
    print(f"  -> hit_before: q[{qstart}:{qend}] x t[{tstart}:{tend}], hit_after: q[0:{qgap}] x t[{real_tstart - tpos}:{real_tend - tpos}], tgaps: {real_tstart}-{real_tend} | identity={identity:.4f}")
    return real_tstart, identity


def run_minimap2_inmem(fa_handle, chrom, start, end, query_name, query_seq, preset):
    ref_seq = fa_handle.fetch(chrom, start, end)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as tmp_ref:
        tmp_ref.write(f">{chrom}:{start}-{end}\n{ref_seq}\n")
        tmp_ref_path = tmp_ref.name
    
    fasta_str = f">{query_name}\n{query_seq}\n"
    result = subprocess.run(
        ["minimap2", "-x", preset, "--secondary=yes", "-N", "100000", "--eqx", "-w5", "-k15", tmp_ref_path, "-"],
        #["minimap2", "-x", preset, "--secondary=yes", "-N", "100000", "--eqx", "-w5", tmp_ref_path, "-"],
        input=fasta_str, text=True, capture_output=True, check=True,
    )
    os.remove(tmp_ref_path)
    return result.stdout



def get_units(arrays, chrom, fasta_path, secondary_cut, unit_sim_cut, df): 
    # fixed parameter for alignment
    MIN_UNIT_SIZE   = 50      
    HEAD_FRAC = 1.0
    HEAD_FRAC_LONG = 1.2
    UNIT_MERGE_POWER = 0.2
    MINIMAP_PRESET = "map-ont"
    GAP_MERGE_PROP = 0.1
    BEST_OVLP_PROP = 0.75
    COLLAPSE_OVLP_PROP = 0.75
    GAP_NEW_UNIT_TOL = 0.1
    FORWARD_OFFSET=0
    TINY_GAP_LEN = 4
    
    RKP = df["RKPos"].to_numpy(np.int32, copy=False)
    fa = pysam.FastaFile(fasta_path)
    result = {}
    for key in arrays:
        start, end, meta = key[0], key[1], arrays[key] # start end of the array
        mode = meta[3]  # mode = representative unit length
        short_hits = []
        unsimilar_hits = []

        if mode < MIN_UNIT_SIZE:
            print(f"[WARNING] mode={mode} < {MIN_UNIT_SIZE}bp -> skip {chrom}:{start}-{end}")
            result[key] = []
            continue
        
        #if mode <= SHORT_MODE:
        #    head_size = round(mode * HEAD_FRAC_SHORT)
        #    head_size = max(MIN_HEAD, min(head_size, max_head))
        #else:
        #    head_size = round(mode * HEAD_FRAC_LONG)
        #    head_size = min(head_size, max_head)
        arr_len = end - start
        head_size = round(mode * HEAD_FRAC)
        head_size_long = round(mode * HEAD_FRAC_LONG)
        #if arr_len < mode * 2.5:
        #    head_size = round(mode * 1.0)
        #else: head_size = round(mode * HEAD_FRAC)

        while True: # query size recursive lagacy... just keep structure
            q_start = start
            q_end   = start + head_size
            q_end_long = start + head_size_long
            print(f"[QUERY] head_size={head_size}, q_start={q_start}, q_end={q_end}")
            query_seq = fa.fetch(chrom, q_start, q_end)
            query_seq_long = fa.fetch(chrom, q_start, q_end_long)

            paf_text = run_minimap2_inmem(fa, chrom, start, end, f"head_{chrom}_{start}", query_seq, MINIMAP_PRESET)
            print(f">>> {chrom}:{start}-{end} head_size={head_size}")
            
            paf_lines = sorted(paf_text.splitlines(), key=lambda x: int(x.split("\t")[7]))
            print("\n".join(paf_lines))

            paf_text_longq = run_minimap2_inmem(fa, chrom, start, end, f"head_{chrom}_{start}", query_seq_long, MINIMAP_PRESET)
            starts = []
            hit_info = {}
            print(f"# aligned: {len(paf_lines)}")
            for line in paf_lines: # PAF parsing
                if not line:
                    print(f"[WARNING] no alignment found {chrom}:{start}-{end} mode={mode} head_size={head_size}, q_start={q_start}, q_end={q_end}")
                    continue
                cols = line.split("\t")

                tstart, tend, nmatch, alen, qlen = int(cols[7]), int(cols[8]),int(cols[9]), int(cols[10]), int(cols[1])
                
                qstart_val = int(cols[2])
                if qstart_val > 0:
                    real_tstart, identity_adjusted = Forgap_alignment(fa, chrom, line, FORWARD_OFFSET, TINY_GAP_LEN)
                    if real_tstart is not None:
                        tstart = real_tstart - start  # absolute -> relative
                        print(f"[5-ALIGN-APPLIED] adjusted tstart from {int(cols[7])} to {tstart}")

                gstart, gend = start + tstart, start + tend
                
                if gstart + qlen > end:
                    overhang = (gstart + qlen) - end
                    eff_qlen = qlen - overhang
                    if eff_qlen <= 0:
                        raise RuntimeError(f"eff_qlen <= 0 at {chrom}:{start}-{end}, gstart={gstart}, qlen={qlen}")
                    print(f"[INFO] EDGE: overlap tail: gstart={gstart}, gend={gend}, qlen={qlen}, eff_qlen={eff_qlen}, overhang={overhang}")
                else:
                    eff_qlen = qlen

                cov = round(alen / eff_qlen, 4)
                identity = round(nmatch / alen, 4) if alen > 0 else 0.0
                
                if cov < secondary_cut:
                    short_hits.append((start + tstart, start + tend, alen, qlen, eff_qlen, cov, identity))
                    print(f"[INFO] short alignment found {chrom}:{start}-{end} mode={mode} head_size={head_size}, q_start={q_start}, q_end={q_end}, t_start={start + tstart}, t_end={start + tend}, alen={alen}, eff_qlen={eff_qlen}, cov={cov}, identity={identity}")
                    continue
                if identity < unit_sim_cut:
                    unsimilar_hits.append((start + tstart, start + tend, alen, qlen, eff_qlen, cov, identity))
                    print(f"[INFO] unsimilar alignment found {chrom}:{start}-{end} mode={mode} head_size={head_size}, q_start={q_start}, q_end={q_end}, t_start={start + tstart}, t_end={start + tend}, alen={alen}, eff_qlen={eff_qlen}, cov={cov}, identity={identity}")
                    continue
                
                starts.append(gstart)
                s1 = None  # NEW
                for field in cols[12:]:  # NEW: optional tags
                    if field.startswith("s1:i:"):
                        s1 = int(field[5:])
                        break
                prev = hit_info.get(gstart)  # NEW
                if (prev is None) or (alen > prev[0]):  # NEW: keep largest alen
                    hit_info[gstart] = (alen, qlen, eff_qlen, gend, cov, identity)       # NEW
                    print(f"[INFO] alignment save {chrom}:{start}-{end} mode={mode} head_size={head_size}, q_start={q_start}, q_end={q_end}, t_start={start + tstart}, t_end={start + tend}, alen={alen}, eff_qlen={eff_qlen}, cov={cov}, identity={identity}")
            
            hit_info = filter_nested_hit_info(hit_info)
            break
        

        #if len(starts) < 2:
        #    print(f"[WARNING] Only {len(starts)} minimap2 hit(s) ...")
        #    result[key] = []
        #    continue
        
        
        refined_units = prepare_refined_units_for_bed(hit_info, mode, UNIT_MERGE_POWER, paf_text_longq, start, end, secondary_cut, unit_sim_cut, GAP_NEW_UNIT_TOL)

        if refined_units: pass
        else: raise RuntimeError(f"len(refined_units) = 0: check codes {chrom}:{start}-{end}")
        u_start_last, u_end_last, unit_len_last, alen_last, qlen_last, eff_last, cov_last, id_last = refined_units[-1]
        
        # for what?
        # forward alignment miss some units in last. why?
        # first alignment is estimated query by mode. (no problem for end, cause end1 == start2) But this is not working for last end -> get real unit len and alignment to get last unit end
        # 1. short tail
        # 2. minimizer fail? caused by low complexity region... -> side effect
        # 3. cosindering last, last -1 ... -> side effect

        q_start, q_end = refined_units[0][0], refined_units[0][1]
        r_start, r_end = max(start, end - (head_size * 3)), end
        if u_end_last < r_start: # non-OVLP
            r_start = max(u_start_last, u_end_last - int(mode * 0.2))
            print(f"[REV-ALIGN] adjust r_start to {r_start} (u_end_last={u_end_last}, shift={int(mode * 0.2)})")
        else: pass

        print(f"[QUERY] head_size={head_size}, q_start={q_start}, q_end={q_end}, r_start={r_start}, r_end={r_end} for termination end identification")
        query_seq = fa.fetch(chrom, q_start, q_end)
        paf_text = run_minimap2_inmem(fa, chrom, r_start, r_end, f"head_{chrom}_{r_start}", query_seq, MINIMAP_PRESET)
        
        term_hits = []
        for line in paf_text.splitlines(): # PAF parsing for last unit
            if 1 == 1:
                #print(f"{line}")
                if not line:
                    raise RuntimeError (f"no alignment found in terminal identification {chrom}:{start}-{end} mode={mode} head_size={head_size}, q_start={q_start}, q_end={q_end}, r_start={r_start}, r_end={r_end}")
                cols = line.split("\t")
                tstart, tend, nmatch, alen, qlen = int(cols[7]), int(cols[8]),int(cols[9]), int(cols[10]), int(cols[1])
                gstart, gend = r_start + tstart, r_start + tend

                if gstart + qlen > end: # overhang for what? we don't want to keep the last broken unit. but overhang revive the last unit's end
                    print(f"[EDGE] overlap tail: gstart={gstart}, gend={gend}, qlen={qlen} -> skip in termination")
                    continue
                #    overhang = (gstart + qlen) - end
                #    eff_qlen = qlen - overhang
                #    if eff_qlen <= 0:
                #        raise RuntimeError(f"eff_qlen <= 0 at {chrom}:{start}-{end}, gstart={gstart}, qlen={qlen}")
                #    print(f"[EDGE] overlap tail: gstart={gstart}, gend={gend}, qlen={qlen}, eff_qlen={eff_qlen}, overhang={overhang}")
                else:
                    eff_qlen = qlen

                cov = round(alen / eff_qlen, 4)
                identity = nmatch / alen if alen > 0 else 0.0

                if cov < secondary_cut:
                    #short_hits.append((start + tstart, start + tend, alen, qlen, eff_qlen, cov, identity))
                    print(f"[INFO] short alignment found {chrom}:{start}-{end} mode={mode} head_size={head_size}, q_start={q_start}, q_end={q_end}, t_start={start + tstart}, t_end={start + tend}, alen={alen}, eff_qlen={eff_qlen}, cov={cov}, identity={identity}")
                    continue
                if identity < unit_sim_cut:
                    #unsimilar_hits.append((start + tstart, start + tend, alen, qlen, eff_qlen, cov, identity))
                    print(f"[INFO] unsimilar alignment found {chrom}:{start}-{end} mode={mode} head_size={head_size}, q_start={q_start}, q_end={q_end}, t_start={start + tstart}, t_end={start + tend}, alen={alen}, eff_qlen={eff_qlen}, cov={cov}, identity={identity}")
                    continue
                term_hits.append((gstart, gend, alen, qlen, eff_qlen, cov, identity))
                print(f"[INFO] TERM-HIT: using alignment gstart={gstart}, gend={gend}, alen={alen}, qlen={qlen}, eff_qlen={eff_qlen}, cov={cov:.4f}, sim={identity:.4f}")
       
        if len(term_hits) == 0: 
            print(f"[WARNING] reverse alignment not found {chrom}:{start}-{end} mode={mode} head_size={head_size}, q_start={q_start}, q_end={q_end}, r_start={r_start}, r_end={r_end}, alignment_len={len(paf_text.splitlines())}")
            print(f"[WARNING] last_end= just estimated by start + mode")
            u_start_last, u_end_last, unit_len_last, alen_last, qlen_last, eff_qlen_last, cov_last, identity_last = refined_units[-1]
            est_end = int(u_start_last + mode)
            refined_units[-1] = (u_start_last, est_end, est_end - u_start_last, alen_last, qlen_last, eff_qlen_last, cov_last, identity_last)
        else: # any alignment in last unit
            # save best alignment by delta (mode, aln)
            term_hits_sorted = sorted(term_hits, key=lambda x: x[0])  ####### 
            collapsed = []
            for gstart, gend, alenT, qlenT, effT, covT, idT in term_hits_sorted:
                lengthT = gend - gstart
                devT = round(abs(lengthT - mode) / mode, 4)
                if not collapsed:
                    collapsed.append([gstart, gend, alenT, qlenT, effT, covT, idT, devT])
                    continue
                pgstart, pgend, palen, pqlen, peff, pcov, pid, pdev = collapsed[-1]
                overlap = min(gend, pgend) - max(gstart, pgstart)
                if overlap <= 0:
                    collapsed.append([gstart, gend, alenT, qlenT, effT, covT, idT, devT])
                    continue
                shorter = min(gend - gstart, pgend - pgstart)
                ov_prop = overlap / shorter
                if ov_prop >= BEST_OVLP_PROP:
                    if devT < pdev:
                        collapsed[-1] = [gstart, gend, alenT, qlenT, effT, covT, idT, devT] # replace
                        print(f"[INFO] REV-ALIGN: last unit replaced {chrom}:{start}-{end} mode={mode} {pgstart}-{pgend}:{pdev} -> {gstart}-{gend}:{devT}")
                    continue
                pre_cut_len = gstart - pgstart
                pre_cut_dev = round(abs(pre_cut_len - mode) / mode, 4)
                if pre_cut_dev < pdev:
                    collapsed[-1] = [pgstart, gstart, palen, pqlen, peff, pcov, pid, pre_cut_dev]
                    print(f"[INFO] REV-ALIGN: last unit pre-alignemnt adjust {chrom}:{start}-{end} mode={mode} {pgend} ({pdev}) to {gstart}({pre_cut_dev}) {pgstart}-{pgend} -> {pgstart}-{gstart}")
                    collapsed.append([gstart, gend, alenT, qlenT, effT, covT, idT, devT])
                else:
                    post_cut_dev = round(abs((gend-pgend) - mode) / mode, 4)
                    print(f"[INFO] REV-ALIGN: last unit post-alignment adjust (with keep pre) {chrom}:{start}-{end} mode={mode} {gstart} ({devT}) to {pgend}({post_cut_dev}) {gstart}-{gend} -> {pgend}-{gend}")
                    collapsed.append([pgend, gend, alenT, qlenT, effT, covT, idT, post_cut_dev])
            # minimap processing DONE
            refined_units = _refine_alignment_when_ovlp(refined_units, collapsed, mode, COLLAPSE_OVLP_PROP)
            ''' legacy
            for gstart, gend, alenT, qlenT, effT, covT, idT, devT in collapsed:
                u_start_last, u_end_last, unit_len_last, alen_last, qlen_last, eff_qlen_last, cov_last, identity_last = refined_units[-1]
                last_end, last_start = u_end_last, u_start_last
                last_dev = round(abs(unit_len_last - mode) / mode, 4)
                lengthT = gend - gstart
                
                ov_last = min(gend, last_end) - max(gstart, last_start)
                shorter = min(lengthT, unit_len_last)
                ov_last_prop = ov_last / shorter if ov_last > 0 else 0.0

                if gstart <= last_start: 
                    continue
                if ov_last_prop >= COLLPASE_OVLP_PROP:
                    print(f"[INFO] REV-ALIGN: skip last hit {gstart}-{gend} (ov_last={ov_last_prop:.3f}, devT={devT}, last_dev={last_dev})")
                    continue
                dist_prop = abs(gstart - last_end) / mode
                # case 1. alignment start in last unit
                if dist_prop <= GAP_MERGE_PROP: # only boundary slightly overlap
                    new_last_end = gstart
                    new_last_len = new_last_end - last_start
                    refined_units[-1] = (last_start, new_last_end, new_last_len, alen_last, qlen_last, eff_qlen_last, cov_last, identity_last)
                    refined_units.append((gstart, gend, lengthT, alenT, qlenT, effT, covT, idT))
                    print(f"[INFO] REV-ALIGN: refine last unit to {last_start}-{new_last_end} and add a terminal unit {gstart}-{gend} (len={lengthT}, mode={mode})")
                elif last_start < gstart < last_end: # last alignment overlap-last dev check
                    pre_cut_len = gstart - last_start
                    pre_cut_dev = round(abs(pre_cut_len - mode) / mode, 4)
                    if pre_cut_dev < last_dev:
                        new_last_end = gstart
                        new_last_len = new_last_end - last_start
                        refined_units[-1] = (last_start, new_last_end, new_last_len, alen_last, qlen_last, eff_qlen_last, cov_last, identity_last)
                        refined_units.append((gstart, gend, lengthT, alenT, qlenT, effT, covT, idT))
                        print(f"[INFO] REV-ALIGN: refine last unit to {last_start}-{new_last_end} and add a terminal unit {gstart}-{gend} (len={lengthT}, mode={mode})")
                    else:
                        refined_units.append((last_end, gend, gend-last_end, alenT, qlenT, effT, covT, idT))
                        print(f"[INFO] REV-ALIGN: elongate terminal unit {gstart}-{gend} to {last_end}-{gend} by last dev {last_dev} < pre_cut_dev {pre_cut_dev})")
                elif gstart >= last_end:
                    post_elon_len = gend - last_end
                    post_elon_dev = round(abs(post_elon_len - mode) / mode, 4)
                    if post_elon_dev < devT:
                        refined_units.append((last_end, gend, post_elon_len, alenT, qlenT, effT, covT, idT))
                        print(f"[INFO] REV-ALIGN: elongate terminal unit {gstart}-{gend} to {last_end}-{gend} by terminal dev {devT} > post_elon_dev {post_elon_dev})")
                    else:
                        refined_units[-1] = (last_start, gstart, gstart - last_start, alen_last, qlen_last, eff_qlen_last, cov_last, identity_last)
                        refined_units.append((gstart, gend, lengthT, alenT, qlenT, effT, covT, idT))
                        print(f"[INFO] REV-ALIGN: refine last unit to {last_start}-{gstart} and add a terminal unit {gstart}-{gend} by post_elon_dev {post_elon_dev} > devT {devT}")
                else:
                    pass
            '''
            if len(refined_units) >= 2:
                first_unit = refined_units[0]
                last_unit = refined_units[-1]
                real_tend, identity_end = Last_tail_alignment(fa, chrom, first_unit, last_unit, end)
                if real_tend is not None:
                    u_start, u_end, unit_len, alen, qlen_old, eff_qlen, cov, identity = refined_units[-1]
                    refined_units[-1] = (u_start, real_tend, real_tend - u_start, alen, qlen_old, eff_qlen, cov, identity_end if identity_end else identity)
                    print(f"[LAST-TAIL-APPLIED] adjusted last unit end from {u_end} to {real_tend}, identity:{identity_end}")
        result[key] = refined_units
        print(f"[SAVED] {chrom}:{start}-{end} ({len(refined_units)} units)\n\n")

    fa.close()
    print("[DONE] get_units completed.\n")
    return result


def _refine_alignment_when_ovlp(refined_units, collapsed, mode, COLLAPSE_OVLP_PROP):
    for gstart, gend, alenT, qlenT, effT, covT, idT, devT in collapsed:
        # when len = 1, pre and pre2 is same
        if len(refined_units) == 1:
            pre_start, pre_end = refined_units[-1][0], refined_units[-1][1]
            pre2_start, pre2_end = pre_start, pre_end
        else:
            pre_start, pre_end = refined_units[-1][0], refined_units[-1][1]
            pre2_start, pre2_end = refined_units[-2][0], refined_units[-2][1]

        pre_len  = pre_end  - pre_start
        pre2_len = pre2_end - pre2_start
        glen    = gend - gstart

        pre_dev  = round(abs(pre_len  - mode) / mode, 4)
        pre2_dev = round(abs(pre2_len - mode) / mode, 4)
        gdev    = round(abs(glen    - mode) / mode, 4)
   
        ovlp = min(gend, pre_end) - max(gstart, pre_start)
        shorter = min(glen, pre_len)
        ovlp_prop = ovlp / shorter if ovlp > 0 else 0.0

        if ovlp_prop >= COLLAPSE_OVLP_PROP: #-> 1. adjust pre2 and pre by adjusting same time minimizaing dev2, dev1. branched by len ==1 or 2
            print(f"")
            if len(refined_units) == 1 or gstart <= pre2_start:
                if gdev < pre_dev: # update by new alignment g-coordinates
                    refined_units[-1] = (gstart, gend, glen, alenT, qlenT, effT, covT, idT)
                    print(f"[INFO] OVLP-REFINE: len=1: replace last unit {pre_start}-{pre_end} (dev={pre_dev}) -> {gstart}-{gend} (dev={gdev})")
                else:
                    print(f"[INFO] PASS: len=1: keep last unit {pre_start}-{pre_end} (dev={pre_dev}) vs {gstart}-{gend} (dev={gdev})")
                continue
            elif len(refined_units) >1:
                pre2_glen = gstart - pre2_start
                pre2_gdev = round(abs(pre2_glen - mode) / mode, 4)
                ori_dev_sum = pre2_dev + pre_dev
                new_dev_sum = pre2_gdev + gdev
                if ori_dev_sum > new_dev_sum:
                    refined_units[-2] = (pre2_start, gstart, pre2_glen, *refined_units[-2][3:])
                    refined_units[-1] = (gstart, gend, glen, alenT, qlenT, effT, covT, idT)
                    print(f"[INFO] OVLP-REFINE: len>1: adjust pre2/pre -> {pre2_start}-{gstart}, {gstart}-{gend} (ori={ori_dev_sum:.4f} -> new={new_dev_sum:.4f})")
                else: 
                    print(f"[INFO] OVLP-REFINE: len>1: keep pre2/pre (ori={ori_dev_sum:.4f} <= new={new_dev_sum:.4f})")
                continue
            else: 
                print(f"[WARNING] OVLP-REFINE: something unexpected happen after ovlp prop check")
        elif gstart < pre_start:
            print(f"[INFO] OVLP-REFINE: left-side non-overlap: ignore g={gstart}-{gend} (pre={pre_start}-{pre_end})")
            continue
        elif gstart >= pre_end or pre_start <= gstart < pre_end: #non ovlp && right side (gstart >= pre_end) -> adjust pre end by minimize dev # no need to care pre 2
            pre_glen = gstart - pre_start
            pre_gdev = round(abs(pre_glen - mode) / mode, 4)
            new_dev_sum = pre_gdev + gdev

            glen_pre = gend - pre_end
            gdev_pre = round(abs(glen_pre - mode) / mode, 4)
            new_gdev_sum = gdev_pre + pre_dev
            if new_dev_sum < new_gdev_sum:
                refined_units[-1] = (pre_start, gstart, pre_glen, *refined_units[-1][3:])
                refined_units.append((gstart, gend, glen, alenT, qlenT, effT, covT, idT))
                print(f"[INFO] OVLP-REFINE right-side: cut pre {pre_start}-{gstart} (dev={pre_gdev:.4f}) + keep g {gstart}-{gend} (dev={gdev:.4f}), sum {new_dev_sum:.4f} < {new_gdev_sum:.4f}")
            else:
                refined_units.append((pre_end, gend, glen_pre, alenT, qlenT, effT, covT, idT))
                print(f"[INFO] OVLP-REFINE right-side: keep pre {pre_start}-{pre_end} (dev={pre_dev:.4f}) + tail {pre_end}-{gend} (dev={gdev_pre:.4f}), sum {new_gdev_sum:.4f} <= {new_dev_sum:.4f}")
            continue
        else:
            print(f"[WARNING] OVLP-REFINE out of branch")
    
    return refined_units



def _find_target_rkmer(pos, RKP, mode): # find closest k-mer pos with target position
    target_dist = 1
    cur_pos = pos
    step = 0
    while True:
        i_pos = RKP[cur_pos]
        if i_pos == 0:
            print(f"[INFO] terminate RKP chain: cur_pos={cur_pos}, RKP[cur_pos]=0 (no further identical k-mer)")
            break
            #raise RuntimeError(f"[UNEXPECTED] {cur_pos} -> 0; fix your upstream codes.")
        dist_prop = abs((i_pos - pos) - mode) / mode
        print(f"[INFO] RKP step={step}, cur={cur_pos}, next={i_pos}, dist_prop={dist_prop:.4f}, best={target_dist:.4f}")
        if dist_prop < target_dist:
            target_dist = dist_prop
            cur_pos = i_pos
        else:
            print(f"[INFO] stop chain: next dist_prop >= current best ({dist_prop:.4f} >= {target_dist:.4f})")
            break
        step += 1
    return cur_pos



def prepare_refined_units_for_bed(hit_info, mode, UNIT_MERGE_POWER, paf_text_longq, start, end, secondary_cut, unit_sim_cut, GAP_NEW_UNIT_TOL):
    long_hit_info = {}
    for line in paf_text_longq.splitlines(): # PAF parsing for long (actually same with paf short parsing)
        if not line:
            continue
        cols = line.split("\t")
        tstart, tend = int(cols[7]), int(cols[8])
        nmatch, alen, qlen = int(cols[9]), int(cols[10]), int(cols[1])
        gstart, gend = start + tstart, start + tend
        if gstart + qlen > end:
            overhang = (gstart + qlen) - end
            eff_qlen = qlen - overhang
            if eff_qlen <= 0:
                continue
        else:
            eff_qlen = qlen
        cov_long = round(alen / eff_qlen, 4)
        identity_long = round(nmatch / alen, 4) if alen > 0 else 0.0
        if cov_long < secondary_cut or identity_long < unit_sim_cut:
            continue
        prev = long_hit_info.get(gstart)
        if (prev is None) or (alen > prev[0]):
            long_hit_info[gstart] = (alen, qlen, eff_qlen, gend, cov_long, identity_long)

    sorted_ustarts = sorted(hit_info.keys()) #[gstart] = (alen, qlen, eff_qlen, gend, cov, identity)
    refined_units = []
    num_units = len(sorted_ustarts)

    for i, u_start in enumerate(sorted_ustarts):
        alen, qlen, eff_qlen, gend_original, cov, identity = hit_info[u_start]
        if i < num_units - 1:
            next_start = sorted_ustarts[i + 1]
            gap = next_start - gend_original
            #if gend_original >= next_start:
            #    u_end = next_start
            if gap <= mode * UNIT_MERGE_POWER:
                u_end = next_start
            else:
                alt = long_hit_info.get(u_start)
                if alt is not None:
                    alenL, qlenL, eff_qlenL, gend_long, covL, identityL = alt
                    gap_long = next_start - gend_long
                    if gap_long <= mode * UNIT_MERGE_POWER:
                        u_end = next_start
                        alen, qlen, eff_qlen, cov, identity = alenL, qlenL, eff_qlenL, covL, identityL
                    else:
                        if abs(gap - mode) / mode < GAP_NEW_UNIT_TOL:
                            gap_start, gap_end = gend_original, next_start
                            gap_unit_len = gap_end - gap_start
                            print(f"[WARNING] (short/long alignment evidence fail: imputation) gap_start={gap_start}, gap_end={gap_end}, unit_len={gap_unit_len}, mode={mode}, {round((gap_unit_len - mode) / mode, 4)}")
                            refined_units.append((gap_start, gap_end, gap_unit_len, 0, 0, 0, 0.0, 0.0))
                            u_end = gap_start
                            print(f"[WARNING] u_end={u_end} just estimated by start to next start")
                        else:
                            u_end = int(u_start + mode)
                            print(f"[WARNING] u_end={u_end} (short/long gap>{mode*UNIT_MERGE_POWER:.1f})")

                else:
                    if abs(gap - mode) / mode < GAP_NEW_UNIT_TOL:
                        gap_start, gap_end = gend_original, next_start
                        gap_unit_len = gap_end - gap_start
                        print(f"[WARNING] (short/long alignment evidence fail: imputation) gap_start={gap_start}, gap_end={gap_end}, unit_len={gap_unit_len}, mode={mode}, {round((gap_unit_len - mode) / mode, 4)}")
                        refined_units.append((gap_start, gap_end, gap_unit_len, 0, 0, 0, 0.0, 0.0))
                        u_end = gap_start
                        print(f"[WARNING] u_end={u_end} just estimated by start to next start")
                    else:
                        u_end = int(u_start + mode)
                        print(f"[WARNING] u_end={u_end} just estimated by start + mode")
                        #u_end = _find_target_rkmer(pos, RKP, mode)
        else:
            u_end = gend_original

        unit_len = u_end - u_start
        print(f"[UNIT-REFINE] u_start={u_start}, u_end={u_end}, unit_len={unit_len}, cov={cov:.4f}, sim={identity:.4f}")
        refined_units.append((u_start, u_end, unit_len, alen, qlen, eff_qlen, cov, identity))

    refined_units.sort(key=lambda x: x[0])
    return refined_units


def main():
    parser = argparse.ArgumentParser(description="Screen k-mer signal regions based on HN and Jaccard thresholds")
    parser.add_argument("-i", "--input", required=True, help="Input k-mer distance file (no header, TSV)")
    parser.add_argument("-o", "--output", help="Output BED file path (optional; default: <input>.arrays.bed)")
    parser.add_argument("-f", "--fasta", required=True, help="Reference FASTA file for unit alignment")
    parser.add_argument("-d", "--hn_detect", type=float, default=25, help="HN detect threshold (default: 25)")
    parser.add_argument("-j", "--jacu_detect", type=float, default=0.8, help="Jaccard detect threshold (default: 0.8)")
    parser.add_argument("-D", "--hn_elon", type=float, default=50, help="HN elongation threshold (default: 50)")
    parser.add_argument("-J", "--jacu_elon", type=float, default=0.5, help="Jaccard elongation threshold (default: 0.5)")
    parser.add_argument("-w", "--win_ter", type=int, default=1000, help="Window size used in similarity. Used for boundary buffer as well (default: 1000)")
    parser.add_argument("-l", "--len_ter_cut", type=int, default=200, help="Minimum continuous drop length for termination (default: 200)")
    parser.add_argument("-t", "--jacu_ter_soft", type=float, default=0.3, help="Termination Jaccard continuous drop threshold (default: 0.3)")
    parser.add_argument("-H", "--jacu_ter_hard", type=float, default=0.5, help="Termination hard cutoff for minimum Jaccard (default: 0.5)")
    parser.add_argument("-e", "--eps_up", type=float, default=0.02, help="Epsilon up threshold for ending drop (default: 0.02)")
    parser.add_argument("-m", "--min_size", type=int, default=500, help="Minimum robust array (tail[~ -w * 2] not included) length to keep (default: 500)")
    parser.add_argument("-p", "--jacu_sep_hard", type=float, default=0.1, help="Unit seperation hard cutoff for minimum Jaccard (maxFR) (default: 0.1)")
    parser.add_argument("-c", "--secondary_cut", type=float, default=0.7, help="Minimum alignment coverage ratio (alen/qlen) to keep secondary hits for unit identification (default: 0.7)")
    parser.add_argument("-M", "--unit_sim_cut", type=float, default=0.6, help="Minimum alignment similarity to keep hits for unit identification (default: 0.6)")
    parser.add_argument("-A", "--arr_mer_cut", type=float, default=0.5, help="Minimum proportion of k-mer overlap to merge arrays (default: 0.5)")
    parser.add_argument("-a", "--arr_seg_cut", type=float, default=0.1, help="Minimum proportion of k-mer overlap to segregate arrays (default: 0.1)")

    args = parser.parse_args()

    # --- Load table ---
    df = load_kmer_table(args.input)
    #df.set_index("Start", drop=False, inplace=True)
    print(df.head())
    n_valid = len(df) - (args.win_ter * 2)
    print(df.iloc[max(0, n_valid - 10): n_valid])   # n_valid 바로 위 10줄 정도 보기


    # --- Run screening ---
    arrays = whole_screening(
        df,
        hn_detect=args.hn_detect,
        jacu_detect=args.jacu_detect,
        hn_elon=args.hn_elon,
        jacu_elon=args.jacu_elon,
        win_ter=args.win_ter,
        len_ter_cut=args.len_ter_cut,
        jacu_ter_soft=args.jacu_ter_soft,
        jacu_ter_hard=args.jacu_ter_hard,
        eps_up=args.eps_up,
        min_size=args.min_size,
        jacu_sep_hard=args.jacu_sep_hard,
        arr_mer_cut=args.arr_mer_cut,
        arr_seg_cut=args.arr_seg_cut,
    )

    print(f"\n[INFO] Detected arrays: {len(arrays)}")
    # --- Determine output path ---
    if arrays:
        seqname = str(df["SeqName"].iloc[0])
        units_by_array = get_units(                # NEW
            arrays=arrays,
            chrom=seqname,
            fasta_path=args.fasta,
            secondary_cut=args.secondary_cut,
            unit_sim_cut=args.unit_sim_cut,
            df=df,
        )
    else:
        units_by_array = {}                        # NEW
     
    if args.output:
        bed_path = args.output
    else:
        bed_path = args.input + ".arrays.bed"

    if arrays:
        seqname = str(df["SeqName"].iloc[0])
        with open(bed_path, "w") as bed:
            array_idx = 0
            for (start, end), meta in arrays.items():
                units = units_by_array.get((start, end), [])
                num_units = len(units)
                unit_len = end - start
                length, mean_dist, arr_ratio, mode_dist, mode_prop = meta
                mode = mode_dist

                if num_units < 2:
                    if num_units == 0 and mode is not None and unit_len > mode * 2.5:
                        print(f"[KEEP-LONG] array {seqname}:{start}-{end} (unit_n=0, len={unit_len}, mode={mode})")
                    else:
                        print(f"[SKIP] array {seqname}:{start}-{end} skipped (unit_n={num_units}, len={unit_len}, mode={mode})")
                        continue

                array_idx += 1
                array_name = f"a{array_idx}"
                bed.write(f"{seqname}\t{start}\t{end}\t{array_name};length={length};unit_n={num_units};mean_u-len={round(length / num_units, 1) if num_units else 'NA'};mean_k-dist(Q1-Q3)={mean_dist:.1f};mode={mode_dist};mode_prop={mode_prop:.4f}\n")
                for unit_idx, (u_start, u_end, unit_len, alen, qlen, eff_qlen, cov, identity) in enumerate(units, start=1):
                    unit_name = f"a{array_idx}.u{unit_idx}"
                    bed.write(f"{seqname}\t{u_start}\t{u_end}\t{unit_name};len={unit_len};cov={cov if cov is not None else 'NA'};sim={identity:.4f};alen={alen};qlen={qlen};eff_qlen={eff_qlen}\n")
                if num_units > 0:
                    last_u_end = units[-1][1]
                    if last_u_end < end:
                        tail_name = f"a{array_idx}.ut"
                        tail_len = end - last_u_end
                        bed.write(f"{seqname}\t{last_u_end}\t{end}\t{tail_name};len={tail_len};cov=NA;sim=NA;alen=NA;qlen=NA;eff_qlen=NA\n") 
        print(f"[INFO] BED file written: {bed_path}")         
    else:
        print("[INFO] No arrays detected; BED file not written.")

if __name__ == "__main__":
    main()
