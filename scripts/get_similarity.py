#!/usr/bin/env python3
import argparse
import pandas as pd
from collections import defaultdict
from numba import njit, types
from numba.typed import Dict
import numpy as np, math
from math import exp
import sys
from tqdm import tqdm
import os
import time

is_tty = sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")
is_batch = not is_tty


def eff_key(val: int, pos: int) -> int:
    return -(pos + 1) if val == 0 else val

# --- kernel helpers ---
def is_zero_key(k: int) -> bool:
    return k < 0

def make_k_table(R: int, sigma: float):
    return [exp(-(d*d)/(2.0*sigma*sigma)) for d in range(R+1)] # using list. not dictionary. len(list) is small

def k_weight(k1: int, k2: int, k_table, R: int) -> float:
    if is_zero_key(k1) or is_zero_key(k2):
        return 0.0
    d = abs(k1 - k2)
    if d > R:
        return 0.0
    return k_table[d]

def debug_cos(A, B, k_table, R, negA=0, negB=0):
    na = 0.0
    nb = 0.0
    dot = 0.0

    for u, au in A.items():
        for j, aj in A.items():
            w = k_weight(u, j, k_table, R)
            if w:
                na += w * au * aj
    na += float(negA)

    for u, bu in B.items():
        for j, bj in B.items():
            w = k_weight(u, j, k_table, R)
            if w:
                nb += w * bu * bj
    nb += float(negB)

    for u, au in A.items():
        for j, bj in B.items():
            w = k_weight(u, j, k_table, R)
            if w:
                dot += w * au * bj

    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def cos_wj_sparse_nonoverlap_incremental_zero_unique_batch(codes: np.ndarray, W: int, StartPos: int, EndPos: int | None, sigma: float, R_mult: float, tqdm_pos: int):
    print("first window cal")
    if EndPos is None:
        EndPos = len(codes)
    n = EndPos - StartPos
    cos_arr = np.full(len(codes), np.nan, dtype=np.float64)
    wj_arr  = np.full(len(codes), np.nan, dtype=np.float64)
    dice_arr = np.full(len(codes), np.nan, dtype=np.float64)
    cos_cov_arr = np.full(len(codes), np.nan, dtype=np.float64)
    cos_neg_arr = np.full(len(codes), np.nan, dtype=np.float64)  # cos_soft * (1 - r_uniq)
    wj_neg_arr  = np.full(len(codes), np.nan, dtype=np.float64)  # soft-Jaccard * (1 - r_uniq)
    cos_hard_arr = np.full(len(codes), np.nan, dtype=np.float64)

    if n < 2*W:
        raise ValueError("Size too small n < 2*Window")

    A = defaultdict(int)  # [0..W-1]
    B = defaultdict(int)  # [W..2W-1]
    negA = 0
    negB = 0
    #coveredA = set()
    #coveredB = set()
    for j in range(StartPos, StartPos + W):
        key = eff_key(int(codes[j]), j)
        if key < 0:  # zero-bin skip
            negA += 1
            continue
        A[key] += 1
    for j in range(StartPos + W, StartPos + 2*W):
        key = eff_key(int(codes[j]), j)
        if key < 0:
            negB += 1
            continue
        B[key] += 1
    
    R = int(R_mult * sigma)
    k_table = make_k_table(R, sigma)
    # soft-cos(x,y) = xTSy / (sqrt(xTSx) * sqrt(xTSy) ) 
    # S=Gaussian kernal weight table
    na = 0.0 # xTSx
    for u, au in A.items(): # k-dists : freq in window 1
        for j, aj in A.items():
            w = k_weight(u, j, k_table, R)
            if w:
                na += w * au * aj
    na += float(negA)

    nb = 0.0 # xTSy
    for u, bu in B.items(): # k-dists : ferq in window 2
        for j, bj in B.items():
            w = k_weight(u, j, k_table, R)
            if w:
                nb += w * bu * bj
    nb += float(negB)

    dot = 0.0 # xTSy
    for u, au in A.items():
        for j, bj in B.items():
            w = k_weight(u, j, k_table, R)
            if w: 
                dot += w * au * bj
                #coveredA.add(u)
                #coveredB.add(j)
    # hard cosine 
    na_hard = 0.0
    for au in A.values():
        na_hard += au * au
    na_hard += float(negA)          

    nb_hard = 0.0
    for bu in B.values():
        nb_hard += bu * bu
    nb_hard += float(negB)

    dot_hard = 0.0
    for u, au in A.items():
        dot_hard += au * B.get(u, 0)  # non-zero key overlap


    # soft jaccard(x,y) = xTSy / (xTSx + yTSy - xTSy)
    den = na + nb - dot

    # hard jaccard initialization
    A_keys = set(A.keys())
    B_keys = set(B.keys()) 
    lenA = len(A_keys)
    lenB = len(B_keys)
    overlap = len(A_keys & B_keys)
    # --- [ADD] kernel coverage counts from first-window dot loop ---

    if na < 0 or nb < 0 or den < 0:
        #den = 1e-12
        print(f"[DEBUG] idx=0, s={StartPos}, na={na:.6f}, nb={nb:.6f}, den={den:.6f}, negA={negA}, negB={negB}, lenA={len(A)}, lenB={len(B)}")
        sys.exit("[FATAL] den<=0 in first window")

    if na == 0 or nb == 0:
        cos_arr[StartPos] = 0.0
        wj_arr[StartPos]  = 0.0
        dice_arr[StartPos] = 0.0
        cos_cov_arr[StartPos] = 0.0
        cos_neg_arr[StartPos] = 0.0
        wj_neg_arr[StartPos]  = 0.0
        cos_hard_arr[StartPos] = 0.0

    else:
        cos_arr[StartPos] = round(dot / (math.sqrt(na) * math.sqrt(nb)), 5)
        wj_arr[StartPos]  = round(dot / den, 5)
        dice_arr[StartPos] = round(2.0 * dot / (na + nb), 5)
        union = lenA + lenB - overlap
        jac_hard = (overlap / union) if union else 0.0
        cos_cov_arr[StartPos] = round((dot / (math.sqrt(na) * math.sqrt(nb))) * jac_hard, 5)
        
        # --- [ADD] kernel-based unique ratio (neg + soft-unique) ---
        rA = negA / W
        rB = negB / W
        r_uniq = 0.5 * (rA + rB)
        cos_neg_arr[StartPos] = round((dot / (math.sqrt(na) * math.sqrt(nb))) * (1.0 - r_uniq), 5)
        wj_neg_arr[StartPos]  = round((dot / den) * (1.0 - r_uniq), 5)
        # -------------------------------------------------------------
        
        if na_hard > 0.0 and nb_hard > 0.0:
            cos_hard_arr[StartPos] = round(dot_hard / (math.sqrt(na_hard) * math.sqrt(nb_hard)), 5)


    print("go to slide")
    last_s = n - 2*W
    pbar = tqdm(range(1, last_s + 1), desc="Sliding windows", unit="bp", mininterval=1.0, dynamic_ncols=True, disable=is_batch, position=tqdm_pos, leave=True)
    last_log = 0.0
    for idx in pbar:
        s = idx - 1 + StartPos
        out_idx = idx + StartPos
        #print(s)
        #print(len(A))
        #print(len(B))
        
        outA_k = eff_key(int(codes[s]),        s)
        inA_k  = eff_key(int(codes[s + W]),    s + W)
        outB_k = eff_key(int(codes[s + W]),    s + W)
        inB_k  = eff_key(int(codes[s + 2*W]),  s + 2*W)
        
        # hard cosine

        deltaA = {} # item:freq dic in window i = out previous first position, in new last position
        if outA_k >= 0:  # zero-bin skip
            deltaA[outA_k] = deltaA.get(outA_k, 0) - 1
        if inA_k >= 0:
            deltaA[inA_k]  = deltaA.get(inA_k,  0) + 1

        deltaB = {} # item:freq dic in window i+1 = out previous first position, in new last position
        if outB_k >= 0:
            deltaB[outB_k] = deltaB.get(outB_k, 0) - 1
        if inB_k >= 0:
            deltaB[inB_k]  = deltaB.get(inB_k,  0) + 1
        
        d_na = 0.0 # na = w* au * aj
        if outA_k < 0:  
            negA -= 1
            d_na -= 1.0
            na_hard -= 1.0 # hard cos update
        if inA_k  < 0:  
            negA += 1
            d_na += 1.0
            na_hard += 1.0
       
        
        # (A + delta A) T S(A+ delta A) = increment
        # = A T SA + A T S delta A + delta A T SA + delta A T S delta A
        # na = A T SA
        # dna = A T S delta A + delta A T SA + delta A T S delta A
        # = (A + delta A) T S(A + delta A) - A T SA
        # = 2A T S delta A + delta A T S delta A
        for j, dj in deltaA.items():
            if dj == 0: # out = in e.g.  s == s +W 
                continue
            for u, au in A.items(): # k-dist:freq dictionary in window 1
                if abs(u - j) > R: continue
                w = k_weight(u, j, k_table, R)
                if w:
                    d_na += 2.0 * w * au * dj # dj has - 1, 0, +1 ( this is delta A. Window move by 1bp.). au is freq of item u in window 1
            for u, du in deltaA.items(): # Soutout + Sinin - 2Soutin = 2 - 2Soutin 
                if du == 0:
                    continue
                if abs(u - j) > R: continue
                w = k_weight(u, j, k_table, R)
                if w:
                    d_na += w * du * dj
        d_nb = 0.0
        if outB_k < 0:
            negB -= 1
            d_nb -= 1.0
            nb_hard -= 1.0
        if inB_k  < 0:
            negB += 1
            d_nb += 1.0
            nb_hard += 1.0
        
        for j, dj in deltaB.items():
            if dj == 0:
                continue
            for u, bu in B.items():
                if abs(u - j) > R: continue
                w = k_weight(u, j, k_table, R)
                if w:
                    d_nb += 2.0 * w * bu * dj
            for u, du in deltaB.items():
                if du == 0:
                    continue
                if abs(u - j) > R: continue
                w = k_weight(u, j, k_table, R)
                if w:
                    d_nb += w * du * dj
        # d_dot = A^T S ΔB + ΔA^T S B + ΔA^T S ΔB
        
        d_dot = 0.0
        
        # A^T S ΔB
        for j, dj in deltaB.items():
            if dj == 0:
                continue
            for u, au in A.items():
                if abs(u - j) > R: continue
                w = k_weight(u, j, k_table, R)
                if w:
                    d_dot += w * au * dj
        # ΔA^T S B
        for u, du in deltaA.items():
            if du == 0:
                continue
            for j, bj in B.items():
                if abs(u - j) > R: continue
                w = k_weight(u, j, k_table, R)
                if w:
                    d_dot += w * du * bj
        # ΔA^T S ΔB
        for u, du in deltaA.items():
            if du == 0:
                continue
            for j, dj in deltaB.items():
                if dj == 0:
                    continue
                if abs(u - j) > R: continue
                w = k_weight(u, j, k_table, R)
                if w:
                    d_dot += w * du * dj
        na  += d_na
        nb  += d_nb
        dot += d_dot
        den = na + nb - dot

        # histogram freq update -2 to +2 and del item if freq == 0
        affected = set(deltaA.keys()) | set(deltaB.keys())
        for k in affected:
            pre_inA = (k in A_keys)     # [ADD for Hard Jac]
            pre_inB = (k in B_keys)     # [ADD for Hard Jac]
            
            # hard cosine
            old_a = A.get(k, 0)
            old_b = B.get(k, 0)

            a_new = A.get(k, 0) + deltaA.get(k, 0) # new freq for affected key
            if a_new == 0:
                if k in A: del A[k]
            else:
                A[k] = a_new

            b_new = B.get(k, 0) + deltaB.get(k, 0)
            if b_new == 0:
                if k in B: del B[k]
            else:
                B[k] = b_new

            # hard cosine
            na_hard  += (a_new * a_new - old_a * old_a)
            nb_hard  += (b_new * b_new - old_b * old_b)
            dot_hard += (a_new * b_new - old_a * old_b)

            post_inA = (a_new > 0)      # [ADD for Hard Jac]
            post_inB = (b_new > 0)      # [ADD for Hard Jac]

            # [ADD] lenA/lenB incremental
            if pre_inA and not post_inA: lenA -= 1      # k in previous A and <= 0 [ADD]
            elif (not pre_inA) and post_inA: lenA += 1  # k isn't previous A and >0 (post) [ADD]
            if pre_inB and not post_inB: lenB -= 1      # [ADD]
            elif (not pre_inB) and post_inB: lenB += 1  # [ADD]

            # [ADD] key set 
            if post_inA and not pre_inA: A_keys.add(k)      # [ADD]
            elif pre_inA and not post_inA: A_keys.discard(k)# [ADD]
            if post_inB and not pre_inB: B_keys.add(k)      # [ADD]
            elif pre_inB and not post_inB: B_keys.discard(k)# [ADD]

            # [ADD] intersection incremental
            overlap += ( (1 if (post_inA and post_inB) else 0) - (1 if (pre_inA and pre_inB) else 0) ) # [ADD]


        if na < 0 or nb < 0 or den < 0:
            print(f"[DEBUG] idx={out_idx}, s={s}, na={na:.6f}, nb={nb:.6f}, den={den:.6f}, negA={negA}, negB={negB}, lenA={len(A)}, lenB={len(B)}")
            sys.exit("[FATAL] Negative or zero value encountered -> terminating.")
        
        if na == 0 or nb == 0:
            cos_arr[out_idx] = 0.0
            wj_arr[out_idx]  = 0.0
            dice_arr[out_idx] = 0.0
            cos_cov_arr[out_idx] = 0.0
            cos_neg_arr[out_idx] = 0.0
            wj_neg_arr[out_idx]  = 0.0
            cos_hard_arr[out_idx] = 0.0

        else:
            cos_arr[out_idx] = round(dot / (math.sqrt(na) * math.sqrt(nb)), 5)
            wj_arr[out_idx]  = round(dot / den, 5)
            dice_arr[out_idx] = round(2.0 * dot / (na + nb), 5)
            union = lenA + lenB - overlap
            jac_hard = (overlap / union) if union else 0.0
            cos_cov_arr[out_idx] = round((dot / (math.sqrt(na) * math.sqrt(nb))) * jac_hard, 5)
            
            # --- [ADD] r_uniq from neg + true-unique, then penalized scores ---
            rA = negA / W
            rB = negB / W
            r_uniq = 0.5 * (rA + rB)

            cos_neg_arr[out_idx] = round((dot / (math.sqrt(na) * math.sqrt(nb))) * (1.0 - r_uniq), 5)
            wj_neg_arr[out_idx]  = round((dot / den) * (1.0 - r_uniq), 5)
            # ---------------------------------------------------------------
            # hard cosine
            cos_hard_arr[out_idx] = round(dot_hard / (math.sqrt(na_hard) * math.sqrt(nb_hard)), 5)

        if is_batch:
            if idx % 100000 == 0:
                print(f"[{out_idx}] |A|={len(A)} |B|={len(B)} cos={cos_arr[out_idx]:.4f} wj={wj_arr[out_idx]:.4f} dice={dice_arr[out_idx]:.4f} cov={cos_cov_arr[out_idx]:.4f} cosu={cos_neg_arr[out_idx]:.4f} wju={wj_neg_arr[out_idx]:.4f} hcos={cos_hard_arr[out_idx]:.4f}", flush=True)
        else:
            now = time.time()
            if now - last_log >= 1.0:
                pbar.set_postfix_str(f"|A|={len(A)} |B|={len(B)} cos={cos_arr[out_idx]:.4f} wj={wj_arr[out_idx]:.4f} dice={dice_arr[out_idx]:.4f} cov={cos_cov_arr[out_idx]:.4f} cosu={cos_neg_arr[out_idx]:.4f} wju={wj_neg_arr[out_idx]:.4f} hcos={cos_hard_arr[out_idx]:.4f}", refresh=False)
                last_log = now

    return cos_arr, wj_arr, dice_arr, cos_cov_arr, cos_neg_arr, wj_neg_arr, cos_hard_arr


def weighted_jaccard_cosine_from_itemcol_strict_start(series: pd.Series, W: int, StartPos: int, EndPos: int | None, sigma: float, R_mult: float, tqdm_pos: int):
    if series.isna().any():
        raise ValueError("Item column contains NA/NaN; clean or fill first.")
    codes = series.to_numpy(dtype=np.int64, copy=False)
    return cos_wj_sparse_nonoverlap_incremental_zero_unique_batch(codes, W, StartPos, EndPos, sigma, R_mult, tqdm_pos)


def main0():
    parser = argparse.ArgumentParser(description="Array or/and Amplicon identification")
    parser.add_argument("input_tsv")
    parser.add_argument("output_tsv")
    parser.add_argument("-W","--window_size",type=int,default=1000,help="Sliding window size [%(default)s]")
    parser.add_argument("-S","--start_pos",type=int,default=0,help="Start position for first window [%(default)s]")
    parser.add_argument("-E", "--end_pos", type=int, default=None,help="End position (exclusive) for last window e.g. -E 10000 [default: full length]")
    parser.add_argument("-s","--sigma",type=float,default=1000,help="Gaussian kernel sigma [%(default)s]")
    parser.add_argument("-R","--r_mult",type=float,default=4.0,help="Radius multiplier: R = int(r_mult * sigma) [%(default)s]")
    parser.add_argument("-I","--item_col",type=int,default=14,help="0-based index for item (default 14=15th col) [%(default)s]")
    parser.add_argument("-P", "--tqdm_pos", type=int, default=0, help="tqdm position for parallel execution (default: 0)") 
    args = parser.parse_args()

    df = pd.read_csv(args.input_tsv, sep="\t", header=None, usecols=[args.item_col], dtype={0: "object"})
    item_series = df.iloc[:, 0]

    cos_next, wj_next, dice_next, cov_next, cosu_next, wju_next, coshard_next  = weighted_jaccard_cosine_from_itemcol_strict_start(
        item_series, args.window_size, args.start_pos, args.end_pos, args.sigma, args.r_mult, args.tqdm_pos,
    )
    

    S = args.start_pos
    E = len(df) if args.end_pos is None else args.end_pos
    #df.iloc[S:min(E, len(df))].to_csv(args.output_tsv, sep="\t", header=False, index=False) 
    wjaccu_series = pd.Series(wju_next[S:min(E, len(wju_next))])
    wjaccu_series.to_csv(args.output_tsv, sep="\t", header=False, index=False, na_rep="NA")

if __name__ == "__main__":
    main0()


