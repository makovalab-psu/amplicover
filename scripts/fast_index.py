#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import math
import csv
from numba import njit
from numba.typed import Dict
from numba import types

@njit
def compute_metrics_numba(distances, W, shannon_out, run_out, geary_out, neff_out):
    n = distances.shape[0]
    if n == 0:
        return

    # frequency map for Shannon
    freq = Dict.empty(key_type=types.int64, value_type=types.int64)

    # state for incremental updates
    H = 0.0
    rc = 0
    S = 0.0
    SS = 0.0
    N = 0.0  # adjacency squared diff sum

    # initial full window at i == W-1
    if n >= W:
        # build initial freq and compute H
        for j in range(W):
            v = distances[j]
            if v in freq:
                freq[v] += 1
            else:
                freq[v] = 1

        H = 0.0
        for key in freq:
            c = freq[key]
            if key == 0:
                H -= c * (1.0 / W) * math.log(1.0 / W)
            else:
                p = c / W
                H -= p * math.log(p)

        # initial run count
        rc = 1
        for j in range(1, W):
            if distances[j] == 0: rc += 1
            else:
                if distances[j] != distances[j-1]:
                    rc += 1

        # initial Geary components
        for j in range(W):
            x = distances[j]
            S += x
            SS += x * x
        for j in range(W-1):
            d = distances[j] - distances[j+1]
            N += d * d

        shannon_out[W-1] = H
        run_out[W-1] = rc
        var_sum = SS - (S * S) / W
        geary_out[W-1] = N / (2 * var_sum) if var_sum != 0.0 else 0.0
        neff_out[W-1] = math.exp(H)

        # slide forward incrementally
        for i in range(W, n):
            old = distances[i - W]
            new = distances[i]

            # --- Shannon incremental ---
            if old != new:
                # remove old
                ca = freq[old]
                if old == 0:
                    H -= (1.0 / W) * math.log(W)
                    if ca == 1: del freq[old]
                    else:           freq[old] = ca - 1
                else: 
                    old_contrib = - (ca / W) * math.log(ca / W)
                    if ca == 1:
                        new_contrib = 0.0
                        del freq[old]
                    else:
                        freq[old] = ca - 1
                        p_new = (ca - 1) / W
                        new_contrib = - p_new * math.log(p_new)
                    H += (new_contrib - old_contrib)

                # add new
                if new in freq:
                    cb = freq[new]
                    if new == 0:
                        H += (1.0 / W) * math.log(W)
                        freq[new] = cb + 1
                    else:
                        old_contrib_b = - (cb / W) * math.log(cb / W)
                        freq[new] = cb + 1
                        p_new_b = (cb + 1) / W
                        new_contrib_b = - p_new_b * math.log(p_new_b)
                        H += (new_contrib_b - old_contrib_b)
                else:
                    if new == 0:
                        H += (1.0 / W) * math.log(W)
                    else:
                        new_contrib_b = - (1 / W) * math.log(1 / W)
                        H += new_contrib_b
                    freq[new] = 1
            # else old == new => Shannon unchanged

            # --- run count incremental ---
            second = distances[i - W + 1]
            if old != second or (old == 0 and second == 0):
                rc -= 1
            prev_last = distances[i - 1]
            if prev_last != new or (prev_last == 0 and new == 0):
                rc += 1

            # --- Geary’s C incremental ---
            # update S, SS
            S = S - old + new
            SS = SS - old * old + new * new

            # update adjacency N
            if old != second:
                N -= (old - second) * (old - second)
            if prev_last != new:
                N += (prev_last - new) * (prev_last - new)

            var_sum = SS - (S * S) / W
            C = N / (2 * var_sum) if var_sum != 0.0 else 0.0

            # store updated metrics
            shannon_out[i] = H
            run_out[i] = rc
            geary_out[i] = C
            neff_out[i] = math.exp(H)
    else:
        return  # fewer than W points: nothing to fill beyond defaults

def main():
    parser = argparse.ArgumentParser(description="Fast sliding metrics with Numba")
    parser.add_argument("input_tsv", help="Input TSV path")
    parser.add_argument("output_tsv", help="Output TSV path")
    parser.add_argument("-W", "--window_size", type=int, default=1000, help="Sliding window size")
    parser.add_argument("-W2", "--neff_summary_window", type=int, default=1000,
                    help="Window size W2 for neff mean/std summary")
    args = parser.parse_args()

    # Read only 8th column quickly using pandas
    dist_series = pd.read_csv(args.input_tsv, sep="\t", header=None, usecols=[14], dtype=np.int64)[14].to_numpy()

    n = dist_series.shape[0]
    # allocate output arrays
    shannon = np.full(n, np.nan, dtype=np.float64)
    run_cnt = np.full(n, -1, dtype=np.int64)  # -1 as placeholder for NA
    geary = np.full(n, np.nan, dtype=np.float64)
    neff = np.full(n, np.nan, dtype=np.float64)

    # compute
    compute_metrics_numba(dist_series, args.window_size, shannon, run_cnt, geary, neff)

    # write output: append to original TSV
    with open(args.input_tsv, newline='') as fin, open(args.output_tsv, "w", newline='') as fout:
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t", lineterminator='\n')
        shift = args.window_size - 1
        W2 = args.neff_summary_window
        for i, row in enumerate(reader):
            src = i + shift  # 원래 full-window 결과가 있는 인덱스를 당겨옴
            if src < n:
                s = shannon[src]
                r = run_cnt[src]
                g = geary[src]
                neff_val = neff[src]
                
                window = neff[src: src + W2]
                if len(window) < W2:
                    mean_val = np.nan
                    std_val = np.nan
                else: 
                    mean_val = window.mean()
                    std_val = window.std(ddof=1)

                writer.writerow([
                    *row,
                    f"{s:.6f}" if not np.isnan(s) else "NA",
                    str(r) if r >= 0 else "NA",
                    f"{g:.6f}" if not np.isnan(g) else "NA",
                    f"{neff_val:.6f}" if not np.isnan(neff_val) else "NA",
                    f"{mean_val:.6f}" if not np.isnan(mean_val) else "NA",
                    f"{std_val:.6f}" if not np.isnan(std_val) else "NA"
                ])
            else:
                writer.writerow(row + ["NA", "NA", "NA", "NA", "NA", "NA"])

if __name__ == "__main__":
    main()

