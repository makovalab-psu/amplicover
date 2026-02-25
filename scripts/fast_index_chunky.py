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
    import time
    parser = argparse.ArgumentParser(description="Fast sliding metrics (chunked output)")
    parser.add_argument("input_tsv", help="Input TSV path")
    parser.add_argument("output_tsv", help="Output TSV path")
    parser.add_argument("-W", "--window_size", type=int, default=1000, help="Sliding window size")
    parser.add_argument("-W2", "--neff_summary_window", type=int, default=1000,
                    help="Window size W2 for neff mean/std summary")
    parser.add_argument("-C", "--chunksize", type=int, default=1000000,
                    help="Chunk size for output (default: 1M lines)")
    args = parser.parse_args()

    start_time = time.time()

    # 1. 14번째 컬럼만 읽기
    print(f"[{time.strftime('%H:%M:%S')}] Reading distance column...")
    read_start = time.time()
    dist_series = pd.read_csv(args.input_tsv, sep="\t", header=None,
                               usecols=[14], dtype=np.int64)[14].to_numpy()
    read_time = time.time() - read_start
    print(f"[{time.strftime('%H:%M:%S')}] Read complete ({read_time:.1f}s)")

    n = dist_series.shape[0]
    shannon = np.full(n, np.nan, dtype=np.float64)
    run_cnt = np.full(n, -1, dtype=np.int64)
    geary = np.full(n, np.nan, dtype=np.float64)
    neff = np.full(n, np.nan, dtype=np.float64)

    # 2. 계산
    print(f"[{time.strftime('%H:%M:%S')}] Computing metrics...")
    calc_start = time.time()
    compute_metrics_numba(dist_series, args.window_size, shannon, run_cnt, geary, neff)
    calc_time = time.time() - calc_start
    print(f"[{time.strftime('%H:%M:%S')}] Computation complete ({calc_time:.1f}s)")

    # 3. neff window 계산 (Forward-looking!)
    print(f"[{time.strftime('%H:%M:%S')}] Computing neff windows...")
    window_start = time.time()

    shift = args.window_size - 1
    W2 = args.neff_summary_window

    # shift 적용
    neff_shifted = np.roll(neff, -shift)
    shannon_shifted = np.roll(shannon, -shift)
    run_shifted = np.roll(run_cnt, -shift)
    geary_shifted = np.roll(geary, -shift)

    # neff mean/std (Forward-looking)
    neff_mean = np.full(n, np.nan, dtype=np.float64)
    neff_std = np.full(n, np.nan, dtype=np.float64)

    for i in range(n - W2 + 1):
        window = neff_shifted[i:i+W2]
        neff_mean[i] = np.mean(window)
        neff_std[i] = np.std(window, ddof=1)

    window_time = time.time() - window_start
    print(f"[{time.strftime('%H:%M:%S')}] Window computation complete ({window_time:.1f}s)")

    # 4. 출력 (청크 단위)
    print(f"[{time.strftime('%H:%M:%S')}] Writing output in chunks...")
    write_start = time.time()

    first_chunk = True
    chunk_reader = pd.read_csv(args.input_tsv, sep="\t", header=None,
                                chunksize=args.chunksize)

    for chunk_idx, chunk in enumerate(chunk_reader):
        chunk = chunk.reset_index(drop=True)  # 인덱스 리셋

        start_idx = chunk_idx * args.chunksize
        end_idx = start_idx + len(chunk)

        # 결과 추가 (컬럼 번호로 명시)
        chunk[16] = shannon_shifted[start_idx:end_idx]
        chunk[17] = run_shifted[start_idx:end_idx]
        chunk[18] = geary_shifted[start_idx:end_idx]
        chunk[19] = neff_shifted[start_idx:end_idx]
        chunk[20] = neff_mean[start_idx:end_idx]
        chunk[21] = neff_std[start_idx:end_idx]

        # NA 처리
        chunk[16] = chunk[16].apply(
            lambda x: f"{x:.6f}" if not np.isnan(x) else "NA")
        chunk[17] = chunk[17].apply(
            lambda x: str(x) if x >= 0 else "NA")
        chunk[18] = chunk[18].apply(
            lambda x: f"{x:.6f}" if not np.isnan(x) else "NA")
        chunk[19] = chunk[19].apply(
            lambda x: f"{x:.6f}" if not np.isnan(x) else "NA")
        chunk[20] = chunk[20].apply(
            lambda x: f"{x:.6f}" if not np.isnan(x) else "NA")
        chunk[21] = chunk[21].apply(
            lambda x: f"{x:.6f}" if not np.isnan(x) else "NA")

        # 쓰기 (컬럼 순서 명시, lineterminator 지정)
        mode = 'w' if first_chunk else 'a'
        chunk.to_csv(args.output_tsv, sep="\t", header=False,
                     index=False, mode=mode,
                     columns=list(range(22)),  # 0~21 순서대로
                     lineterminator='\n')      # 줄바꿈 명시
        first_chunk = False

        if (chunk_idx + 1) % 10 == 0:
            print(f"  Processed {(chunk_idx + 1) * args.chunksize:,} lines...")

    write_time = time.time() - write_start
    print(f"[{time.strftime('%H:%M:%S')}] Write complete ({write_time:.1f}s)")

    # 요약
    total_time = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Read:        {read_time:.1f}s ({read_time/total_time*100:.1f}%)")
    print(f"Compute:     {calc_time:.1f}s ({calc_time/total_time*100:.1f}%)")
    print(f"Window:      {window_time:.1f}s ({window_time/total_time*100:.1f}%)")
    print(f"Write:       {write_time:.1f}s ({write_time/total_time*100:.1f}%)")
    print(f"Total:       {total_time:.1f}s ({total_time/60:.1f}min)")

if __name__ == "__main__":
    main()
