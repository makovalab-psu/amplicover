#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SIM_OPTS=""
ARRAY_OPTS=""
OUT_BED=""

usage() {
    cat << EOF
Amplicover - prototype v0.1
Usage: $0 [OPTIONS] <kdist.tsv> <ref.fasta>

Required:
  input_tsv          Input TSV file
  fasta              Reference FASTA file

Similarity and entropy options (optional):
  -W <int>           Window size  [1000]
  -s <float>         Gaussian kernel sigma  [1000]
  -R <float>         Radius multiplier: R = int(r_mult * sigma)  [4.0]

Array calling options (optional):
  -d <float>         HN detect threshold  [25]
  -j <float>         Jaccard detect threshold  [0.8]
  -D <float>         HN elongation threshold  [50]
  -J <float>         Jaccard elongation threshold  [0.5]
  -l <int>           Minimum continuous drop length for termination  [200]
  -t <float>         Termination Jaccard continuous drop threshold  [0.3]
  -H <float>         Termination hard cutoff for minimum Jaccard  [0.5]
  -e <float>         Epsilon up threshold for ending drop  [0.02]
  -m <int>           Minimum robust array (tail[~ -w * 2] not included) length to keep  [500]
  -p <float>         Unit seperation hard cutoff for minimum Jaccard (maxFR)  [0.1]
  -c <float>         Minimum alignment coverage ratio (alen/qlen) to keep secondary hits for unit identification  [0.7]
  -M <float>         Minimum alignment similarity to keep hits for unit identification  [0.6]
  -A <float>         Minimum proportion of k-mer overlap to merge arrays  [0.5]
  -a <float>         Minimum proportion of k-mer overlap to segregate arrays  [0.1]

Output:
  -o <path>          Output BED file path  [<input>_arrays.bed]

Other:
  -h                 Help

Example:
  $0 k-dist.tsv ref.fa
  $0 -W 1000 -s 1000 -R 4.0 -d 25 -j 0.8 -o output.bed input.tsv ref.fa

EOF
    exit 0
}

# Parse options
while getopts "o:W:s:R:d:j:D:J:l:t:H:e:m:p:c:M:A:a:h" opt; do
    case $opt in
	o) OUT_BED="$OPTARG" ;;
        W) SIM_OPTS="$SIM_OPTS -W $OPTARG" ;;
        s) SIM_OPTS="$SIM_OPTS -s $OPTARG" ;;
        R) SIM_OPTS="$SIM_OPTS -R $OPTARG" ;;
        d) ARRAY_OPTS="$ARRAY_OPTS -d $OPTARG" ;;
        j) ARRAY_OPTS="$ARRAY_OPTS -j $OPTARG" ;;
        D) ARRAY_OPTS="$ARRAY_OPTS -D $OPTARG" ;;
        J) ARRAY_OPTS="$ARRAY_OPTS -J $OPTARG" ;;
        l) ARRAY_OPTS="$ARRAY_OPTS -l $OPTARG" ;;
        t) ARRAY_OPTS="$ARRAY_OPTS -t $OPTARG" ;;
        H) ARRAY_OPTS="$ARRAY_OPTS -H $OPTARG" ;;
        e) ARRAY_OPTS="$ARRAY_OPTS -e $OPTARG" ;;
        m) ARRAY_OPTS="$ARRAY_OPTS -m $OPTARG" ;;
        p) ARRAY_OPTS="$ARRAY_OPTS -p $OPTARG" ;;
        c) ARRAY_OPTS="$ARRAY_OPTS -c $OPTARG" ;;
        M) ARRAY_OPTS="$ARRAY_OPTS -M $OPTARG" ;;
        A) ARRAY_OPTS="$ARRAY_OPTS -A $OPTARG" ;;
        a) ARRAY_OPTS="$ARRAY_OPTS -a $OPTARG" ;;
        h) usage ;;
        *) echo "Invalid option. Use -h for help." >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))

# Check required arguments
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments" >&2
    echo "Use -h for help" >&2
    exit 1
fi

if ! command -v minimap2 &> /dev/null; then
    echo "[ERROR] minimap2 not found!" >&2
    echo "Please export PATH of minimap2 binary" >&2
    exit 1
fi

input=$1
fasta=$2

[ -z "$OUT_BED" ] && OUT_BED="${input}_arrays.bed"

WIN=$(echo "$SIM_OPTS" | grep -oP '(?<=-W )[0-9]+' || echo "1000")

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== Amplicover Pipeline Start ====="
echo "  Input: $input"
echo "  FASTA: $fasta"
echo "  Window: $WIN"


# Step 1: Shannon entropy
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1: Shannon-entropy calculation..."
echo "[CMD] python3 $SCRIPT_DIR/scripts/fast_index.py -W $WIN -W2 $WIN $input ${input}_index"
python3 "$SCRIPT_DIR/scripts/fast_index.py" -W $WIN -W2 $WIN $input ${input}_index


# Step 2: Jacu calculation (3 cores-parallel)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2: Soft-Jaccard calculation (parallel)..."

echo "[CMD] python3 $SCRIPT_DIR/scripts/get_similarity.py $SIM_OPTS -I 7 ${input}_index ${input}_forjac"
python3 "$SCRIPT_DIR/scripts/get_similarity.py" $SIM_OPTS -I 7 -P 0 ${input}_index ${input}_forjac &
echo "[CMD] python3 $SCRIPT_DIR/scripts/get_similarity.py $SIM_OPTS -I 11 ${input}_index ${input}_revjac"
python3 "$SCRIPT_DIR/scripts/get_similarity.py" $SIM_OPTS -I 11 -P 1 ${input}_index ${input}_revjac &
echo "[CMD] python3 $SCRIPT_DIR/scripts/get_similarity.py $SIM_OPTS -I 14 ${input}_index ${input}_minjac"
python3 "$SCRIPT_DIR/scripts/get_similarity.py" $SIM_OPTS -I 14 -P 2 ${input}_index ${input}_minjac &
wait


# Step 3: Verify line counts
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 3: Verifying line counts..."
n_index=$(wc -l < ${input}_index)
n_for=$(wc -l < ${input}_forjac)
n_rev=$(wc -l < ${input}_revjac)
n_min=$(wc -l < ${input}_minjac)

if [ "$n_index" != "$n_for" ] || [ "$n_index" != "$n_rev" ] || [ "$n_index" != "$n_min" ]; then
    echo "[ERROR] Line count mismatch!" >&2
    echo "  index: $n_index" >&2
    echo "  forjac: $n_for" >&2
    echo "  revjac: $n_rev" >&2
    echo "  minjac: $n_min" >&2
    #echo "[INFO] Checking /tmp storage:" >&2
    #df -h /tmp >&2
    #echo "[CMD] rm -f /tmp/forjac.$$ /tmp/revjac.$$ /tmp/minjac.$$"
    rm -f ${input}_forjac ${input}_revjac ${input}_minjac
    exit 1
fi
echo "[INFO] Line count verified: $n_index lines"


# Step 4: Merge maxJacu and minJacu
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 4: Merging maxJacu and minJacu..."
paste ${input}_index ${input}_forjac ${input}_revjac ${input}_minjac | \
awk -F'\t' 'BEGIN{OFS="\t"} {
    for(i=1; i<=NF-3; i++) printf "%s%s", $i, OFS
    forjac = $(NF-2)
    revjac = $(NF-1)
    minjac = $NF
    maxjac = (forjac > revjac) ? forjac : revjac
    if (maxjac == 0) maxjac = "0.0"
    if (minjac == 0) minjac = "0.0"
    print maxjac, minjac
}' > ${input}_tdb.tsv

#echo "[CMD] rm -f /tmp/forjac.$$ /tmp/revjac.$$ /tmp/minjac.$$"
rm -f ${input}_forjac ${input}_revjac ${input}_minjac
echo "[INFO] TDB created: ${input}_tdb.tsv"


# Step 5: Array calling
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 5: Array calling..."
echo "[CMD] python3 $SCRIPT_DIR/scripts/get_arrays.py -i ${input}_tdb.tsv -o $OUT_BED -f $fasta -w $WIN $ARRAY_OPTS"
python3 "$SCRIPT_DIR/scripts/get_arrays.py" -i ${input}_tdb.tsv -o $OUT_BED -f $fasta -w $WIN $ARRAY_OPTS 2> "${OUT_BED}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== Pipeline Complete ====="
echo "  Output TDB: ${input}_tdb.tsv"
echo "  Output BED: $OUT_BED"

# Step 6: Amplicon calling



