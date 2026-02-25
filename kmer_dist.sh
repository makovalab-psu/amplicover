#!/bin/bash
set -euo pipefail

THREADS=4
MEM="20G"

while getopts "t:m:h" opt; do
    case $opt in
        t) THREADS="$OPTARG" ;;
        m) MEM="$OPTARG" ;;
	h) echo "Usage: $0 [-t threads (default: 4)] [-m memory (default: 20G)] <fasta> <.meryl/ db> <output>"
           exit 0 ;;
        *) exit 1 ;;
    esac
done
shift $((OPTIND-1))

if [ $# -lt 3 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 [-t threads (default: 4)] [-m memory (default: 20G)] <fasta> <.meryl/ db> <output>"
    exit 1
fi


INPUT_FASTA="$1"
KDB="$2"
OUTPUT="$3"

WORKDIR=$(dirname "$OUTPUT")
TMPDIR="$WORKDIR/tmp_sort_$$"
mkdir -p "$TMPDIR"
TDB="$WORKDIR/tdb_$$.txt"

trap "rm -rf $TMPDIR $TDB $TDB.F.sorted $TDB.R.sorted" EXIT

# 1. meryl-lookup
echo "[$(date)] meryl-lookup"
meryl-lookup -threads "$THREADS" -memory "$MEM" -dump -sequence "$INPUT_FASTA" -mers "$KDB/" | \
  awk -F"\t" '{print $1"\t"$3"\t"$3+1"\t"$6+$8"\t"$5"\t"$7}' > "$TDB"

# 2. Forward
echo "[$(date)] Sorting Forward"
LC_ALL=C sort -k5,5 -k2,2n --parallel="$THREADS" -S "$MEM" -T "$TMPDIR" "$TDB" | \
  awk 'BEGIN{OFS="\t"} {
    if($5==p) {dist=$2-pp; print $1,$2,$3,$4,$5,$6,pp,dist,sqrt(dist),log(dist)}
    else {print $1,$2,$3,$4,$5,$6,0,0,0,999999} 
    p=$5; pp=$2}' | \
  LC_ALL=C sort -k2,2n --parallel="$THREADS" -S "$MEM" -T "$TMPDIR" > "$TDB.F.sorted"

# 3. Reverse
echo "[$(date)] Sorting Reverse"
LC_ALL=C sort -k5,5 -k2,2nr --parallel="$THREADS" -S "$MEM" -T "$TMPDIR" "$TDB" | \
  awk 'BEGIN{OFS="\t"} {
    if($5==p) {dist=pp-$2; print $1,$2,$3,$4,$5,$6,pp,dist,sqrt(dist),log(dist)}
    else {print $1,$2,$3,$4,$5,$6,0,0,0,999999} 
    p=$5; pp=$2}' | \
  LC_ALL=C sort -k2,2n --parallel="$THREADS" -S "$MEM" -T "$TMPDIR" > "$TDB.R.sorted"

# 4. Join
echo "[$(date)] Join + min/max"
paste "$TDB.F.sorted" "$TDB.R.sorted" | awk -F "\t" '{print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10"\t"$17"\t"$18"\t"$19"\t"$20}' | awk -F"\t" 'BEGIN{OFS="\t"}{min=($8 == 0 && $12 == 0? 0: ($8 == 0? $12: ($12 == 0? $8: ($8 < $12? $8 : $12)))); max=($8 == 0 && $12 == 0? 0: ($8 == 0? $8 : ($12 == 0? $12: ($8 > $12? $8 : $12)))); print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,min,max}' > "$OUTPUT"

# 5. Amplicon DB
echo "[$(date)] Amplicon db consctruction..."
awk -F "\t" '{print $1"\t"$2"\t"$4"\t"$5}' "$OUTPUT" | sort -k4,4 -k2,2n --parallel="$THREADS" -S "$MEM" -T "$TMPDIR" > ${OUTPUT}_sortKF

echo "[$(date)] Done: $OUTPUT"

