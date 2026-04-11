# Amplicover

Amplicon identification tool for vertebrate sex chromosomes based on k-mers, under development in the [Makova Lab](https://github.com/makovalab-psu)

> **Note:** This tool is currently under active development.



## Dependencies
- Python >= 3.13
- [Meryl 1.0](https://github.com/marbl/meryl)
- [minimap2](https://github.com/lh3/minimap2)



### Python packages
```
pandas==2.3.1
numpy==2.2.6
numba==0.61.2
pysam==0.23.3
edlib==1.3.9.post1
tqdm==4.67.1
```



## Installation
```bash
git clone https://github.com/KoByungJune/amplicover.git
cd amplicover
mamba create -n amplicover_env python=3.13 pandas numpy numba pysam edlib tqdm -c conda-forge
mamba activate amplicover_env
```

Meryl 1.0 and minimap2 must be installed separately and available in your `$PATH`.



## Usage

### Step 1: Build Meryl database
```bash
meryl count k=21 output <asm.meryl> <asm.fasta>
```

### Step 2: Compute K-mer distance
```bash
./kmer_dist.sh <asm.fasta> <asm.meryl> [k-dist.tsv]
```

Optional output path for k-dist.tsv; default is the input FASTA directory.


### Step 3: Identify amplicons
```bash
./amplicover.sh [-o output.bed] <k-dist.tsv> <asm.fasta>
```



## Input

- A single-chromosome genome assembly in FASTA format (gap-free; repeat masking not required)

## Output



BED format with two record types:

- **Array records** (`a1`, `a2`, ...): identified sequence arrays with summary statistics (length, unit count, mean unit length, mean k-mer distance (Q1-Q3), mode k-mer distance, proportion of mode k-mer distance)
- **Unit records** (`a2.u1`, `a2.u2`, ...): individual units within each array with alignment metrics (coverage, sequence identity, alignment length, query length, effective query length - trimmed when estimated query exceeds array boundary)



## Citation

If you reference Amplicover, please cite:

> Github repository: https://github.com/makovalab-psu/amplicover
>
> Zenodo DOI: 10.5281/zenodo.19516839



## License
All rights reserved. This software is provided for reference only. Redistribution or modification is not permitted without prior written consent from the authors.



## Contact
byungjune.ko@gmail.com


