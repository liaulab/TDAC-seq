# DddA

## 1. Overview

This is the repository containing code to analyze TDAC-seq data as well as reproducing results from the TDAC-seq paper. TDAC-seq utilizes a double strand DNA deaminase DddA11 to stencil chromatin fibers. Using nanopore sequencing of the stenciled DNA, we can measure chromatin accessibility and detect protein footprints. TDAC-seq allows readout of genetic perturbations suchs as CRISPR deletions and adenine base editor mutations as well as chromatin accessibility simultaneously on the same single DNA molecules, enabling the functional dissection of regulatory elements.

![TDAC-seq-schematic](TDAC-seq-schematic.png)

## 2. Installation and system requirements

First, you'll need `git` and `mamba`/`conda`, if you don't already have these tools.

Clone or download all the files in this repository. Then change the current directory to the repository folder.
```
git clone https://github.com/liaulab/TDAC-seq/
cd TDAC-seq
```

Install dependencies, which are described in `environment.yml`. This can take up to ~30 minutes to install everything from scratch.
```
mamba env create -f environment.yml -n tdac-seq -y
```

Then activate the environment. Do this everytime before you use the package for the first time:
```
mamba activate tdac-seq
```

Install our software package containing tools to process TDAC-seq data. This should take less than a minute because all dependencies have been installed through mamba above.
```
pip install .
```
Now you can import modules like `from tdac_seq.ddda_dataset import ddda_dataset`. See below for more about how to use these tools. For typical datasets, you'll need up to 512 GB memory, and the only code that is currently configured for multiprocessing is read alignment where we typically use 16 CPUs.

## 3. Tutorial

We have prepared a tutorial to show you how to run the basic functions to analyze TDAC-seq data! In this tutorial, we test run on a small dataset where a subset of cells will carry a CRISPR deletion at the target locus. The CRISPR deletion is expected to disrupt a CTCF motif and affect local chromatin organization. Here we will first align reads, call DddA mutations, detect CRISPR deletions, and then analyze the effect of deletions on chromatin accessibility and footprints. We will also demonstrate the expected input and output format.

The tutorial can be found here https://github.com/liaulab/TDAC-seq/tree/main/analysis/tutorial/tutorial.ipynb

The tutorial data can be downloaded from https://zenodo.org/records/14659105?preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTczNjk5NjY4NiwiZXhwIjoyMDUxMjIyMzk5fQ.eyJpZCI6IjFmZmExYmI2LWEzMjktNDgwNS04YmQzLTJlYmE3ODA3ZTE1MCIsImRhdGEiOnt9LCJyYW5kb20iOiIyMjBhZjBjZGIwN2M0NDhjYzJiN2U2N2QxOWEyY2VmOCJ9.YVfrGNDcczYj6gNssUTV1KiqshVSGYGsetcZYG4Y6ZNa7VVQw0RIOzJIS6oeIkodWnQNmw_XCJjCWxTnvatHBw

## 4. Reproducing results

### Below are the notebooks for reproducing results in our paper:

dCas9 footprints: https://github.com/liaulab/TDAC-seq/tree/main/analysis/HJR231_dCas9

cCRE dynamics during CD34 differentiation: https://github.com/liaulab/TDAC-seq/tree/main/analysis/HJR234_CD34_differentiation

Examine effects of CRISPR deletion at a CTCF motif (single sgRNA): https://github.com/liaulab/TDAC-seq/tree/main/analysis/HJR235_CTCF_editing

CRISPR and ABE screen at HS2, as well as deaminase-free controls: https://github.com/liaulab/TDAC-seq/tree/main/analysis/HJR244_CRISPR_screen

Calculating DddA sequence bias using DddA treatment on naked DNA: https://github.com/liaulab/TDAC-seq/tree/main/analysis/HJR245_naked_DNA

Example TDAC-seq regions: https://github.com/liaulab/TDAC-seq/tree/main/analysis/HJR250_TDACseq

DddA11-whole genome sequencing and aggregate footprints: https://github.com/liaulab/TDAC-seq/tree/main/analysis/HJR253_DddA_WGS

HS2 single sgRNA bulk ATAC validation: https://github.com/liaulab/TDAC-seq/tree/main/analysis/HJR269_ATAC

HBG bulk ATAC tracks visualization: https://github.com/HYsxe/TDAC_seq/blob/main/analysis/HJR288_HBG/diffATAC_HJR288.ipynb

Dissecting the effect of CTCF and YY1 in the HOXA locus: https://github.com/liaulab/TDAC-seq/tree/main/analysis/HJR309_CTCF_YY1

GFI1B screen sgRNA target sites: https://github.com/liaulab/TDAC-seq/tree/main/analysis/HJR312_GFI1B_screen

MGYFPDa829 CD34 differentiation: https://github.com/liaulab/TDAC-seq/tree/main/analysis/HJR313_MGYFPDa829_CD34

Comparing MGYFPDa829 and DddA11 at different loci: https://github.com/liaulab/TDAC-seq/tree/main/analysis/HJR321_MGYFPDa829_benchmark

Benchmarking Cas9-based target locus enrichment efficiency with TDAC-seq: https://github.com/liaulab/TDAC-seq/tree/main/analysis/Locus_enrich_benchmark

Whole-genome benchmark of GpC-based accessibility measurements with TDAC-seq: https://github.com/liaulab/TDAC-seq/tree/main/analysis/SMF_benchmark

Basic unit tests of the code: https://github.com/liaulab/TDAC-seq/tree/main/analysis/test_cases

### Below are Python scripts for reproducing results in our paper:

First, to pre-process the data (i.e. align and deduplicate reads), make sure the `fastq_file` paths in [`data/datasets.tsv`](data/datasets.tsv) have been filled out correctly. Then run [`analysis/align_all.py`](analysis/align_all.py) and then [`analysis/dedup_all.py`](analysis/dedup_all.py). These will output `ddda_data.pkl` files.
- Co-accessibility of DADs (Fig. 1e, Extended Data Fig. 1k): [`analysis/coactuation.py`](analysis/coactuation.py)
- Overlap between DADs and ATAC peaks (Fig. 1f-g, Extended Data Fig. 1g & 1j): [`analysis/dads.py`](analysis/dads.py) and [`analysis/dads_roc.py`](analysis/dads_roc.py)
- HOXA accessibility by CRISPR deletions covering CTCF and/or YY1 (Fig. 2f): [`analysis/hoxa.py`](analysis/hoxa.py)
- TDAC-seq at HBG locus treated with sgRNA68 (Fig. 3f, Extended Data Fig. 3d-e): [`analysis/sgrna68.py`](analysis/sgrna68.py)
- HS2 and GFI1B screen heatmap, allele table, boxplots, etc. (Figs. 4-6 and Extended Data Figs. 4-6): [`analysis/hs2.py`](analysis/hs2.py)
- TDAC-seq condition optimization (Extended Data Fig. 1h, 6a): [`analysis/reps_ddda_conditions.py`](analysis/reps_ddda_conditions.py)
- UMI validation by CRISPR mutation (Extended Data Fig. 4c): [`analysis/dedup_validation.py`](analysis/dedup_validation.py)
- Overlay of TDAC-seq replicates (Supplementary Fig. 1): [`analysis/correlate_reps_hjr250.py`](analysis/correlate_reps_hjr250.py)
- Scatter plot of CRISPR screen reps (Supplementary Fig. 3): [`analysis/gfi1b_reps.py`](analysis/gfi1b_reps.py)


## 5. Contact

If you have any questions for us, please open an issue in this repo! Thank you for contributing!
