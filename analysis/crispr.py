import pandas as pd
from tdac_seq.ddda_dataset import ddda_dataset
import re
import itertools
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import sys
from tdac_seq.refseq import extract_ref_seq, regions_df_to_dict
from tdac_seq.crispr import find_cutsite
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors
import mappy
import os
import multiprocessing as mp
import pickle
import itertools
import tempfile
from tdac_seq.utils import get_fastq_iter, parse_cigar, ref_bp_consumed
from tdac_seq.refseq import extract_ref_seq
import pyBigWig

def align_helper(args):
    reads_batch, ref_seq, start_gap_threshold, end_gap_threshold, ref_fasta_fname = args

    aligner = mappy.Aligner(ref_fasta_fname, preset="map-ont")

    out = ""

    for read_id, read_seq, _ in reads_batch:

        # Align the current read
        alignments = [a for a in aligner.map(read_seq, cs=True)]
        
        # Remove multi-mappers and unmapped reads
        if len(alignments) == 0 or len(alignments) > 1:
            continue
        else:
            alignment = alignments[0]
        
        # Remove reads with large gaps at the beginning or end
        if alignment.r_st >= start_gap_threshold:
            continue
        if alignment.r_en <= len(ref_seq) - end_gap_threshold:
            continue
        
        cur_pos = alignment.r_st
        del_sofar = []
        num_ct_edits = 0
        num_ga_edits = 0
        for cs_op, cs_arg in parse_cigar(alignment.cs):
            if cs_op == "-" and len(cs_arg) >= 5:
                del_sofar.append((cur_pos, len(cs_arg)))
            elif cs_op == "*":
                edit_from, edit_to = cs_arg
                if edit_from == "c" and edit_to == "t":
                    num_ct_edits += 1
                elif edit_from == "g" and edit_to == "a":
                    num_ga_edits += 1
            cur_pos += ref_bp_consumed(cs_op, cs_arg)
        if len(del_sofar) == 0:
            del_start = 0
            del_len = 0
        elif len(del_sofar) > 1:
            continue
        else:
            del_start, del_len = del_sofar[0]

        out += f"{read_id}\t{del_start}\t{del_len}\t{num_ct_edits}\t{num_ga_edits}\n"

    return out

def align_reads(aln_fname, locus, ref_seq, fastq_file, total_reads, start_gap_threshold, end_gap_threshold):
    with open(aln_fname, "wt") as aln_file, mp.Pool(mp.cpu_count()) as pool, tempfile.NamedTemporaryFile("w+t") as ref_fasta:
        
        ref_fasta.write(f">{locus}\n{ref_seq}\n")
        ref_fasta.flush()

        for out in pool.imap(
                align_helper,
                ((reads_batch, ref_seq, start_gap_threshold, end_gap_threshold, ref_fasta.name) for reads_batch in itertools.batched(tqdm(get_fastq_iter(fastq_file), total=total_reads), 10000)),
                ):
            if out is not None:
                aln_file.write(out)

def main(dataset_id: int, start_gap_threshold: int = 500, end_gap_threshold: int = 500):
    parent_working_dir_prefix = "output/"

    regions = pd.read_csv('data/regions.tsv', sep='\t', index_col="name")
    datasets = pd.read_csv('data/datasets.tsv', sep='\t', index_col="ID")
    datasets['regions'] = datasets['regions'].str.split(',')
    datasets['working_dir'] = parent_working_dir_prefix + datasets.index
    def parse_guides(input: str) -> pd.Series:
        if input == 'none':
            return pd.Series(name='spacer')
        elif re.match(r'^[ACGT]{20}$', input):
            return pd.Series([input.upper()], index=['sgRNA'], name='spacer')
        elif input.startswith('!'):
            # get library from tsv file
            out = pd.read_csv(input[1:], sep='\t', index_col=0, header=None).squeeze('columns').str.upper()
            out.rename('spacer', inplace=True)
            return out

    assert set(itertools.chain.from_iterable(datasets['regions'])).issubset(regions.index), "Some regions are not in the regions.tsv file"
    if not os.path.exists('data/ref.fa'):
        extract_ref_seq(genome_file='/n/holystore01/LABS/liau_lab/Users/sps/genomes/hg38_ucsc/hg38.fa', region_dict=regions_df_to_dict(regions), out_fasta_path='data/ref.fa')

    data_info = datasets.iloc[dataset_id].copy()
    id = data_info.name
    if pd.isna(data_info['num_reads']):
        total_reads = None
    else:
        total_reads = int(data_info['num_reads'])
    plot_dir = f"plots/{id}/del"
    os.makedirs(plot_dir, exist_ok=True)
    working_dir = os.path.join(data_info['working_dir'], "del")
    os.makedirs(working_dir, exist_ok=True)

    logging.info(f"Processing dataset {id}")

    object_fname = os.path.join(data_info['working_dir'], "ddda_data.pkl")
    if not os.path.exists(object_fname):
        logging.error(f"Pre-aligned data not found: {object_fname}. Skipping dataset {id}")
        return

    logging.info(f"Loading pre-aligned data from {object_fname}")
    with open(object_fname, "rb") as f:
        ddda_data: ddda_dataset = pickle.load(f)

    assert len(ddda_data.ref_seq_dict) == 1
    locus = list(ddda_data.ref_seq_dict.keys())[0]

    # Load sgRNA sequences
    guides = parse_guides(data_info['sgRNA'])

    # Get sequence of the target locus
    ref_seq = ddda_data.ref_seq_dict[locus]
    seq_len = len(ref_seq)

    # Re-align to get full raw CRISPR info
    aln_fname = os.path.join(working_dir, "aln.tsv")
    if not os.path.exists(aln_fname):
        if os.path.exists(os.path.join(plot_dir, "aln.tsv")):
            logging.warning(f"Moving {plot_dir}/aln.tsv to {aln_fname}")
            os.rename(os.path.join(plot_dir, "aln.tsv"), aln_fname)
        else:
            logging.info("Aligning reads")
            align_reads(aln_fname, locus, ref_seq, data_info['fastq_file'], total_reads, start_gap_threshold, end_gap_threshold)

    # Find sgRNA positions in the target locus
    target_sites = guides.apply(lambda x: find_cutsite(ref_seq, x))

    MAX_DEL_LENGTH = 30
    logging.info("Calculating scores")
    scores_sum = np.zeros((seq_len, MAX_DEL_LENGTH), dtype=int) # sequence length, deletion length. coordinates are start of deletion; sum at first, then divide by count to get mean
    counts = np.zeros((seq_len, MAX_DEL_LENGTH), dtype=int)
    with open(aln_fname, "rt") as f:
        for line in tqdm(f):
            read_id, start, length, num_ct_edits, num_ga_edits = line.strip().split('\t')
            start, length, num_ct_edits, num_ga_edits = map(int, (start, length, num_ct_edits, num_ga_edits))
            if length >= MAX_DEL_LENGTH:
                continue
            counts[start, length] += 1
            if num_ct_edits > num_ga_edits:
                scores_sum[start, length] += num_ct_edits
            else:
                scores_sum[start, length] += num_ga_edits
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = scores_sum / counts

    score_in_window = np.zeros_like(scores)

    if locus == "LCR_new":
        windows = [slice(1470, 1510), slice(1100, 1800)]
        minimal_motif = slice(1488, 1499)
        min_coverage = 100
    elif locus == "MYC":
        windows = [slice(700, 1300), slice(1900, 3300)]
        minimal_motif = None
        min_coverage = 50
    elif locus == "ZBTB38":
        windows = [slice(500, 1000), slice(1900, 2800), slice(3800, 4300)]
        minimal_motif = None
        min_coverage = 50
    elif locus == "HOXA":
        windows = [slice(2600, 2675)]
        minimal_motif = slice(27161134 - 27158522, 27161152 - 27158522 + 1)
        if id == 'HJR235_4':
            min_coverage = 200
        else:
            min_coverage = 2
    else:
        windows = []
        minimal_motif = None
        min_coverage = 100
        logging.warning("No windows defined for this locus")

    # Visualize del coverage and sgRNA positions
    fig, (ax, cax) = plt.subplots(figsize=(100, 3), ncols=2, gridspec_kw=dict(width_ratios=[100, 0.1], wspace=0.01))
    counts_plot = counts.copy().astype(float)
    counts_plot[:, 0] = np.nan
    cmap = plt.cm.viridis_r
    norm = matplotlib.colors.LogNorm()
    x = np.arange(seq_len)
    ax.pcolormesh(x, np.arange(MAX_DEL_LENGTH), counts_plot.T, cmap=cmap, norm=norm)
    ax.invert_yaxis()
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label="Num reads containing deletion", cax=cax)
    ax.set_ylabel("Length of CRISPR deletion")
    ax.set_xlabel("Start of CRISPR deletion")
    ax.set_xticks(target_sites - 0.5, minor=True, labels=target_sites.index.to_series().apply("sg{}".format), rotation=45, ha='right')
    ax.tick_params(axis='x', which='minor', direction='out', top=False, bottom=True, labelbottom=True, labeltop=False, length=20)
    fig.savefig(os.path.join(plot_dir, "coverage_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(sorted(counts_plot[~np.isnan(counts_plot)], reverse=True))
    ax.axhline(min_coverage, color='red')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Rank of CRISPR deletion genotype")
    ax.set_ylabel("Num reads")
    fig.savefig(os.path.join(plot_dir, "coverage_knee.pdf"), bbox_inches="tight")
    plt.close(fig)

    atac = pyBigWig.open("https://www.encodeproject.org/files/ENCFF357GNC/@@download/ENCFF357GNC.bigWig")
    atac_vals = np.array(atac.values(*ddda_data.region_dict[locus]))

    wt = scores[~np.isnan(scores[:, 0]), 0].mean()
    if np.isnan(wt):
        wt = np.median(scores[~np.isnan(scores)])
    logging.info(f"WT score: {wt}")

    fig, (ax_atac, ax, ax_counts) = plt.subplots(figsize=(100, 4), nrows=3, sharex=True)
    ax_atac.fill_between(np.arange(len(atac_vals)), atac_vals, color='black', alpha=0.5)
    ax_atac.set_ylabel("ATAC-seq", rotation=0, ha='right')
    cmap = plt.cm.coolwarm
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=wt)
    scores_plot = scores.copy() + score_in_window
    scores_plot[:, 0] = np.nan
    scores_plot[counts < min_coverage] = np.nan
    ax.pcolormesh(np.arange(seq_len), np.arange(MAX_DEL_LENGTH), scores_plot.T, cmap=cmap, norm=norm)
    ax.invert_yaxis()
    ax.set_xlabel("Start of CRISPR deletion")
    ax.set_ylabel("Length of CRISPR deletion", rotation=0, ha='right')
    # fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label="Num DddA edits")
    counts_plot = counts.copy().astype(float)
    counts_plot[:, 0] = np.nan
    ax_counts.pcolormesh(np.arange(seq_len), np.arange(MAX_DEL_LENGTH), counts_plot.T, cmap=plt.cm.viridis, norm=matplotlib.colors.LogNorm())
    ax_counts.invert_yaxis()
    ax_counts.set_ylabel("Num reads containing deletion", rotation=0, ha='right')
    fig.savefig(os.path.join(plot_dir, "accessibility_full.pdf"), bbox_inches="tight")
    plt.close(fig)

    for window in windows:
        inches_per_square = 0.2
        main_width = inches_per_square * (window.stop - window.start)
        cbar_width = 0.3
        cbar_wspace = 0.6
        fig, (ax, cax) = plt.subplots(
            figsize=(main_width + cbar_wspace + cbar_width, inches_per_square * MAX_DEL_LENGTH),
            ncols=2,
            gridspec_kw=dict(width_ratios=[main_width, cbar_width], wspace=cbar_wspace / (main_width + cbar_wspace + cbar_width)),
            )
        ax.set_aspect('equal')
        ax.set_title("{}:{:,}-{:,}".format(*ddda_data.region_dict[locus]))
        x = np.arange(window.start, window.stop)
        scores_plot = scores.copy() + score_in_window
        scores_plot[:, 0] = np.nan
        scores_plot[counts < min_coverage] = np.nan
        vmin = np.nanpercentile(scores_plot[window], 1)
        vmax = np.nanpercentile(scores_plot[window], 99)
        if np.isnan(vmin):
            vmin = wt
        if np.isnan(vmax):
            vmax = wt
        ax.pcolormesh(x, np.arange(MAX_DEL_LENGTH), scores_plot[window].T, cmap=plt.cm.coolwarm, norm=matplotlib.colors.TwoSlopeNorm(vmin=min(vmin, wt - 1), vcenter=wt, vmax=max(vmax, wt + 1)))
        ax.invert_yaxis()
        ax.set_xlabel("Start of CRISPR deletion")
        ax.set_ylabel("Length of CRISPR deletion")
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label="Num DddA edits", cax=cax)
        cbar.set_ticks([wt], labels=["WT"], minor=True)
        # set minor tick length to same as major
        cbar.ax.tick_params(which='minor', length=plt.rcParams['ytick.major.size'])
        ax.set_xticks(x, labels=[ref_seq[i] for i in range(window.start, window.stop)])
        ax.set_xticks(target_sites - 0.5, minor=True, labels=target_sites.index.to_series().apply("sg{}".format), rotation=45, ha='right')
        ax.tick_params(axis='x', which='minor', direction='out', top=False, bottom=True, labelbottom=True, labeltop=False, length=20)
        ax.set_xlim(window.start - 0.5, window.stop - 0.5)
        if minimal_motif is not None:
            for tk in ax.get_xticklabels():
                if tk.get_position()[0] >= minimal_motif.start and tk.get_position()[0] < minimal_motif.stop:
                    tk.set_color('red')
        fig.savefig(os.path.join(plot_dir, f"accessibility_{window.start}-{window.stop}.pdf"), bbox_inches="tight")
        plt.close(fig)

    logging.info("Done")

if __name__ == '__main__':
    assert len(sys.argv) == 2
    dataset_id = int(sys.argv[1])
    main(dataset_id)
