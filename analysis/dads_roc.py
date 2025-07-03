import pandas as pd
from tdac_seq.ddda_dataset import ddda_dataset
import itertools
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tdac_seq.analysis.utils import prepare_matplotlib
prepare_matplotlib()
from tqdm import tqdm
from tdac_seq.utils import get_atac_peaks
from dads import convolve_func, hmm_big_func
from tdac_seq.dad import _call_dads_hmm_bias, call_dads_hmm_bias
import plotly.io
plotly.io.templates.default = "simple_white"
import sys
import scipy.sparse
import matplotlib.ticker
import seaborn as sns

if __name__ == "__main__":
    if int(sys.argv[1]) == 0:
        func = _call_dads_hmm_bias
    elif int(sys.argv[1]) == 1:
        func = hmm_big_func(70, call_dads_hmm_bias)
    elif int(sys.argv[1]) == 2:
        func = convolve_func(window=70, thresh=None)
    else:
        raise ValueError("Invalid function index")

    parent_working_dir = "output/dads_roc"
    os.makedirs(parent_working_dir, exist_ok=True)
    atac_fname = "output/dads/atac.pkl"
    read_counts_fname = "output/dads/read_counts.tsv"
    parent_plots_dir = "plots/dads_roc"

    logging.info(f"Processing {func.__name__}")

    reads_idx = slice(10000)

    working_dir = os.path.join(parent_working_dir, func.__name__)
    os.makedirs(working_dir, exist_ok=True)
    plots_dir = os.path.join(parent_plots_dir, func.__name__)
    os.makedirs(plots_dir, exist_ok=True)

    dad_fname = os.path.join(working_dir, 'dad.pkl')

    regions = pd.read_csv('data/regions.tsv', sep='\t', index_col="name")
    datasets = pd.read_csv('data/datasets.tsv', sep='\t', index_col="ID")
    datasets = datasets.loc[datasets.index[datasets.index.str.startswith('HJR250_')]]
    datasets['regions'] = datasets['regions'].str.split(',')
    assert set(itertools.chain.from_iterable(datasets['regions'])).issubset(regions.index), "Some regions are not in the regions.tsv file"

    atac = pickle.load(open(atac_fname, 'rb'))
    read_counts = pd.read_csv(read_counts_fname, sep='\t', index_col=0, header=None).squeeze()

    if os.path.exists(dad_fname):
        dad = pickle.load(open(dad_fname, 'rb'))
    else:
        dad = []
        for id, data_info in tqdm(datasets.iterrows(), total=len(datasets), desc="Datasets"):
            logging.info(f"Processing dataset {id}")

            object_fname = os.path.join('output', id, "ddda_data.pkl")
            if not os.path.exists(object_fname):
                logging.error(f"Pre-aligned data not found: {object_fname}. Skipping dataset {id}")
                continue

            logging.info(f"Loading pre-aligned data from {object_fname}")
            with open(object_fname, "rb") as f:
                ddda_data: ddda_dataset = pickle.load(f)

            assert len(ddda_data.ref_seq_dict) == 1

            # Make genome tracks
            locus, ref_seq = next(iter(ddda_data.ref_seq_dict.items()))

            # inds = ddda_data.deduped_inds[locus]
            inds = slice(None)
            edits = ddda_data.edit_dict[locus][inds]
            editable = np.array([i for i, b in enumerate(ref_seq) if b in "CG" and i > 500 and i < len(ref_seq) - 500])
            strands = ddda_data.read_strands[locus][inds]
            del_matrix = ddda_data.del_dict[locus][inds]
            genomic_region = ddda_data.region_dict[locus]

            # raw DddA edits track
            edits_plot = edits[reads_idx].toarray()
            strands_plot = strands[reads_idx]
            # runs of DddA edits
            dad.append(func(edits_plot, ref_seq, strands_plot))

            logging.info(f"Done with dataset {id}")

        pickle.dump(dad, open(dad_fname, 'wb'))

    dad = [_dad if not scipy.sparse.issparse(_dad) else _dad.toarray() for _dad in dad]

    df = dict()
    THRESHS = np.linspace(min(_dad.min() for _dad in dad), max(_dad.max() for _dad in dad), 100)
    for _atac, _dad, id in zip(atac, dad, read_counts.index):
        logging.info(f"Processing dataset {id}")

        logging.info("Retrieving ATAC peaks")
        genomic_region = regions.loc[datasets.loc[id, 'regions'][0], ['chr', 'start', 'end']].values
        peaks = get_atac_peaks(*genomic_region)
        if len(peaks) == 0:
            logging.error(f"No peaks found for {genomic_region}")
            continue
        peak_start, peak_end = map(np.array, zip(*peaks))
        peak_start = np.maximum(peak_start, 0)
        peak_end = np.minimum(peak_end, _dad.shape[1])

        logging.info("Plotting atac track")
        fig, (ax, ax2) = plt.subplots(figsize=(6, 4), nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax.fill_between(np.arange(len(_atac)), _atac, linewidth=0, color=plt.cm.tab10(0))
        ax.twinx().plot(_dad.mean(axis=0), color=plt.cm.tab10(1))
        for start, end, color in zip(peak_start, peak_end, plt.cm.tab10.colors):
            ax2.barh(y=1, left=start, width=end - start, color=color)
        ax2.set_yticks([])
        ax2.set_ylim(0, 2)
        fig.savefig(os.path.join(plots_dir, f"atac_peaks_{id}.pdf"), bbox_inches="tight")
        plt.close(fig)

    peak_intensity = []
    true_positive_rate = []
    peak_starts = []
    peak_ends = []
    peak_dataset = []
    dad_atac_overlap = []
    loci = []
    for _atac, _dad, id in zip(atac, dad, read_counts.index):
        logging.info(f"Processing dataset {id}")

        logging.info("Retrieving ATAC peaks")
        genomic_region = regions.loc[datasets.loc[id, 'regions'][0], ['chr', 'start', 'end']].values
        peaks_all = get_atac_peaks(*genomic_region)
        # exclude peaks that are too close to the edges
        peaks = [(start, end) for start, end in peaks_all if start > 500 and end < genomic_region[2] - genomic_region[1] - 500]
        if len(peaks) == 0:
            logging.error(f"No peaks found for {genomic_region}")
            continue
        # trim peaks that are too close to the edges
        peak_start, peak_end = map(np.array, zip(*peaks))
        peak_start = np.maximum(peak_start, 0)
        peak_end = np.minimum(peak_end, _dad.shape[1])
        peak_start_all, peak_end_all = map(np.array, zip(*peaks_all))
        peak_start_all = np.maximum(peak_start_all, 0)
        peak_end_all = np.minimum(peak_end_all, _dad.shape[1])
        # add to list
        peak_starts.extend(peak_start)
        peak_ends.extend(peak_end)
        peak_dataset.extend([id] * len(peak_start))
        peak_mask_all = np.zeros(_dad.shape[1], dtype=bool)
        for start, end in zip(peak_start_all, peak_end_all):
            peak_mask_all[start:end] = True
        dad_atac_overlap.append((_dad[:, peak_mask_all].mean(), _dad[:, ~peak_mask_all].mean()))
        loci.append(id)

        for peak_i, (start, end) in enumerate(zip(peak_start, peak_end)):
            dad_vals = _dad[:, start:end].max(axis=1)
            dad_binary = dad_vals[:, np.newaxis] >= THRESHS[np.newaxis, :]
            peak_intensity.append(np.percentile(_atac[start:end], 99)) # to avoid outliers
            true_positive_rate.append(dad_binary.mean(axis=0))
    peak_intensity = np.array(peak_intensity)
    true_positive_rate = np.array(true_positive_rate)
    peak_starts = np.array(peak_starts)
    peak_ends = np.array(peak_ends)
    peak_dataset = np.array(peak_dataset)
    dad_atac_overlap = np.array(dad_atac_overlap).T
    loci = np.array(loci)

    bad_loci = ['HJR250_1', 'HJR250_2', 'HJR250_4', 'HJR250_5', 'HJR250_7', 'HJR250_13', 'HJR250_16'] # these datasets have no ATAC peaks and/or insufficient reads
    mask_clean = np.array([id not in bad_loci for id in peak_dataset])
    mask_loci_clean = np.array([id not in bad_loci for id in loci])

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(*dad_atac_overlap[:, mask_loci_clean], alpha=0.5, s=20)
    lims = (0, dad_atac_overlap[:, mask_loci_clean].max() * 1.05)
    ax.set_xlabel("% of bases covered by\nDAD $\\bf{and}$ ATAC peak")
    ax.set_ylabel("% of bases covered by\nDAD $\\bf{but\ not}$ ATAC peak")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    for x, y, locus in zip(*dad_atac_overlap[:, mask_loci_clean], loci[mask_loci_clean]):
        if y / x > 0.5:
            ax.annotate(locus, xy=(x, y), xytext=(4, 0), textcoords='offset points', fontsize=8, ha="left", va="center")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, 'k--')
    fig.savefig(os.path.join(plots_dir, f"dad_vs_atac_perbase_clean.pdf"), bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2, 4))
    bins = [200, 1800]
    plot_params = dict(x=np.digitize(peak_intensity[mask_clean], bins=bins), y=true_positive_rate[mask_clean, 1], ax=ax)
    sns.boxplot(**plot_params, showfliers=False, color='lightgray')
    sns.swarmplot(**plot_params, color='black')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f"<{bins[0]}", f"{bins[0]}-{bins[1]}", f"≥{bins[1]}"])
    ax.set_xlabel("ATAC peak intensity")
    ax.set_ylabel("True positive rate\n(% reads with DAD)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    fig.savefig(os.path.join(plots_dir, f"roc2.pdf"), bbox_inches='tight')
    plt.close(fig)

    logging.info(f"Done with {func.__name__}")
