import pandas as pd
from tdac_seq.ddda_dataset import ddda_dataset
import itertools
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
from tdac_seq.refseq import extract_ref_seq, regions_df_to_dict
import pickle
import numpy as np
import pyBigWig
from tdac_seq.dad import call_dads_hmm, call_dads_merge, call_dads_hmm_bias
import matplotlib.pyplot as plt
from tdac_seq.analysis.utils import prepare_matplotlib
prepare_matplotlib()
import scipy.stats
import sys
from tqdm import tqdm
from tdac_seq.plots import plot_hic_map
import scipy.signal
import matplotlib.ticker
from tdac_seq.analysis.coactuation import calc_coactuation
import seaborn as sns
from tdac_seq.plots import plot_atac_peaks
from tdac_seq.utils import get_atac_peaks
import multiprocessing as mp

def raw(edits, ref_seq, strands):
    return edits

def hmm(edits, ref_seq, strands):
    return call_dads_hmm(edits, ref_seq, strands)

def merge(edits, ref_seq, strands, thresh: int):
    return call_dads_merge(edits, distance_thresh=thresh)

def merge_func(thresh):
    def func(edits, ref_seq, strands):
        return merge(edits, ref_seq, strands, thresh)
    func.__name__ = f"merge_{thresh}"
    return func

def _convolve(edits, window: int):
    return scipy.signal.convolve(edits, np.full((1, window), 1. / window), mode='same')

def convolve_func(window, thresh: float | None = 0.25 * 0.5):
    def func(edits, ref_seq, strands):
        scores = _convolve(edits, window)
        if thresh is None:
            return scores
        return scores > thresh
    if thresh is None:
        func.__name__ = f"convolve_{window}"
    else:
        func.__name__ = f"convolve_{window}_{thresh}"
    return func

def hmm_big(edits, ref_seq, strands, thresh: int, hmm_func):
    dad = hmm_func(edits, ref_seq, strands)
    dad_cleaned = np.zeros_like(dad, dtype=bool)
    for i, _dad in enumerate(dad):
        dad_start = None
        for position, in_dad in enumerate(_dad):
            if in_dad:
                if dad_start is None:
                    dad_start = position
            else:
                if dad_start is not None:
                    if position - dad_start >= thresh:
                        dad_cleaned[i, dad_start:position] = True
                    dad_start = None
    return dad_cleaned

def hmm_big_func(thresh, hmm_func):
    def func(edits, ref_seq, strands):
        return hmm_big(edits, ref_seq, strands, thresh, hmm_func)
    func.__name__ = f"{hmm_func.__name__.lstrip('call_dads_')}_big_{thresh}"
    return func

def hmm_bias(edits, ref_seq, strands):
    return call_dads_hmm_bias(edits, ref_seq, strands)

DAD_FUNCS = [
    raw,
    hmm,
    merge_func(50),
    merge_func(65),
    merge_func(70),
    merge_func(80),
    convolve_func(5),
    convolve_func(10),
    convolve_func(15),
    convolve_func(20),
    convolve_func(50),
    convolve_func(70),
    convolve_func(100),
    convolve_func(100, thresh=0.02),
    convolve_func(100, thresh=0.03),
    convolve_func(100, thresh=0.04),
    convolve_func(100, thresh=0.05),
    convolve_func(100, thresh=0.1),
    convolve_func(100, thresh=0.25),
    hmm_big_func(10, call_dads_hmm),
    hmm_big_func(50, call_dads_hmm),
    hmm_big_func(70, call_dads_hmm),
    hmm_bias,
    hmm_big_func(10, call_dads_hmm_bias),
    hmm_big_func(50, call_dads_hmm_bias),
    hmm_big_func(70, call_dads_hmm_bias),
]

if __name__ == "__main__":
    parent_working_dir = "output/dads"
    os.makedirs(parent_working_dir, exist_ok=True)
    atac_fname = os.path.join(parent_working_dir, 'atac.pkl')
    read_counts_fname = os.path.join(parent_working_dir, "read_counts.tsv")
    parent_plots_dir = "plots/dads"
    for i, func in enumerate(DAD_FUNCS):
        if len(sys.argv) > 1 and sys.argv[1] != str(i):
            continue
        logging.info(f"Processing {func.__name__}")

        reads_idx = slice(10000)

        working_dir = os.path.join(parent_working_dir, func.__name__)
        os.makedirs(working_dir, exist_ok=True)
        plots_dir = os.path.join(parent_plots_dir, func.__name__)
        os.makedirs(plots_dir, exist_ok=True)

        dad_fname = os.path.join(working_dir, 'dad.pkl')

        regions = pd.read_csv('data/regions.tsv', sep='\t', index_col="name")
        datasets = pd.read_csv('data/datasets.tsv', sep='\t', index_col="ID")
        datasets = datasets.loc[datasets.index[datasets.index.str.startswith('HJR250_') | datasets.index.str.startswith('HJR306_17_')]]
        datasets['regions'] = datasets['regions'].str.split(',')
        assert set(itertools.chain.from_iterable(datasets['regions'])).issubset(regions.index), "Some regions are not in the regions.tsv file"
        if not os.path.exists('data/ref.fa'):
            extract_ref_seq(genome_file='/n/holystore01/LABS/liau_lab/Users/sps/genomes/hg38_ucsc/hg38.fa', region_dict=regions_df_to_dict(regions), out_fasta_path='data/ref.fa')

        atac = []
        dad = []
        read_counts = pd.Series(dtype=int)
        if os.path.exists(atac_fname) and os.path.exists(dad_fname) and os.path.exists(read_counts_fname):
            atac = pickle.load(open(atac_fname, 'rb'))
            dad = pickle.load(open(dad_fname, 'rb'))
            read_counts = pd.read_csv(read_counts_fname, sep='\t', index_col=0, header=None).squeeze()
        if (clamp := min(len(atac), len(dad), len(read_counts))) < len(datasets):
            atac = atac[:clamp]
            dad = dad[:clamp]
            read_counts = read_counts[:clamp]
            for id, data_info in tqdm(datasets.iterrows(), total=len(datasets), desc="Datasets"):
                if id in read_counts.index:
                    logging.info(f"Dataset {id} already processed. Skipping.")
                    continue

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

                read_counts[id] = ddda_data.edit_dict[locus].shape[0]
                # inds = ddda_data.deduped_inds[locus]
                inds = slice(None)
                edits = ddda_data.edit_dict[locus][inds]
                editable = np.array([i for i, b in enumerate(ref_seq) if b in "CG" and i > 500 and i < len(ref_seq) - 500])
                strands = ddda_data.read_strands[locus][inds]
                del_matrix = ddda_data.del_dict[locus][inds]
                genomic_region = ddda_data.region_dict[locus]

                atac.append(pyBigWig.open('https://www.encodeproject.org/files/ENCFF357GNC/@@download/ENCFF357GNC.bigWig').values(*genomic_region))
                # raw DddA edits track
                edits_plot = edits[reads_idx].toarray()
                strands_plot = strands[reads_idx]
                # runs of DddA edits
                dad.append(func(edits_plot, ref_seq, strands_plot))

                logging.info(f"Done with dataset {id}")

            pickle.dump(atac, open(atac_fname, 'wb'))
            pickle.dump(dad, open(dad_fname, 'wb'))
            read_counts.to_csv(read_counts_fname, sep='\t', header=False)

        for _atac, _dad, id in zip(atac, dad, read_counts.index):
            logging.info(f"Processing dataset {id}")

            dad_atac_overlap_fname = os.path.join(working_dir, f"dad_peak_overlap_{id}.npz")
            if not os.path.exists(dad_atac_overlap_fname):
                logging.info("Retrieving ATAC peaks")
                genomic_region = regions.loc[datasets.loc[id, 'regions'][0], ['chr', 'start', 'end']].values
                peaks = get_atac_peaks(*genomic_region)
                fig = plot_atac_peaks(_atac, peaks)
                os.makedirs(os.path.join(plots_dir, "ref_atac_peaks"), exist_ok=True)
                fig.savefig(os.path.join(plots_dir, "ref_atac_peaks", f"atac_peaks_{id}.pdf"), bbox_inches="tight")
                plt.close(fig)
                if len(peaks) == 0:
                    logging.error(f"No peaks found for {genomic_region}")
                else:
                    logging.info("Calculating DAD lengths")
                    _dad[:, 0] = 0
                    _dad[:, -1] = 0
                    _dad_diff = _dad[:, 1:].astype(int) - _dad[:, :-1].astype(int)
                    dad_start = np.nonzero(_dad_diff == 1)[1]
                    dad_end = np.nonzero(_dad_diff == -1)[1]
                    dad_length = dad_end - dad_start
                    peak_start, peak_end = map(np.array, zip(*peaks))

                    atac_peak_overlap = np.zeros((len(peak_start), len(_dad)), dtype=bool)
                    for peak_i, (start, end) in enumerate(zip(peak_start, peak_end)):
                        atac_peak_overlap[peak_i] = np.any(_dad[:, start:end] > 0, axis=1)

                    logging.info("Counting ATAC peaks overlapping DADs")
                    dad_peak_overlap = np.zeros_like(dad_start, dtype=bool)
                    for i, (start, end) in enumerate(zip(dad_start, dad_end)):
                        dad_peak_overlap[i] = np.any((peak_start <= end) & (start <= peak_end))
                    np.savez_compressed(dad_atac_overlap_fname, dad_peak_overlap=dad_peak_overlap, dad_start=dad_start, dad_end=dad_end, dad_length=dad_length, atac_peak_overlap=atac_peak_overlap)

            if id not in ("HJR250_6", "HJR250_19", "HJR306_17_r1", "HJR306_17_r2"):
                continue

            locus = datasets.loc[id, 'regions'][0]
            
            logging.info(f"Calculating cooperativity for {id}")
            with np.errstate(divide='ignore', invalid='ignore'):
                corr = np.corrcoef(_dad, rowvar=False)
            logging.info(f"Calculating coactuation for {id}")
            coactuation = calc_coactuation(_dad)

            fisher_fname = os.path.join(working_dir, f"fisher_{id}.npz")
            if os.path.exists(fisher_fname):
                pvals = np.load(fisher_fname)['pvals']
            else:
                logging.info(f"Calculating Fisher's exact test for {id}")
                both = _dad.T.astype(int) @ _dad.astype(int)
                neither = (1 - _dad.T) @ (1 - _dad)
                one_not = _dad.T @ (1 - _dad)
                not_one = one_not.T
                assert np.all(both + neither + one_not + not_one == _dad.shape[0])
                def fisher_test(args):
                    i, j, table = args
                    res = scipy.stats.fisher_exact(table)
                    return i, j, res.pvalue
                pvals = np.full(both.shape, np.nan)
                with mp.Pool(mp.cpu_count()) as pool:
                    results = pool.imap(fisher_test, ((i, j, [[both[i, j], one_not[i, j]], [not_one[i, j], neither[i, j]]]) for i, j in itertools.combinations(range(both.shape[0]), 2)), chunksize=100)
                    for i, j, pvalue in tqdm(results, desc="Fisher's exact test", total=both.shape[0] * (both.shape[0] - 1) // 2):
                        pvals[i, j] = pvalue
                        pvals[j, i] = pvalue
                np.savez_compressed(fisher_fname, pvals=pvals)

            marginal = _dad.mean(axis=0)
            coactuation_expected = np.outer(marginal, marginal)

            for label, fname, data, cmap, norm in [
                ("DddA correlation", f"cooperativity_{id}.pdf", corr, plt.cm.bwr, plt.Normalize(-1, 1)),
                ("Co-accessibility", f"coactuation_{id}.pdf", coactuation, matplotlib.colors.LinearSegmentedColormap.from_list("WhiteOrangeRed", ['white', 'orange', 'red']), plt.Normalize(0, coactuation.max())),
                ("Fisher's exact test p-value", f"fisher_{id}.pdf", pvals, plt.cm.Greens_r, matplotlib.colors.LogNorm(vmin=np.nanpercentile(pvals[pvals < 1], 5), vmax=1)),
                ("Observed - expected co-accessibility", f"obsexpcoactuation_{id}.pdf", coactuation - coactuation_expected, plt.cm.bwr, plt.Normalize(-np.abs(coactuation - coactuation_expected).max(), np.abs(coactuation - coactuation_expected).max())),
                ]:
                logging.info(f"Plotting {label} for {id}")
                fig, (ax, ax_dads, ax_atac) = plt.subplots(nrows=3, figsize=(4, 7), gridspec_kw=dict(height_ratios=[5, 5, 1], hspace=0.2), dpi=300)
                plot_hic_map(data, fig, ax, cbar_kwargs=dict(label=label, format=matplotlib.ticker.PercentFormatter(1.) if label == "Co-accessibility" else None), cmap=cmap, norm=norm)
                # ATAC
                ax_atac.fill_between(np.arange(len(_atac)), _atac)
                ax_atac.set_yticks([])
                ax_atac.set_ylabel("ATAC")
                ax_atac.set_ylim(bottom=0)
                ax_atac.set_xlabel(f"{locus} ({regions.loc[locus, 'chr']}: {regions.loc[locus, 'start']:,}-{regions.loc[locus, 'end']:,})")
                ax_atac.spines['top'].set_visible(False)
                ax_atac.spines['right'].set_visible(False)
                ax_atac.set_xlim(0, len(_atac))
                # Dads per read
                _dad_plot = _dad[reads_idx]
                num_edits = np.sum(_dad_plot, axis=1)
                edits_argsort = np.argsort(num_edits)[::-1]
                h = ax_dads.pcolormesh(_dad_plot[edits_argsort], norm=plt.Normalize(0, 1), cmap=matplotlib.colors.ListedColormap([(1, 1, 1, 1), plt.cm.Purples(255)]), rasterized=True)
                ax_dads.xaxis.set_visible(False)
                sns.despine(ax=ax_dads, left=True, bottom=True, right=True, top=True)
                ax_dads.set_ylim(bottom=_dad_plot.shape[0], top=0)
                ax_dads.set_xlim(left=0, right=_dad_plot.shape[1])
                fig.savefig(os.path.join(plots_dir, fname), bbox_inches="tight")
                plt.close(fig)

            logging.info(f"Done with {id}")

        dad_agg = [_dad.mean(axis=0) for _dad in dad]

        for suffix, ids_to_exclude in {
            "": [],
            "clean": ['HJR250_1', 'HJR250_2', 'HJR250_4', 'HJR250_5', 'HJR250_7', 'HJR250_13', 'HJR250_16', 'HJR306_17_r1', 'HJR306_17_r2', 'HJR306_17_r3'], # these datasets have no ATAC peaks and/or insufficient reads
            }.items():

            dad_peak_overlap = []
            dad_length = []
            dad_peak_overlap_perlocus = {}
            atac_peak_overlap = []
            atac_peak_overlap_perlocus = {}
            for id in read_counts.index:
                if id in ids_to_exclude:
                    continue
                fname = os.path.join(working_dir, f"dad_peak_overlap_{id}.npz")
                if not os.path.exists(fname):
                    continue
                data = np.load(fname)
                dad_peak_overlap.extend(data['dad_peak_overlap'])
                dad_length.extend(data['dad_length'])
                dad_peak_overlap_perlocus[id] = np.mean(data['dad_peak_overlap'])
                atac_peak_overlap.extend(data['atac_peak_overlap'].flatten())
                atac_peak_overlap_perlocus[id] = np.mean(data['atac_peak_overlap'], axis=1)
            dad_peak_overlap = np.array(dad_peak_overlap)
            dad_length = np.array(dad_length)
            sig = scipy.stats.ttest_ind(dad_length[dad_peak_overlap], dad_length[~dad_peak_overlap], equal_var=False)
            fig, ax = plt.subplots(figsize=(2, 4))
            sns.boxplot(x=dad_peak_overlap, y=dad_length, showfliers=False, ax=ax)
            ax.set_title(f"p-value = {sig.pvalue:.2e}\nt-statistic = {sig.statistic:.2f}\ndf = {sig.df:.0f}\n'False' DADs = {np.sum(~dad_peak_overlap)}\n'True' DADs = {np.sum(dad_peak_overlap)}")
            ax.set_xlabel("DADs overlapping\nATAC peak")
            ax.set_ylabel("DAD length (bp)")
            fig.savefig(os.path.join(plots_dir, f"dad_length_HJR250_all{suffix}.pdf"), bbox_inches="tight")
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie([np.mean(dad_peak_overlap), 1 - np.mean(dad_peak_overlap)], labels=["Overlapping ATAC peak", "Not overlapping ATAC peak"], autopct='%1.1f%%')
            fig.savefig(os.path.join(plots_dir, f"dad_peak_overlap_HJR250_all{suffix}.pdf"), bbox_inches="tight")
            plt.close(fig)
            fig, ax = plt.subplots()
            ax.bar(range(len(dad_peak_overlap_perlocus)), dad_peak_overlap_perlocus.values())
            ax.set_ylim(0, 1)
            ax.set_xticks(range(len(dad_peak_overlap_perlocus)))
            ax.set_xticklabels(dad_peak_overlap_perlocus.keys(), rotation=45, ha='right')
            ax.set_ylabel("Fraction of DADs\noverlapping ATAC peak")
            fig.savefig(os.path.join(plots_dir, f"dad_peak_overlap_HJR250_perlocus{suffix}.pdf"), bbox_inches="tight")
            plt.close(fig)
            fig, ax = plt.subplots()
            df = pd.DataFrame.from_dict(atac_peak_overlap_perlocus, orient='index')
            df.columns = [f"Peak {i + 1}" for i in range(df.shape[1])]
            df.rename_axis(index="Locus", inplace=True)
            df.rename_axis(columns="ATAC peak", inplace=True)
            atac_peak_overlap_dad = df.stack().rename('Value').to_frame()
            sns.barplot(data=atac_peak_overlap_dad, ax=ax, x='Locus', y='Value', hue='ATAC peak')
            ax.set_xticks(range(len(atac_peak_overlap_perlocus)))
            ax.set_xticklabels(atac_peak_overlap_perlocus.keys(), rotation=45, ha='right')
            ax.set_ylabel("Fraction of ATAC peaks\noverlapping DAD")
            ax.legend(title="ATAC peak", loc='upper left', bbox_to_anchor=(1, 1))
            fig.savefig(os.path.join(plots_dir, f"atac_peak_overlap_HJR250_perlocus{suffix}.pdf"), bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(nrows=len(atac) - len(ids_to_exclude))
            atac_clean = [_atac for _atac, id in zip(atac, read_counts.index) if id not in ids_to_exclude]
            dad_clean = [_dad for _dad, id in zip(dad_agg, read_counts.index) if id not in ids_to_exclude]
            atac_max = max(np.max(_atac) for _atac in atac_clean)
            dad_max = max(np.max(_dad) for _dad in dad_clean)
            for _atac, _dad, _ax, id in zip(atac_clean, dad_clean, axes, [id for id in read_counts.index if id not in ids_to_exclude]):
                _ax.plot(np.array(_atac) / atac_max)
                _ax.plot(np.array(_dad) / dad_max)
                _ax.set_yticks([])
                _ax.set_xticks([])
                _ax.spines['top'].set_visible(False)
                _ax.spines['right'].set_visible(False)
                _ax.spines['left'].set_visible(False)
                _ax.set_ylabel(f"{datasets.loc[id, 'regions'][0]} ({id})", rotation=0, ha='right', va='center')
            plt.savefig(os.path.join(plots_dir, f"tracks{suffix}.pdf"), bbox_inches="tight")
            plt.close(fig)

            dad_peak_overlap_per_locus_clean = pd.Series(dad_peak_overlap_perlocus)
            dad_peak_overlap_per_locus_clean.drop(index=ids_to_exclude, inplace=True, errors='ignore')
            atac_peak_overlap_dad_clean = atac_peak_overlap_dad['Value']
            atac_peak_overlap_dad_clean.drop(index=ids_to_exclude, level=0, inplace=True, errors='ignore')
            fig, ax = plt.subplots(figsize=(4, 3))
            plot_params = dict(
                x=['Fraction of DADs\noverlapping ATAC peak'] * len(dad_peak_overlap_per_locus_clean) + ['Fraction of ATAC peaks\noverlapping DAD'] * len(atac_peak_overlap_dad_clean),
                y=np.concatenate([dad_peak_overlap_per_locus_clean, atac_peak_overlap_dad_clean]),
                ax=ax,
                )
            sns.swarmplot(**plot_params, color='black', clip_on=False, zorder=100)
            sns.boxplot(**plot_params, showfliers=False, color='lightgray', capprops=dict(clip_on=False))
            sns.despine(fig, top=True, right=True, bottom=True)
            ax.tick_params(axis='x', which='both', length=0)
            ax.set_ylim(0, 1)
            fig.savefig(os.path.join(plots_dir, f"together_overlap_HJR250{suffix}.pdf"), bbox_inches="tight")
            plt.close(fig)

        logging.info(f"Done with {func.__name__}")
