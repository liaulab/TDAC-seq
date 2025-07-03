import pandas as pd
from tdac_seq.ddda_dataset import ddda_dataset
import itertools
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import pickle
import numpy as np
from tdac_seq.dad import call_dads_hmm_bias
from tdac_seq.analysis.utils import parse_guides_and_props, prepare_matplotlib
prepare_matplotlib()
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from tdac_seq.analysis.screen import calculate_guide_effects
from tdac_seq.crispr import find_editable_pos
import matplotlib.patches
import matplotlib.lines
from tdac_seq.constants import peaks_for_locus
import scipy.stats
import seaborn as sns
from typing import Literal
from tdac_seq.analysis.deletion import plot_accessibility_alleletable_by_motif
from seaborn.categorical import Beeswarm
import matplotlib.colors
import matplotlib.ticker
import re
import plotly.io
plotly.io.templates.default = "plotly_white"
import plotly.express as px
from tdac_seq.plots import plot_schematic
from tdac_seq.utils import masked_smooth, smooth
from tdac_seq.analysis.minimal_motif import minimal_motif_search, plot_motif_search
from tdac_seq.analysis.utils import set_xtick_genomic_coords
from tdac_seq.constants import get_motifs
import pyBigWig

def _draw_rectangle(ax, x, w, y, h=1):
    ax.add_patch(matplotlib.patches.Rectangle((x, y - 0.5), w, h, edgecolor='white', facecolor='black', lw=0.2))

def _plot_coverage(read_count: np.ndarray, min_coverage: int, fname: str, deduplicated: bool) -> None:
    reads_str = "deduplicated reads" if deduplicated else "reads"
    fig, ax = plt.subplots()
    ax.plot(np.sort(read_count)[::-1])
    ax.axhline(min_coverage, color='red')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Genotype rank')
    ax.set_ylabel(f'Number of {reads_str}')
    ax.set_title(f"Number of genotypes with at least {min_coverage} {reads_str}: {np.sum(read_count >= min_coverage)}")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)

def _plot_heatmap(fname: str, method: Literal['dad', 'raw'], comparison: Literal['diff', 'agg'], is_abe: bool, data_plot: np.ndarray, genotypes: list[tuple], guides: pd.DataFrame = None, genomic_region: tuple[str, int, int] = None, agg_mtx_wt: np.ndarray = None, indicate_num_abe_edits: Literal['markercolor', 'line'] = None, xlim: tuple[int] = None) -> None:
    """
    if is_abe, then genotypes is a list of tuples of positions. Otherwise, it's a list of tuples of (del_start, del_len).
    if not is_abe and guides is not None, then the guides are plotted on the heatmap.
    """
    if xlim is None:
        xlim = (500, data_plot.shape[1] - 500)
    if method == 'dad':
        norm_max = 0.3
        label = "DAD score"
    elif method == 'raw':
        if is_abe:
            norm_max = 0.09
        else:
            norm_max = 0.1
        label = "DddA rate"
    if comparison == 'diff':
        cmap = plt.cm.RdBu_r
        label += " difference"
        # norm = plt.Normalize(vmin=-norm_max, vmax=norm_max)
        norm = matplotlib.colors.CenteredNorm(vcenter=0)
        fig_width = 6 * data_plot.shape[1] / 4000
        fig_height = min(0.15 * data_plot.shape[0], 10)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    elif comparison == 'agg':
        cmap = plt.cm.Purples
        norm = plt.Normalize(vmin=0)
        fig_width = 6 * data_plot.shape[1] / 4000
        fig_height = min(0.15 * (data_plot.shape[0] + 1), 10)
        wt_height = 0.4
        fig, (ax_wt, ax) = plt.subplots(figsize=(fig_width, fig_height), dpi=150, nrows=2, sharex=True, gridspec_kw=dict(height_ratios=[wt_height / fig_height, 1 - wt_height / fig_height], hspace=0))
    ax.pcolormesh(*np.meshgrid(np.arange(data_plot.shape[1]), np.arange(data_plot.shape[0])), data_plot, cmap=cmap, norm=norm, rasterized=True)
    if is_abe:
        for i, abe_edits in enumerate(genotypes):
            for _abe_edit in abe_edits:
                _draw_rectangle(ax, x=_abe_edit - 10, w=20, y=i)
        if indicate_num_abe_edits is not None:
            multiedits = np.array([len(abe_edits) for abe_edits in genotypes])
            if indicate_num_abe_edits == 'markercolor':
                for i, c in zip(range(1, multiedits.max() + 1), plt.cm.Set1.colors):
                    mask = multiedits == i
                    ax.scatter(np.full(mask.sum(), 300), np.arange(len(genotypes))[mask], color=c, s=5, clip_on=False, marker="s", label=i)
                ax.legend(title="Num ABE edits", loc='upper left', bbox_to_anchor=(1.05, 1))
            elif indicate_num_abe_edits == 'line':
                assert sorted(multiedits) == list(multiedits)
                for num_abe_edits in range(1, multiedits.max() + 1):
                    row_i_start = np.searchsorted(multiedits, num_abe_edits, side='left') # inclusive
                    row_i_end = np.searchsorted(multiedits, num_abe_edits, side='right') - 1 # inclusive
                    x = xlim[1] + (xlim[1] - xlim[0]) * 0.12
                    ax.annotate(f"{num_abe_edits} ABE edit{'s' if num_abe_edits != 1 else ''}", xy=(x, (row_i_start + row_i_end) / 2), xytext=(2, 0), textcoords="offset points", annotation_clip=False, ha='left', va='center', rotation=90 if (row_i_end - row_i_start) > 7 else 0, fontsize=6)
                    ax.add_artist(matplotlib.lines.Line2D((x, x), (row_i_start - 0.3, row_i_end + 0.3), color='black', clip_on=False, lw=0.5))
    else:
        for i, (del_start, del_len) in enumerate(genotypes):
            _draw_rectangle(ax, x=del_start, w=del_len, y=i)
    ax.set_xlim(*xlim)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=label, location="bottom")
    if genomic_region is not None:
        ax.set_xlabel(f"Position in the locus\namplicon = {genomic_region[0]}:{genomic_region[1]:,}-{genomic_region[2]:,}\nvisible = {genomic_region[0]}:{genomic_region[1] + xlim[0]:,}-{genomic_region[1] + xlim[1]:,}")
    else:
        ax.set_xlabel(f"Position in the locus")
    ax2 = ax.twinx()
    ax.set_ylim(-0.5, len(genotypes) - 0.5)
    ax2.set_ylim(-0.5, len(genotypes) - 0.5)
    ax.invert_yaxis()
    ax2.invert_yaxis()
    ax2.set_yticks(np.arange(len(genotypes)))
    ax2.tick_params(axis='y', length=1, width=0.1, pad=1)
    if is_abe:
        ax2.set_yticklabels(['+'.join(map(str, genotype)) for genotype in genotypes], fontsize=2)
    else:
        ax2.set_yticklabels([f"Δ{del_start}-{del_start + del_len}" for del_start, del_len in genotypes], fontsize=1)
    # assign each cutsite to a guide that has the closest cutsite
    if not is_abe:
        dels = np.array(genotypes)
        del_start = dels[:, 0]
        del_end = dels[:, 0] + dels[:, 1]
        if guides is not None: # draw bars for guide groups
            guides_plot = guides.copy()
            guides_assignments = []
            for _del_start, _del_end in zip(del_start, del_end):
                guides_assignments.append(
                    tuple(guides.index[(_del_start <= guides['cutsite'] + 2) & (guides['cutsite'] - 2 <= _del_end)])
                    )
            guides_plot['row_i_start'] = pd.NA
            guides_plot['row_i_end'] = pd.NA
            for guide in guides_plot.itertuples():
                guides_plot.loc[guide.Index, 'row_i_start'] = min([i for i, x in enumerate(guides_assignments) if guide.Index in x], default=pd.NA)
                guides_plot.loc[guide.Index, 'row_i_end'] = max([i for i, x in enumerate(guides_assignments) if guide.Index in x], default=pd.NA)
            guides_plot['row_i_start'] = np.searchsorted(del_start, guides_plot['cutsite'] - 20, side='left')
            guides_plot['row_i_end'] = np.searchsorted(del_end, guides_plot['cutsite'] + 20, side='right')
            guides_plot.sort_values(['row_i_start', 'row_i_end'], inplace=True)
            num_cols = 6
            buffer = 10
            prev_end = np.full(num_cols, dtype=int, fill_value=-buffer)
            for guide in guides_plot.itertuples():
                x = np.argmax(guide.row_i_start >= prev_end + buffer)
                prev_end[x] = guide.row_i_end
                x = 240 - (120 * x)
                ax.annotate(f"sg{guide.Index}", xy=(x, (guide.row_i_start + guide.row_i_end) / 2), annotation_clip=False, ha='right', va='center', rotation=90, fontsize=6)
                ax.add_artist(matplotlib.lines.Line2D((x, x), (guide.row_i_start, guide.row_i_end), color='black', clip_on=False, lw=0.5))
    if comparison == 'agg':
        assert agg_mtx_wt is not None
        ax_wt.pcolormesh(agg_mtx_wt.mean(axis=0)[np.newaxis], cmap=cmap, norm=norm, rasterized=True)
        ax_wt.invert_yaxis()
        ax_wt.set_yticks([0.5])
        ax_wt.set_yticklabels(["WT"])
    fig.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close(fig)

regions = pd.read_csv('data/regions.tsv', sep='\t', index_col="name")
datasets = pd.read_csv('data/datasets.tsv', sep='\t', index_col="ID")
datasets['regions'] = datasets['regions'].str.split(',')
assert set(itertools.chain.from_iterable(datasets['regions'])).issubset(regions.index), "Some regions are not in the regions.tsv file"
datasets['regions_of_interest'] = datasets['regions_of_interest'].str.split(',').fillna('')

def main(dataset_id: int, subset_id: int = None, num_subsets: int = None):
    data_info = datasets.iloc[dataset_id]
    id = data_info.name
    is_abe = data_info['ABE']

    logging.info(f"Processing dataset {id}")

    working_dir = os.path.join("output", id)
    os.makedirs(working_dir, exist_ok=True)
    plots_dir = os.path.join("plots", id, "hs2")
    os.makedirs(plots_dir, exist_ok=True)

    object_fname = os.path.join(working_dir, "ddda_data.pkl")
    if not os.path.exists(object_fname):
        logging.error(f"Pre-aligned data not found: {object_fname}. Skipping dataset {id}")
        return

    logging.info(f"Loading pre-aligned data from {object_fname}")
    with open(object_fname, "rb") as f:
        ddda_data: ddda_dataset = pickle.load(f)

    assert len(ddda_data.ref_seq_dict) == 1

    locus, ref_seq = next(iter(ddda_data.ref_seq_dict.items()))
    edits = ddda_data.edit_dict[locus]
    # Compute a read-by-position matrix of deletion labels
    if is_abe:
        crispr_matrix = ddda_data.ABE_edit_dict[locus]
    else:
        crispr_matrix = ddda_data.del_dict[locus]
    # Get strandedness of each read
    strands = ddda_data.read_strands[locus]
    strand_read_inds = {
        "C_to_T" : np.where(strands == 0)[0],
        "G_to_A" : np.where(strands == 1)[0]
    }
    genomic_region: tuple[str, int, int] = ddda_data.region_dict[locus]
    peaks = peaks_for_locus(locus, genomic_region)

    # Load sgRNA sequences
    guides = parse_guides_and_props(data_info['sgRNA'], ref_seq)

    if is_abe:
        # Get deletion genotypes by start of deletion and length of deletion
        editable_pos = find_editable_pos(ref_seq, guides, editing_window=(2, 8))
        editable_pos_list = sorted(editable_pos)
        genotype_fname = os.path.join(working_dir, 'genotypes.tsv')
        if os.path.exists(genotype_fname):
            logging.info(f"Loading genotypes from {genotype_fname}")
            genotypes = pd.read_csv(genotype_fname, sep='\t', index_col='read')
            genotypes['abe_edits'] = genotypes['abe_edits'].apply(eval)
        else:
            logging.info('Calling ABE genotypes')
            genotypes = pd.DataFrame({'ag_edits': map(tuple, crispr_matrix.tolil().rows)}).rename_axis(index='read')
            genotypes['abe_edits'] = genotypes['ag_edits'].apply(lambda x: tuple(pos for pos in x if pos in editable_pos))
            genotypes['abe_only'] = genotypes['ag_edits'] == genotypes['abe_edits']
            genotypes.to_csv(genotype_fname, sep='\t')
        genotypes.drop(index=genotypes[~genotypes['abe_only']].index, inplace=True)
        genotypes_agg = genotypes.groupby('abe_edits').size().rename('count')
        genotypes_agg.drop(index=tuple(), inplace=True)
        genotypes_agg.drop(index=genotypes_agg[genotypes_agg < 100].index, inplace=True)
    else:
        # Get deletion genotypes by start of deletion and length of deletion
        genotype_fname = os.path.join(working_dir, 'genotypes.tsv')
        if os.path.exists(genotype_fname):
            logging.info(f"Loading genotypes from {genotype_fname}")
            genotypes = pd.read_csv(genotype_fname, sep='\t', index_col='read')
        else:
            logging.info('Calling deletion genotypes')
            del_start = crispr_matrix.argmax(axis=1)
            del_len = crispr_matrix.sum(axis=1)
            del_only = np.zeros(crispr_matrix.shape[0], dtype=bool)
            for i in trange(len(del_only), desc="Checking deletion-only reads"):
                del_only[i] = crispr_matrix[[i], del_start[i]:del_start[i] + del_len[i]].sum() == del_len[i]
            genotypes = pd.DataFrame({'del_start': del_start, 'del_len': del_len, 'del_only': del_only}).rename_axis(index='read')
            genotypes.to_csv(genotype_fname, sep='\t')
        genotypes.drop(index=genotypes[~genotypes['del_only']].index, inplace=True)
        genotypes_agg = genotypes.groupby(['del_start', 'del_len']).size().rename('count')
        genotypes_agg.drop(index=genotypes_agg[genotypes_agg < 100].index, inplace=True)
        genotypes_agg.drop(index=(0, 0), inplace=True)
    def get_subset_slice(subset_id, num_subsets):
        return slice(len(genotypes_agg) * subset_id // num_subsets, len(genotypes_agg) * (subset_id + 1) // num_subsets)
    if subset_id is not None:
        subset_slice = get_subset_slice(subset_id, num_subsets)
        logging.info(f"Getting subset {subset_id} out of {num_subsets}, which is genotypes {subset_slice.start} to {subset_slice.stop} out of {len(genotypes_agg)}")
        genotypes_agg = genotypes_agg.iloc[subset_slice]
        sgrna_results_fname = os.path.join(working_dir, f"sgRNA_results_bias_byGenotype_{subset_id}_outof{num_subsets}.pkl")
    else:
        sgrna_results_fname = os.path.join(working_dir, "sgRNA_results_bias_byGenotype.pkl")

    # Load or generate sgRNA results
    sgRNA_results = {
        "raw": {sgRNA_pos: {} for sgRNA_pos in genotypes_agg.index},
        "dad": {sgRNA_pos: {} for sgRNA_pos in genotypes_agg.index},
        }
    if os.path.exists(sgrna_results_fname):
        logging.info(f"Loading sgRNA results from {sgrna_results_fname}")
        with open(sgrna_results_fname, "rb") as f:
            sgRNA_results = pickle.load(f)
    elif subset_id is None and next((m for fname in os.listdir(working_dir) if (m := re.match(r'^sgRNA_results_bias_byGenotype_\d+_outof(\d+)\.pkl$', fname)) is not None), None) is not None:
        # load and combine sgRNA results from multiple subsets
        num_subsets = int(m.group(1))
        for _subset_id in range(num_subsets):
            _sgrna_results_fname = os.path.join(working_dir, f"sgRNA_results_bias_byGenotype_{_subset_id}_outof{num_subsets}.pkl")
            logging.info(f"Loading sgRNA results from {_sgrna_results_fname} and combining")
            with open(_sgrna_results_fname, "rb") as f:
                _sgRNA_results = pickle.load(f)
            for method, sgRNA_results_sub in _sgRNA_results.items():
                assert all(len(sgRNA_results[method][genotype]) == 0 for genotype in sgRNA_results_sub.keys()), "Redundant sgRNA results between subsets"
                sgRNA_results[method].update(sgRNA_results_sub)
            logging.info('Done combining sgRNA results')
            for method in sgRNA_results:
                for genotype in genotypes_agg[get_subset_slice(_subset_id, num_subsets)].index:
                    if len(sgRNA_results[method][genotype]) == 0:
                        logging.warning(f"Missing sgRNA results for {genotype} in {method}")
            logging.info('Done checking sgRNA results')
    else:
        logging.info(f"Calculating sgRNA results")

        dad_fname = os.path.join(working_dir, 'dad_hmm_bias.pkl')
        if os.path.exists(dad_fname):
            logging.info(f"Loading DADs from {dad_fname}")
            dad = pickle.load(open(dad_fname, 'rb'))
        else:
            dad = []
            logging.info(f"Calling DADs")
            # runs of DddA edits
            dad.append(call_dads_hmm_bias(edits.toarray(), ref_seq, strands))
            pickle.dump(dad, open(dad_fname, 'wb'))
        dad = dad[0]

        for genotype, read_count in tqdm(genotypes_agg.items(), desc="sgRNA", total=len(genotypes_agg)):
            if is_abe:
                abe_edits = genotype
            else:
                del_start, del_len = genotype

            ###############################################################
            # Select reads with and without deletion by the current sgRNA #
            ###############################################################
        
            # Find reads where the sgRNA target site is covered by a deletion
            if not is_abe:
                del_end = del_start + del_len
                target_site_del = np.squeeze(np.array(np.max(crispr_matrix[:, del_start:del_end], axis=1).todense()))
                del_read_inds = np.where(target_site_del == 1)[0]

                # Only keep reads where positions outside of the vicinity of the sgRNA target site are not deleted
                upstream_filter = np.squeeze(np.array(np.max(crispr_matrix[:, :del_start], axis=1).todense())) == 0
                downstream_filter = np.squeeze(np.array(np.max(crispr_matrix[:,del_end:], axis=1).todense())) == 0
                filter = np.where(upstream_filter & downstream_filter)[0]
                del_read_inds = np.intersect1d(del_read_inds, filter)                
            else:
                # don't go by genotypes series here because i've dropped some entries from there

                target_site_del = np.sum(crispr_matrix[:, abe_edits], axis=1)
                del_read_inds = np.where(target_site_del == len(abe_edits))[0]

                # Only keep reads where positions outside of the vicinity of the sgRNA target site are not deleted
                filter = np.where(np.sum(crispr_matrix, axis=1) == len(abe_edits))[0]
                del_read_inds = np.intersect1d(del_read_inds, filter)

            # Separately, also find reads without any deletion as a control
            read_with_del = np.sum(crispr_matrix, axis=1)
            undel_read_inds = np.where(read_with_del == 0)[0]

            ###################################
            # Down-sampling and deduplication #
            ###################################
            
            # Down-sample the number of reads
            min_num = min(len(del_read_inds), len(undel_read_inds), 5000)
            del_read_inds = np.random.choice(del_read_inds, min_num, replace=False)
            undel_read_inds = np.random.choice(undel_read_inds, min_num, replace=False)

            try:
                # De-duplicate reads
                del_read_ids = ddda_data.dedup_reads(
                    locus = locus, 
                    read_ids = np.array(ddda_data.read_ids[locus])[del_read_inds]
                )
                undel_read_ids = ddda_data.dedup_reads(
                    locus = locus, 
                    read_ids = np.array(ddda_data.read_ids[locus])[undel_read_inds]
                )
            except:
                logging.error(f"Error in deduplication for {genotype}")
                continue

            # For each read ID, get its index in the full read ID list
            locus_ids = ddda_data.read_ids[locus]
            locus_id_dict = dict(zip(locus_ids, np.arange(len(locus_ids))))
            del_read_inds = np.array([*map(locus_id_dict.get, del_read_ids)])
            undel_read_inds = np.array([*map(locus_id_dict.get, undel_read_ids)])

            ###############################################################
            # Calculate DddA editing rate for deleted and undeleted reads #
            ###############################################################

            for strand in  ["C_to_T", "G_to_A"]:

                # Calculate average editing rate on deleted reads for both C-to-T and G-to-A strands
                del_read_inds_stranded = np.intersect1d(del_read_inds, strand_read_inds[strand])
                if len(del_read_inds_stranded) > 0:
                    del_edits = np.squeeze(np.array(np.mean(edits[del_read_inds_stranded, :], axis=0)))
                    del_dads = np.squeeze(np.array(np.mean(dad[del_read_inds_stranded, :], axis=0)))
                else:
                    del_edits = np.full(len(ref_seq), np.nan)
                    del_dads = np.full(len(ref_seq), np.nan)
        
                # Calculate average editing rate on undeleted reads for both C-to-T and G-to-A strands
                undel_read_inds_stranded = np.intersect1d(undel_read_inds, strand_read_inds[strand])
                if len(undel_read_inds_stranded) > 0:
                    undel_edits = np.squeeze(np.array(np.mean(edits[undel_read_inds_stranded, :], axis=0)))
                    undel_dads = np.squeeze(np.array(np.mean(dad[undel_read_inds_stranded, :], axis=0)))
                else:
                    undel_edits = np.full(len(ref_seq), np.nan)
                    undel_dads = np.full(len(ref_seq), np.nan)

                # Mask edits in the deleted region
                if not is_abe:
                    del_edits[del_start:del_end] = 0
                    undel_edits[del_start:del_end] = 0
                    del_dads[del_start:del_end] = 0
                    undel_dads[del_start:del_end] = 0
        
                sgRNA_results['raw'][genotype][strand] = {
                    "total_reads": read_count,
                    "n_reads" : min_num,
                    "n_deduped_del_reads": len(del_read_inds_stranded),
                    "n_deduped_undel_reads": len(undel_read_inds_stranded),
                    "del_edits" : del_edits,
                    "undel_edits" : undel_edits,
                    "n_del_edits_in_peak": np.array([edits[del_read_inds_stranded, peak[0]:peak[1]].sum(axis=1) for peak in peaks]),
                    "n_undel_edits_in_peak": np.array([edits[undel_read_inds_stranded, peak[0]:peak[1]].sum(axis=1) for peak in peaks]),
                    'del_read_inds_stranded': del_read_inds_stranded,
                    'undel_read_inds_stranded': undel_read_inds_stranded,
                }
                sgRNA_results['dad'][genotype][strand] = {
                    "total_reads": read_count,
                    "n_reads" : min_num,
                    "n_deduped_del_reads": len(del_read_inds_stranded),
                    "n_deduped_undel_reads": len(undel_read_inds_stranded),
                    "del_edits" : del_dads,
                    "undel_edits" : undel_dads,
                    "n_del_edits_in_peak": np.array([dad[del_read_inds_stranded, peak[0]:peak[1]].sum(axis=1) for peak in peaks]),
                    "n_undel_edits_in_peak": np.array([dad[undel_read_inds_stranded, peak[0]:peak[1]].sum(axis=1) for peak in peaks]),
                    'del_read_inds_stranded': del_read_inds_stranded,
                    'undel_read_inds_stranded': undel_read_inds_stranded,
                }

        with open(sgrna_results_fname, "wb") as f:
            pickle.dump(sgRNA_results, f)

    # re-run without subset_id to make plots
    if subset_id is not None:
        return

    for method, sgRNA_results_sub in sgRNA_results.items():
        effect_mtx, agg_mtx, agg_mtx_wt, read_count, cut_sites_ = calculate_guide_effects(ref_seq, sgRNA_results_sub, smooth=method == 'raw', is_abe=is_abe)
        read_count_dedup = np.array([
            sum(sgRNA_results_sub[genotype][strand]['n_deduped_del_reads'] for strand in ['C_to_T', 'G_to_A']) 
            for genotype in cut_sites_
        ])
        if list(map(tuple, cut_sites_)) == genotypes_agg.index.tolist():
            logging.info("Genotypes match")
        else:
            assert all(genotype in cut_sites_ for genotype in genotypes_agg.index)
            mask = [cut_sites_.index(genotype) for genotype in genotypes_agg.index]
            effect_mtx = effect_mtx[mask]
            agg_mtx = agg_mtx[mask]
            agg_mtx_wt = agg_mtx_wt[mask]
            read_count = read_count[mask]
            cut_sites_ = [cut_sites_[i] for i in mask]
            read_count_dedup = read_count_dedup[mask]
        MAX_DEL_LENGTH = 30
        if not is_abe:
            # get rid of huge deletions
            mask = genotypes_agg.index.get_level_values('del_len') <= MAX_DEL_LENGTH
            # get rid of deletions not near a guide
            mask &= genotypes_agg.index.to_frame().apply(lambda x: any(min(abs(x['del_start'] - cutsite), abs(x['del_start'] + x['del_len'] - cutsite)) < 20 for cutsite in guides['cutsite']), axis=1)
            logging.info(f"Keeping {mask.sum()} out of {len(mask)} genotypes. {', '.join(guides['cutsite'].astype(str))}")
            # set min coverage
            if locus == 'HOXA':
                min_coverage = {
                    'HJR309_rep1_1': 180,
                    'HJR309_rep1_2': 120,
                    'HJR309_rep1_3': 10,
                    'HJR309_rep1_pool': 180,
                    'HJR309_rep2_1': 280,
                    'HJR309_rep2_2': 280,
                    'HJR309_rep2_3': 10,
                    'HJR309_rep2_pool': 220,
                }.get(id, 180)
            elif id.startswith('HJR312'):
                min_coverage = 3000
            else:
                min_coverage = {
                    'HJR308_Ddda11_1': 200,
                    'HJR308_Ddda11_2': 600,
                    'HJR308_Ddda11_3': 800,
                    'HJR308_Ddda11_5': 300,
                    'HJR308_Ddda11_6': 400,
                    'HJR308_MGY_1': 600,
                    'HJR308_MGY_2': 600,
                    'HJR308_MGY_6': 3000,
                }.get(id, 400)
            _plot_coverage(read_count_dedup[mask], min_coverage, os.path.join(plots_dir, f"read_count_dedup({method}).pdf"), deduplicated=True)
        else:
            mask = np.ones(len(read_count_dedup), dtype=bool)
            min_coverage = 100
            _plot_coverage(read_count_dedup, min_coverage, os.path.join(plots_dir, f"read_count_dedup({method}).pdf"), deduplicated=True)

        mask &= read_count_dedup >= min_coverage
        effect_mtx = effect_mtx[mask]
        agg_mtx = agg_mtx[mask]
        agg_mtx_wt = agg_mtx_wt[mask]
        read_count = read_count[mask]
        read_count_dedup = read_count_dedup[mask]
        cut_sites_ = [x for m, x in zip(mask, cut_sites_) if m]

        wt_mean = 0
        wt_count = 0
        pvals = np.zeros(len(cut_sites_))
        means = np.zeros(len(cut_sites_))
        counts = np.zeros(len(cut_sites_), dtype=int)
        means_diffs = np.zeros(len(cut_sites_))
        if locus == 'GFI1B':
            peak_i = 2
        elif locus == 'LCR_new':
            peak_i = 1
        else:
            raise ValueError(f"Need to indicate peak of interest for {locus}")
        for i, genotype in enumerate(cut_sites_):
            x_del = np.concatenate([sgRNA_results_sub[genotype][strand]['n_del_edits_in_peak'][peak_i] for strand in ['C_to_T', 'G_to_A']])
            x_undel = np.concatenate([sgRNA_results_sub[genotype][strand]['n_undel_edits_in_peak'][peak_i] for strand in ['C_to_T', 'G_to_A']])
            counts[i] = np.sum([sgRNA_results_sub[genotype][strand]['n_deduped_del_reads'] for strand in ['C_to_T', 'G_to_A']])
            res = scipy.stats.ttest_ind(x_del, x_undel, equal_var=False)
            pvals[i] = res.pvalue
            means[i] = np.mean(x_del)
            means_diffs[i] = effect_mtx[i][slice(*peaks[peak_i])].sum()
            wt_mean += np.sum(x_undel)
            wt_count += len(x_undel)
        wt_mean /= wt_count

        # Minimal motif clustering analysis
        motif_search = get_motifs(locus)
        motifs = list(motif_search.get('motif', {}).values())
        motifs_highlight = list(motif_search['motif'][motif_name] for motif_name in motif_search.get('motif_highlight', []))
        if (motif_search_params := motif_search.get(locus)) is not None:
            motif_search_fname = os.path.join(working_dir, "hs2", "motif_search.npz")
            os.makedirs(os.path.dirname(motif_search_fname), exist_ok=True)
            scores = None
            if not os.path.exists(motif_search_fname):
                # only calculate if needed to save time
                scores = []
                for genotype in tqdm(cut_sites_, leave=False):
                    del_read_inds = np.concatenate([sgRNA_results_sub[genotype][strand]['del_read_inds_stranded'] for strand in ['C_to_T', 'G_to_A']])
                    undel_read_inds = np.concatenate([sgRNA_results_sub[genotype][strand]['undel_read_inds_stranded'] for strand in ['C_to_T', 'G_to_A']])
                    # count edits in atac peak of interest but not in deletion
                    peak_of_interest = np.zeros(len(ref_seq), dtype=bool)
                    peak_of_interest[slice(*peaks[peak_i])] = True
                    peak_of_interest[genotype[0]:genotype[0] + genotype[1]] = False
                    num_edits_del = edits[del_read_inds][:, peak_of_interest].sum(axis=1)
                    num_edits_wt = edits[undel_read_inds][:, peak_of_interest].sum(axis=1)
                    diff_edits = num_edits_del - num_edits_wt.mean()
                    scores.extend(map(tuple, itertools.product([genotype[0]], [genotype[1]], diff_edits)))
                scores = pd.DataFrame.from_records(scores, columns=['del_start', 'del_len', 'num_edits'])
            clustering_heatmap, clustering_heatmap_coverage = minimal_motif_search(
                scores=scores,
                motif_search_fname=motif_search_fname,
                searchrange_start=motif_search_params['searchrange_start'],
            )
            plot_motif_search(
                plot_dir=plots_dir,
                heatmap=clustering_heatmap,
                genomic_coords=genomic_region,
                searchrange_start=motif_search_params['searchrange_start'],
                heatmap_coverage=clustering_heatmap_coverage,
                motif_annotations=motif_search_params['motif'],
                pval_thresh=-400,
            )

        _tracks_folder = os.path.join(plots_dir, "tracks", method)
        os.makedirs(_tracks_folder, exist_ok=True)
        for roi in motifs:
            genotypes_to_plot = [x for x in cut_sites_ if (x[0] <= roi[1]) and (x[0] + x[1] >= roi[0])]
            fig, axes = plt.subplots(nrows=len(genotypes_to_plot), ncols=2, figsize=(6, 0.6 * len(genotypes_to_plot)), sharex=True, sharey=True)
            for i, (genotype, ax) in enumerate(zip(genotypes_to_plot, axes)):
                for j, (strand, _ax) in enumerate(zip(['C_to_T', 'G_to_A'], ax)):
                    track_wt = sgRNA_results_sub[genotype][strand]['undel_edits']
                    track_mut = sgRNA_results_sub[genotype][strand]['del_edits']
                    if method == 'raw':
                        mask = np.zeros(len(ref_seq), dtype=bool)
                        mask[genotype[0]:genotype[0] + genotype[1]] = True
                        track_wt = masked_smooth(track_wt, mask=mask, radius=50)
                        track_mut = masked_smooth(track_mut, mask=mask, radius=50)
                    _ax.plot(track_wt, label='WT', color='black', zorder=1)
                    _ax.plot(track_mut, label='Mut', color='red', zorder=0)
                    if i == 0:
                        _ax.set_title(strand)
                    if j == 0:
                        _ax.set_ylabel(f"Δ{genotype[0]}-{genotype[0] + genotype[1]}", rotation=0, ha='right', va='center')
                    if i == 0 and j == 1:
                        _ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
            sns.despine(fig)
            fig.savefig(os.path.join(_tracks_folder, f'tracks_{roi[0]}-{roi[1]}.pdf'), bbox_inches='tight')
            plt.close(fig)

        fig, ax = plt.subplots()
        if not is_abe:
            # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("gnuplot2_truncated", plt.cm.gnuplot2_r(np.linspace(0.1, 0.9, 100)))
            cmap = plt.cm.rainbow
            norm = plt.Normalize()
            colors = cmap(norm(np.array(cut_sites_)[:, 0] + 0.5 * np.array(cut_sites_)[:, 1]))
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Position of CRISPR deletion (center)')

            data = pd.DataFrame(cut_sites_, columns=['deletion start', 'deletion length'])
            data['num edits'] = means
            data['pval'] = pvals
            data['num deduped reads'] = counts
            data['diff edits'] = means_diffs
            px_fig = px.scatter(
                data,
                x='num edits',
                y='pval',
                log_y=True,
                color='deletion start',
                color_continuous_scale='rainbow',
                hover_data=['deletion start', 'deletion length', 'num deduped reads'],
                labels={'num edits': f'Num DddA edits in accessible region\n(chr{genomic_region[0]}:{genomic_region[1] + peaks[peak_i][0]:,}-{genomic_region[1] + peaks[peak_i][1]:,})', 'pval': 'p-value'}
                )
            px_fig.update_yaxes(autorange="reversed", exponentformat='power')
            px_fig.add_vline(x=wt_mean, line_color='red', line_width=2)
            px_fig.write_html(os.path.join(plots_dir, f"volcano({method}).html"))
            # add wildtype
            data = pd.concat([data, pd.DataFrame([{'deletion start': 0, 'deletion length': 0, 'num edits': wt_mean, 'pval': None, 'num deduped reads': wt_count, 'diff edits': 0}])], ignore_index=True)
            data.to_csv(os.path.join(working_dir, f'volcano({method}).tsv'), sep='\t', index=False)

            fig_interactive = px.scatter(
                data.drop(data[data['deletion length'] == 0].index),
                x='deletion start',
                y='deletion length',
                color='num edits',
                color_continuous_scale='rdbu_r',
                color_continuous_midpoint=data[data['deletion length'] == 0]['num edits'].mean(),
                hover_data=['deletion start', 'deletion length', 'num edits', 'pval', 'num deduped reads'],
            )
            fig_interactive.update_yaxes(autorange="reversed")
            fig_interactive.write_html(os.path.join(plots_dir, f'deletion_scatter_{method}.html'))

            edits_nodedup = edits[:, slice(*peaks[peak_i])].sum(axis=1)
            edits_dedup = np.concatenate([sgRNA_results_sub[genotype][strand]['n_del_edits_in_peak'][peak_i] for strand in ['C_to_T', 'G_to_A'] for genotype in cut_sites_])
            bins = np.histogram_bin_edges(np.concatenate([edits_nodedup, edits_dedup]), bins=50)
            fig_dedup, ax = plt.subplots()
            ax.hist(edits_nodedup, bins=bins, label='No dedup', density=True, alpha=0.5)
            ax.hist(edits_dedup, bins=bins, label='Dedup', density=True, alpha=0.5)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
            ax.set_xlabel(f'Num DddA edits in accessible region\n(chr{genomic_region[0]}:{genomic_region[1] + peaks[peak_i][0]:,}-{genomic_region[1] + peaks[peak_i][1]:,})')
            ax.set_ylabel('Fraction of reads')
            fig_dedup.savefig(os.path.join(plots_dir, f"dedup_effect_{method}.pdf"), bbox_inches="tight")
            plt.close(fig_dedup)
        else:
            num_abe_edits = np.array([len(genotype) for genotype in cut_sites_])
            cmap = matplotlib.colors.ListedColormap(plt.cm.Set1.colors[:num_abe_edits.max()])
            norm = matplotlib.colors.BoundaryNorm(np.arange(num_abe_edits.max() + 1) + 1 - 0.5, cmap.N)
            colors = cmap(norm((num_abe_edits)))
            for x, y, genotype in zip(means, pvals, cut_sites_):
                if x < 22 or x > wt_mean or y < 1e-30 or any(hit in genotype for hit in [1410, 1411, 1492, 1495, 1498]):
                    orientation = dict(xytext=(2, 2), ha='left', rotation=45) if x > wt_mean else dict(xytext=(-2, 2), ha='right', rotation=-45)
                    ax.annotate(f"{'+'.join(map(str, genotype))}", (x, y), textcoords="offset points", va='bottom', fontsize=4, **orientation, alpha=0.5)
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Num ABE edits', ticks=np.arange(num_abe_edits.max() + 1))

            data = pd.Series(cut_sites_, name='abe_edits').to_frame()
            data['num edits'] = means
            data['pval'] = pvals
            data['num deduped reads'] = counts
            data['diff edits'] = means_diffs
            # add wildtype
            data = pd.concat([data, pd.DataFrame([{'abe_edits': tuple(), 'num edits': wt_mean, 'pval': None, 'num deduped reads': wt_count, 'diff edits': 0}])], ignore_index=True)
            data.to_csv(os.path.join(working_dir, f'volcano({method}).tsv'), sep='\t', index=False)

        ax.scatter(means, pvals, c=colors, s=5)
        ax.set_xlabel(f'Num DddA edits in accessible region\n(chr{genomic_region[0]}:{genomic_region[1] + peaks[peak_i][0]:,}-{genomic_region[1] + peaks[peak_i][1]:,})')
        ax.set_ylabel('p-value')
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.axvline(wt_mean, color='red')
        sns.despine(fig)
        fig.savefig(os.path.join(plots_dir, f"volcano({method}).pdf"), bbox_inches="tight")
        plt.close(fig)

        if is_abe: # violin
            num_abe_edits = np.array([len(genotype) for genotype in cut_sites_])

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.set_ylim(4.5, 37.5)
            ax.set_xlim(0.5, 4.5)
            for i in range(1, num_abe_edits.max() + 1):
                mask = num_abe_edits == i
                orig_xy_data = np.stack([
                    num_abe_edits[mask],
                    means[mask],
                ]).T
                orig_xy = ax.transData.transform(orig_xy_data)
                radii = np.full(mask.sum(), 2.5)
                orig_xyr = np.c_[orig_xy, radii]
                # Sort along the value axis to facilitate the beeswarm
                sorter = np.argsort(orig_xy[:, 1])
                orig_xyr = orig_xyr[sorter]
                # Adjust points along the categorical axis to prevent overlaps
                new_xyr = np.empty_like(orig_xyr)
                beeswarm = Beeswarm()
                new_xyr[sorter] = beeswarm.beeswarm(orig_xyr)
                # Transform the point coordinates back to data coordinates
                new_xy = new_xyr[:, :2]
                new_x_data, new_y_data = ax.transData.inverted().transform(new_xy).T
                ax.scatter(new_x_data, new_y_data, s=5, color=plt.cm.Set1.colors[i - 1])
                for x, y, s in zip(new_x_data, new_y_data, [cut_sites_[_i] for _i, _mask in enumerate(mask) if _mask]):
                    if y <= 25:
                        ax.annotate(f"{'+'.join(map(str, s))}", (x, y), xytext=(3, 0), textcoords="offset points", va='center', ha='left', fontsize=4, alpha=0.5)
            ax.axhline(wt_mean, color='black')
            ax.set_xlabel('Num ABE edits')
            ax.set_ylabel(f'Mean DddA edits in peak {peak_i}\n({genomic_region[0]}:{genomic_region[1] + peaks[peak_i][0]:,}-{genomic_region[1] + peaks[peak_i][1]:,})')
            sns.despine(fig)
            fig.savefig(os.path.join(plots_dir, f"abe_violin({method}).pdf"), bbox_inches="tight")
            plt.close(fig)

        XLIMS = {'': None}
        if id.startswith('HJR309'):
            XLIMS['_zoom'] = (1500, 3500) # chr7:27160022-27162022
        for comparison in ('diff', 'agg'):
            data_plot = effect_mtx.copy() if comparison == 'diff' else agg_mtx.copy()

            if not is_abe:
                fig, (ax, ax_hist, ax_cbar) = plt.subplots(figsize=(10, 3), ncols=3, gridspec_kw=dict(width_ratios=[9, 0.7, 0.3], wspace=0))
                cmap = plt.cm.coolwarm
                if comparison == 'diff':
                    _wt_score = 0
                    _label = 'Diff DddA edits'
                else:
                    _wt_score = np.mean(np.concatenate([sgRNA_results_sub[genotype][strand]['n_undel_edits_in_peak'][peak_i] for strand in ['C_to_T', 'G_to_A'] for genotype in cut_sites_]))
                    _label = 'Num DddA edits'
                _label += f" in peak {peak_i}\n({genomic_region[0]}:{genomic_region[1] + peaks[peak_i][0]:,}-{genomic_region[1] + peaks[peak_i][1]:,})"
                _scores = data_plot[:, slice(*peaks[peak_i])].sum(axis=1)
                norm = matplotlib.colors.CenteredNorm(vcenter=_wt_score)
                min_size = 0.5
                size_slope = 8
                def size_norm(pval):
                    return np.maximum(min_size, -np.log10(pval) / size_slope)
                ax.scatter(*zip(*cut_sites_), c=_scores, cmap=cmap, norm=norm, s=size_norm(pvals))
                for pval in [10 ** (-min_size * size_slope)] + list(reversed(matplotlib.ticker.LogLocator(base=10, numticks=3).tick_values(max(pvals.min(), 1e-300), 10 ** (-min_size * size_slope))[1:-2])):
                    ax_cbar.scatter([], [], c='gray', s=size_norm(pval), label=f'{pval:.0e}')
                fig.legend(title='p-value', loc='upper left', bbox_to_anchor=(1, 1))
                fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cbar, label=_label)
                ax_cbar.set_yticks([_wt_score], minor=True)
                ax_cbar.set_yticklabels(['WT'], minor=True)
                hist_vals, _, _ = ax_hist.hist(_scores, bins=50, orientation='horizontal', color='gray')
                ax_hist.set_ylim(ax_cbar.get_ylim())
                ax_hist.set_xlim(left=hist_vals.max() * 2, right=0)
                ax_hist.xaxis.set_visible(False)
                ax_hist.yaxis.set_visible(False)
                sns.despine(ax=ax_hist, left=True, bottom=True, right=True, top=True)
                ax.set_ylabel('Length of deletion')
                xlim = (guides['cutsite'].min() - 20, guides['cutsite'].max() + 20)
                ax.set_xlim(xlim)
                ax.set_xlabel(f'Start position of deletion\nAmplicon = {genomic_region[0]}:{genomic_region[1]:,}-{genomic_region[2]:,}\nVisible = {genomic_region[0]}:{genomic_region[1] + xlim[0]:,}-{genomic_region[1] + xlim[1]:,}')
                set_xtick_genomic_coords(ax, xlim, genomic_region)
                fig.savefig(os.path.join(plots_dir, f"deletion_scatter_{method}_{comparison}.pdf"), bbox_inches="tight")
                if id.startswith('HJR312'):
                    ax2 = ax.twiny()
                    ax2.set_xlim(xlim)
                    for motif in motifs_highlight:
                        ax2.axvspan(*motif, color='gray', alpha=0.3, linewidth=0)
                    ax2.set_xticks([(motif[0] + motif[1]) / 2 for motif in motifs_highlight])
                    ax2.set_xticklabels([f"{motif[0]}-{motif[1]}" for motif in motifs_highlight], rotation=90, fontsize=8)
                    fig.savefig(os.path.join(plots_dir, f"deletion_scatter_{method}_{comparison}_annot.pdf"), bbox_inches="tight")
                plt.close(fig)

                window = 100
                tdac_wt = smooth(edits[genotypes[genotypes['del_len'] == 0].index].mean(axis=0), window=window, how='sum')
                fig, (ax_del, ax_atac, ax_wt) = plt.subplots(figsize=(9, 3), nrows=3, sharex=True)
                bw_kwargs = dict(x=np.arange(len(ref_seq)), y1=0)
                ax_del.fill_between(**bw_kwargs, y2=crispr_matrix.mean(axis=0), color='red')
                ax_del.set_ylim(bottom=0)
                ax_del.set_ylabel('Frac reads with deletion', rotation=0, ha='right', va='center')
                ax_del.axvspan(*xlim, color='gray', alpha=0.3, linewidth=0)
                if locus == 'GFI1B':
                    atac = pyBigWig.open("/n/holystore01/LABS/liau_lab/Users/heejinroh/HJR308_ATAC/HJR308_8/HJR308_8_rmdups_normalized.bw")
                else:
                    atac = pyBigWig.open("https://www.encodeproject.org/files/ENCFF357GNC/@@download/ENCFF357GNC.bigWig")
                atac_vals = np.array(atac.values(genomic_region[0], genomic_region[1], genomic_region[2] + 1))
                ax_atac.fill_between(**bw_kwargs, y2=atac_vals, color="gray")
                ax_atac.set_ylim(bottom=0)
                ax_atac.set_ylabel('ATAC', rotation=0, ha='right', va='center')
                ax_atac.axvspan(*xlim, color='gray', alpha=0.3, linewidth=0)
                ax_wt.fill_between(**bw_kwargs, y2=tdac_wt, color='purple')
                ax_wt.set_ylim(bottom=0)
                ax_wt.set_ylabel(f'Num DddA edits\nin wildtype reads\nin rolling {window}-bp window', rotation=0, ha='right', va='center')
                ax_wt.axvspan(*xlim, color='gray', alpha=0.3, linewidth=0)
                genomic_coords_ticks = matplotlib.ticker.AutoLocator().tick_values(genomic_region[1], genomic_region[2])[1:-1]
                ax_wt.set_xticks(genomic_coords_ticks - genomic_region[1], minor=True)
                ax_wt.set_xticklabels([f"{int(x):,}" for x in genomic_coords_ticks], minor=True, rotation=90, fontsize=8)
                ax_wt.set_xlim(500, len(ref_seq) - 500)
                ax_wt.set_xlabel(f'Position in amplicon\nAmplicon={genomic_region[0]}:{genomic_region[1]:,}-{genomic_region[2]:,}\nVisible={genomic_region[0]}:{genomic_region[1] + 500:,}-{genomic_region[2] - 500:,}')
                sns.despine(fig=fig, right=True, top=True)
                fig.savefig(os.path.join(plots_dir, f"deletion_scatter_locus_schematic.pdf"), bbox_inches="tight")
                plt.close(fig)

                data = pd.Series(
                    data_plot[:, slice(*peaks[peak_i])].sum(axis=1),
                    index=pd.MultiIndex.from_tuples(cut_sites_, names=['del_start', 'del_len']),
                    name='num edits',
                ).reset_index()
                data['del_end'] = data['del_start'] + data['del_len']
                ylim = (data['num edits'].min(), data['num edits'].max())
                yrange = ylim[1] - ylim[0]
                ylim = (ylim[0] - 0.05 * yrange, ylim[1] + 0.05 * yrange)
                for motif in motifs:
                    _individual_plots_dir = os.path.join(plots_dir, 'individual_cre', f'{method}_{comparison}')
                    os.makedirs(_individual_plots_dir, exist_ok=True)
                    fig = plot_schematic(
                        motifs={'': motif},
                        ref_seq=ref_seq,
                        genomic_coords=genomic_region,
                        window_in_amplicon=(motif[0] - 40, motif[1] + 40),
                    )
                    fig.savefig(os.path.join(_individual_plots_dir, f'schematic_{motif[0]}-{motif[1]}.pdf'), bbox_inches='tight')
                    plt.close(fig)
                    data['full_del'] = (motif[0] >= data['del_start']) & (motif[1] <= data['del_end'])
                    data['part_del'] = (motif[0] <= data['del_end']) & (motif[1] >= data['del_start'])
                    data['label'] = 'other'
                    data.loc[data['part_del'], 'label'] = 'part'
                    data.loc[data['full_del'], 'label'] = 'full'
                    data['label'] = pd.Categorical(data['label'], categories=['full', 'part', 'other'])
                    for exclude_others in [False, True]:
                        if exclude_others:
                            order = ['full', 'part']
                            xlabels = ['Fully\ndeleted', 'Partially\ndeleted']
                            plot_fname_suffix = 'boxplotclean'
                        else:
                            order = ['full', 'part', 'other']
                            xlabels = ['Fully\ndeleted', 'Partially\ndeleted', 'Other']
                            plot_fname_suffix = 'boxplot'
                        fig, ax = plt.subplots(figsize=(len(order) * 2 / 3, 3))
                        if not exclude_others:
                            sns.boxplot(data[data['label'] == 'other'], x='label', y='num edits', ax=ax, color='lightgray', order=order, flierprops=dict(markersize=3))
                        sns.boxplot(data[data['label'] != 'other'], x='label', y='num edits', ax=ax, color='lightgray', order=order, showfliers=False)
                        sns.swarmplot(data[data['label'] != 'other'], x='label', y='num edits', ax=ax, color='black', order=order, size=3)
                        ax.axhline(_wt_score, color='red')
                        ax.set_ylabel(f'{_label} in peak {peak_i}\n({genomic_region[0]}:{genomic_region[1] + peaks[peak_i][0]:,}-{genomic_region[1] + peaks[peak_i][1]:,})')
                        ax.set_xlabel(f'Mutation status of\nCRE at {motif[0]}-{motif[1]}')
                        ax.set_xticks(range(len(order)))
                        ax.set_xticklabels(xlabels, rotation=45, ha='right')
                        ax.set_ylim(ylim)
                        fig.savefig(os.path.join(_individual_plots_dir, f'{plot_fname_suffix}_{motif[0]}-{motif[1]}.pdf'), bbox_inches='tight')
                        plt.close(fig)

            for xlim_suffix, xlim in XLIMS.items():
                _plot_heatmap(
                    fname=os.path.join(plots_dir, f"sgRNA_{comparison}_edit_heatmap({method})_byGenotype{xlim_suffix}.pdf"),
                    method=method,
                    comparison=comparison,
                    is_abe=is_abe,
                    data_plot=data_plot,
                    genotypes=cut_sites_,
                    guides=guides,
                    genomic_region=genomic_region,
                    agg_mtx_wt=agg_mtx_wt.copy() if comparison == 'agg' else None,
                    xlim=xlim,
                )

                if is_abe:
                    # indicate number of abe_edits by mcolor
                    _plot_heatmap(
                        fname=os.path.join(plots_dir, f"sgRNA_{comparison}_edit_heatmap({method})_byGenotype_multiabe{xlim_suffix}.pdf"),
                        method=method,
                        comparison=comparison,
                        is_abe=is_abe,
                        data_plot=effect_mtx.copy() if comparison == 'diff' else agg_mtx.copy(),
                        genotypes=cut_sites_,
                        guides=guides,
                        genomic_region=genomic_region,
                        agg_mtx_wt=agg_mtx_wt.copy() if comparison == 'agg' else None,
                        indicate_num_abe_edits='markercolor',
                        xlim=xlim,
                    )
                    # sort by number of abe_edits and then position of abe_edits
                    indices = sorted(range(len(cut_sites_)), key=lambda i: (len(cut_sites_[i]), cut_sites_[i]))
                    _plot_heatmap(
                        fname=os.path.join(plots_dir, f"sgRNA_{comparison}_edit_heatmap({method})_byGenotype_grouped{xlim_suffix}.pdf"),
                        method=method,
                        comparison=comparison,
                        is_abe=is_abe,
                        data_plot=effect_mtx[indices].copy() if comparison == 'diff' else agg_mtx[indices].copy(),
                        genotypes=[cut_sites_[i] for i in indices],
                        guides=guides,
                        genomic_region=genomic_region,
                        agg_mtx_wt=agg_mtx_wt[indices].copy() if comparison == 'agg' else None,
                        indicate_num_abe_edits='line',
                        xlim=xlim,
                    )

        allele_table_rois = []
        if locus == "LCR_new":
            if is_abe:
                allele_table_rois = [
                    ((1410, 1411), slice(1390, 1430)),
                    ((1492, 1493), slice(1470, 1510)),
                ]
            else:
                allele_table_rois = [
                    ((1488, 1498), slice(1470, 1510)),
                ]
        scores_df_values = []
        if is_abe:
            undel_genotype = tuple()
        else:
            undel_genotype = (0, 0)
        genotype_col_names = ['del_start', 'del_len'] if not is_abe else ['abe_edits']
        for i, genotype in enumerate(cut_sites_):
            x_undel = np.concatenate([sgRNA_results_sub[genotype][strand]['n_undel_edits_in_peak'][peak_i] for strand in ['C_to_T', 'G_to_A']])
            x_del = np.concatenate([sgRNA_results_sub[genotype][strand]['n_del_edits_in_peak'][peak_i] for strand in ['C_to_T', 'G_to_A']])
            if is_abe:
                scores_df_values.extend(itertools.product([genotype], x_del))
                scores_df_values.extend(itertools.product([undel_genotype], x_undel))
            else:
                scores_df_values.extend(itertools.product([genotype[0]], [genotype[1]], x_del))
                scores_df_values.extend(itertools.product([undel_genotype[0]], [undel_genotype[1]], x_undel))
        scores_df = pd.DataFrame(scores_df_values, columns=genotype_col_names + [peak_i])
        scores_df.set_index(genotype_col_names, inplace=True, append=True)
        scores_df = scores_df[peak_i]
        scores_agg = scores_df.groupby(genotype_col_names).size().rename('count').to_frame()
        scores_agg['freq'] = scores_agg['count'] / scores_agg['count'].sum()
        scores_agg.reset_index(inplace=True)
        for motif, window in allele_table_rois:
            if is_abe:
                editable_pos_sub = set(pos for pos in editable_pos if window.start <= pos <= window.stop)
                mask = scores_agg['abe_edits'].apply(lambda x: all(pos in editable_pos_sub for pos in x))
            else:
                mask = (scores_agg['del_start'] >= window.start) & (scores_agg['del_start'] + scores_agg['del_len'] <= window.stop)
            plot_accessibility_alleletable_by_motif(
                plot_fname=os.path.join(plots_dir, f"alleletable_motif{motif[0]}-{motif[1]}_({method}).pdf"),
                window=window,
                genomic_coords=genomic_region,
                ref_seq=ref_seq,
                target_sites=guides['cutsite'],
                scores_agg=scores_agg[mask],
                min_coverage=min_coverage,
                scores_df=scores_df,
                motif=motif,
                is_abe=is_abe,
                plot_kind='box',
                )

    logging.info(f"Done with {id}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        main(int(sys.argv[1]))
    elif len(sys.argv) == 4:
        main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
