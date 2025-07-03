import pandas as pd
import itertools
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import matplotlib.pyplot as plt
plt.set_loglevel("WARNING")
from tqdm import tqdm
import itertools 
from tdac_seq.utils import get_atac_peaks
from tdac_seq.analysis.utils import parse_guides_and_props, prepare_matplotlib
prepare_matplotlib()
from tdac_seq.refseq import extract_ref_seq_from_fasta
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import re
from tdac_seq.plots import plot_schematic

parent_working_dir_prefix = "output/"

regions = pd.read_csv('data/regions.tsv', sep='\t', index_col="name")
datasets = pd.read_csv('data/datasets.tsv', sep='\t', index_col="ID")
datasets['regions'] = datasets['regions'].str.split(',')
datasets['working_dir'] = parent_working_dir_prefix + datasets.index
assert set(itertools.chain.from_iterable(datasets['regions'])).issubset(regions.index), "Some regions are not in the regions.tsv file"
plot_dir = f"plots/hoxa"
os.makedirs(plot_dir, exist_ok=True)
locus = 'HOXA'
ref_seq_dict = extract_ref_seq_from_fasta('data/ref.fa', [locus])
assert len(ref_seq_dict) == 1
genomic_coords = regions.loc[locus, ['chr', 'start', 'end']].values
# Get sequence of the target locus
ref_seq = ref_seq_dict[locus]
seq_len = len(ref_seq)
min_coverage = 100
peaks = [(0, len(ref_seq)), (2600, 2675), *get_atac_peaks(*genomic_coords)]
peak_i = 2
peak_start, peak_end = peaks[peak_i]

start_gap_threshold: int = 500
end_gap_threshold: int = 500

DATASET_IDS = ['HJR235_4', 'HJR309_rep1_1', 'HJR309_rep1_2', 'HJR309_rep1_3', 'HJR309_rep1_pool', 'HJR309_rep2_1', 'HJR309_rep2_2', 'HJR309_rep2_3', 'HJR309_rep2_pool', 'HJR309_235']
all_guides = []
all_scores_agg = []
all_scores_diff = []
all_scores_diff_const = []
all_scores_div = []
all_scores_div_const = []
for dataset_id in DATASET_IDS:
    data_info = datasets.loc[dataset_id].copy()
    id = data_info.name
    total_reads = int(data_info['num_reads'])
    working_dir = os.path.join(data_info['working_dir'], "del")
    os.makedirs(working_dir, exist_ok=True)

    logging.info(f"Processing dataset {id}")

    # Load sgRNA sequences
    guides = parse_guides_and_props(data_info['sgRNA'], ref_seq)
    if len(guides) == 1:
        if guides.iloc[0]['spacer'] == 'GCCATCTGCTGGCCGCCGTT':
            all_guides.append('sgRNA4')
        else:
            all_guides.append(guides.index[0])
    else:
        all_guides.append('pool')

    # Re-align to get full raw CRISPR info
    aln_fname = os.path.join(working_dir, "aln.tsv")
    assert os.path.exists(aln_fname)
    with open(aln_fname, "rt") as aln_file:
        first_line = aln_file.readline()
    assert first_line.startswith("read_id")

    MAX_DEL_LENGTH = 30
    logging.info("Calculating scores")

    with open(os.path.join(data_info['working_dir'], "sgRNA_results_bias_byGenotype.pkl"), "rb") as f:
        sgRNA_results = pickle.load(f)['raw']
    scores_diff = [] # start, length, diff_edits
    scores_diff_const = []
    scores_div = []
    scores_div_const = []
    const_mask = (2600, 2675)
    for genotype, _sgRNA_results in sgRNA_results.items():
        if genotype[1] > MAX_DEL_LENGTH:
            continue
        edits_del = sum(sgRNA_results[genotype][strand]['del_edits'][peak_start:peak_end] for strand in _sgRNA_results) / len(_sgRNA_results)
        edits_wt = sum(sgRNA_results[genotype][strand]['undel_edits'][peak_start:peak_end] for strand in _sgRNA_results) / len(_sgRNA_results)
        deduped_counts = sum(sgRNA_results[genotype][strand]['n_deduped_del_reads'] for strand in _sgRNA_results)
        # aggregate
        num_edits_del = np.sum(edits_del)
        num_edits_wt = np.sum(edits_wt)
        num_edits_del_mask = np.sum(edits_del[genotype[0]:genotype[0] + genotype[1]])
        num_edits_wt_mask = np.sum(edits_wt[genotype[0]:genotype[0] + genotype[1]])
        num_edits_del_const = np.sum(edits_del[const_mask[0]:const_mask[1]])
        num_edits_wt_const = np.sum(edits_wt[const_mask[0]:const_mask[1]])
        # mask edits in deleted region
        scores_diff.append((*genotype, (num_edits_del - num_edits_del_mask) - (num_edits_wt - num_edits_wt_mask), deduped_counts))
        scores_diff_const.append((*genotype, (num_edits_del - num_edits_del_const) - (num_edits_wt - num_edits_wt_const), deduped_counts))
        scores_div.append((*genotype, (num_edits_del - num_edits_del_mask) / (num_edits_wt - num_edits_wt_mask), deduped_counts))
        scores_div_const.append((*genotype, (num_edits_del - num_edits_del_const) / (num_edits_wt - num_edits_wt_const), deduped_counts))
    scores_diff = pd.DataFrame(scores_diff, columns=['del_start', 'del_len', 'mean', 'count']).set_index(['del_start', 'del_len'])
    scores_diff_const = pd.DataFrame(scores_diff_const, columns=['del_start', 'del_len', 'mean', 'count']).set_index(['del_start', 'del_len'])
    scores_div = pd.DataFrame(scores_div, columns=['del_start', 'del_len', 'mean', 'count']).set_index(['del_start', 'del_len'])
    scores_div_const = pd.DataFrame(scores_div_const, columns=['del_start', 'del_len', 'mean', 'count']).set_index(['del_start', 'del_len'])
    all_scores_diff.append(scores_diff)
    all_scores_diff_const.append(scores_diff_const)
    all_scores_div.append(scores_div)
    all_scores_div_const.append(scores_div_const)

    scores_df = [] # (start, length, *num edits for each peak)
    with open(aln_fname, "rt") as f:
        header = f.readline().rstrip('\n').split('\t')
        assert header == ['read_id', 'del_start', 'del_len'] + list(itertools.chain.from_iterable((f"num_ct_edits_{start}-{end}", f"num_ga_edits_{start}-{end}") for start, end in peaks))
        for line in tqdm(f):
            entries = line.rstrip('\n').split('\t')
            read_id, start, length, *edits = entries
            num_ct_edits = edits[0::2]
            num_ga_edits = edits[1::2]
            start, length = map(int, (start, length))
            num_ct_edits = list(map(int, num_ct_edits))
            num_ga_edits = list(map(int, num_ga_edits))
            if length >= MAX_DEL_LENGTH:
                continue
            if num_ct_edits[0] > num_ga_edits[0]:
                scores_df.append((read_id, start, length, 'ct', *num_ct_edits))
            else:
                scores_df.append((read_id, start, length, 'ga', *num_ga_edits))
    scores_df = pd.DataFrame(scores_df, columns=['read', 'del_start', 'del_len', 'strand'] + list(range(len(peaks)))).set_index(['read', 'del_start', 'del_len', 'strand'])
    assert all((scores_df.index.get_level_values('del_len') == 0) == (scores_df.index.get_level_values('del_start') == 0))
    scores_agg = scores_df[peak_i].groupby(['del_start', 'del_len']).describe()
    wt, wt_count = scores_agg.loc[(0, 0), ['mean', 'count']]
    logging.info(f"WT score: {wt} ({wt_count} reads)")
    all_scores_agg.append(scores_agg)

    logging.info("Done")

def annotate(scores, motifs):
    scores['del_end'] = scores['del_start'] + scores['del_len']
    for motif_name, motif in motifs.items():
        scores[f'motif_{motif_name}_full'] = (scores['del_start'] <= motif[0]) & (scores['del_end'] >= motif[1])
        scores[f'motif_{motif_name}_part'] = (scores['del_start'] <= motif[1]) & (scores['del_end'] >= motif[0])

def label_partial_full(scores):
    scores['annotation'] = 'Other'
    scores.loc[scores['motif_CTCF_part'] & ~scores['motif_YY1_part'], 'annotation'] = 'CTCF only partially deleted'
    scores.loc[scores['motif_CTCF_full'] & ~scores['motif_YY1_part'], 'annotation'] = 'CTCF only fully deleted'
    scores.loc[~scores['motif_CTCF_part'] & scores['motif_YY1_part'], 'annotation'] = 'YY1 only partially deleted'
    scores.loc[~scores['motif_CTCF_part'] & scores['motif_YY1_full'], 'annotation'] = 'YY1 only fully deleted'
    scores.loc[scores['motif_CTCF_part'] & scores['motif_YY1_part'], 'annotation'] = 'both CTCF and YY1 partially deleted'
    scores.loc[scores['motif_CTCF_full'] & scores['motif_YY1_full'], 'annotation'] = 'both CTCF and YY1 partially deleted'
    scores.loc[~scores['motif_CTCF_part'] & ~scores['motif_YY1_part'], 'annotation'] = 'neither CTCF nor YY1 perturbed'
    scores.loc[scores['del_len'] == 0, 'annotation'] = 'WT'

def label_partial_only(scores):
    scores['annotation'] = 'Other'
    scores.loc[scores['motif_CTCF_part'] & ~scores['motif_YY1_part'], 'annotation'] = 'CTCF only deleted'
    scores.loc[~scores['motif_CTCF_part'] & scores['motif_YY1_part'], 'annotation'] = 'YY1 only deleted'
    scores.loc[scores['motif_CTCF_part'] & scores['motif_YY1_part'], 'annotation'] = 'both CTCF and YY1 deleted'
    scores.loc[~scores['motif_CTCF_part'] & ~scores['motif_YY1_part'], 'annotation'] = 'neither CTCF nor YY1 deleted'
    scores.loc[scores['del_len'] == 0, 'annotation'] = 'WT'

def _get_scores(all_scores, min_coverage):
    all_scores_agg = pd.concat(all_scores, keys=DATASET_IDS, names=['dataset_name'])
    scores = all_scores_agg.drop(index=all_scores_agg.index[all_scores_agg['count'] < min_coverage])
    scores.reset_index(inplace=True)
    scores['sgRNA'] = scores['dataset_name'].map({dataset_name: guides for dataset_name, guides in zip(DATASET_IDS, all_guides)})
    scores['rep'] = scores['dataset_name'].apply(lambda x: m.group(1) if (m := re.match(r'HJR309_rep(\d+)_[\d+|pool]', x)) is not None else {'HJR235_4': '1', 'HJR309_235': '2'}.get(x))
    return scores

scores = _get_scores(all_scores_agg, min_coverage)
scores_diff = _get_scores(all_scores_diff, 50)
scores_diff_const = _get_scores(all_scores_diff_const, 50)
scores_div = _get_scores(all_scores_div, 50)
scores_div_const = _get_scores(all_scores_div_const, 50)

hue_order = sorted(scores['sgRNA'].unique())
MOTIFS = {
    "minimal": {"CTCF": (2613, 2619), "YY1": (2630, 2636)},
    "unibind": {"CTCF": (2611, 2629), "YY1": (2626, 2638)},
}
min_motif_start = min(motif[0] for motifs in MOTIFS.values() for motif in motifs.values())
max_motif_end = max(motif[1] for motifs in MOTIFS.values() for motif in motifs.values())

for motifs_name, motifs in MOTIFS.items():
    annotate(scores, motifs)
    annotate(scores_diff, motifs)
    annotate(scores_diff_const, motifs)
    annotate(scores_div, motifs)
    annotate(scores_div_const, motifs)
    for classes_name, classes_order, classes_func in [
        ('partialfull', [
            'WT',
            'neither CTCF nor YY1 perturbed',
            'CTCF only partially deleted',
            'CTCF only fully deleted',
            'YY1 only partially deleted',
            'YY1 only fully deleted',
            'both CTCF and YY1 partially deleted',
            'both CTCF and YY1 fully deleted',
        ], label_partial_full),
        ('anydeletion', [
            'WT',
            'neither CTCF nor YY1 deleted',
            'CTCF only deleted',
            'YY1 only deleted',
            'both CTCF and YY1 deleted',
        ], label_partial_only),
    ]:
        classes_func(scores)
        classes_func(scores_diff)
        classes_func(scores_diff_const)
        classes_func(scores_div)
        classes_func(scores_div_const)
        for scores_df, scores_kind, ylabel in zip([scores, scores_diff, scores_diff_const, scores_div, scores_div_const], ['agg', 'diff', 'diffconst', 'div', 'divconst'], ['Num', 'Diff', 'Diff', 'Ratio', 'Ratio']):
            _plot_dir = os.path.join(plot_dir, classes_name + '_' + motifs_name + '_' + scores_kind)
            os.makedirs(_plot_dir, exist_ok=True)

            fig = plot_schematic(
                motifs=motifs,
                ref_seq=ref_seq,
                genomic_coords=genomic_coords,
                window_in_amplicon=(min_motif_start - 10, max_motif_end + 10),
            )
            fig.savefig(os.path.join(_plot_dir, 'motif_schematic.pdf'), bbox_inches='tight')
            plt.close(fig)

            for _scores_name, _scores in {
                "all": scores_df,
                "HJR235": scores_df[scores_df['dataset_name'] == 'HJR235_4'],
                "HJR309_single": scores_df[(scores_df['dataset_name'] != 'HJR235_4') & (scores_df['sgRNA'] != 'pool')],
                "HJR309_all": scores_df[scores_df['dataset_name'] != 'HJR235_4'],
                "HJR309_pool": scores_df[scores_df['sgRNA'] == 'pool'],
                }.items():
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.swarmplot(data=_scores, x='annotation', y='mean', hue='rep', ax=ax, dodge=True, order=classes_order)
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
                    tick.set_ha('right')
                ax.legend(title='rep', bbox_to_anchor=(1.02, 1), loc='upper left')
                ax.set_ylabel(f"{ylabel} DddA edits in accessible region\n(peak {peak_i} = {genomic_coords[0]}:{genomic_coords[1] + peak_start:,}-{genomic_coords[1] + peak_end:,})")
                if scores_kind.startswith('diff'):
                    ax.axhline(0, color='gray', linestyle='--')
                elif scores_kind.startswith('div'):
                    ax.axhline(1, color='gray', linestyle='--')
                fig.savefig(os.path.join(_plot_dir, f"boxplot_{_scores_name}.pdf"), bbox_inches='tight')
                plt.close(fig)

                fig, (ax1, ax2) = plt.subplots(figsize=(6, 6), nrows=2, sharex=True)
                sns.swarmplot(data=_scores[_scores['rep'] == '1'], x='annotation', y='mean', hue='sgRNA', ax=ax1, dodge=False, order=classes_order, hue_order=hue_order)
                sns.swarmplot(data=_scores[_scores['rep'] == '2'], x='annotation', y='mean', hue='sgRNA', ax=ax2, dodge=False, order=classes_order, hue_order=hue_order, legend=False)
                for tick in ax2.get_xticklabels():
                    tick.set_rotation(45)
                    tick.set_ha('right')
                ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
                ax1.set_ylabel(f"{ylabel} DddA edits in accessible region (rep 1)\n(peak {peak_i} = {genomic_coords[0]}:{genomic_coords[1] + peak_start:,}-{genomic_coords[1] + peak_end:,})")
                ax2.set_ylabel(f"{ylabel} DddA edits in accessible region (rep 2)\n(peak {peak_i} = {genomic_coords[0]}:{genomic_coords[1] + peak_start:,}-{genomic_coords[1] + peak_end:,})")
                if scores_kind.startswith('diff'):
                    ax1.axhline(0, color='gray', linestyle='--')
                    ax2.axhline(0, color='gray', linestyle='--')
                elif scores_kind.startswith('div'):
                    ax1.axhline(1, color='gray', linestyle='--')
                    ax2.axhline(1, color='gray', linestyle='--')
                fig.savefig(os.path.join(_plot_dir, f"boxplotByGuide_{_scores_name}.pdf"), bbox_inches='tight')
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(4, 4))
            scores_pivoted = scores_df.pivot(columns='rep', values='mean', index=['sgRNA', 'del_start', 'del_len']).dropna().reset_index()
            annotate(scores_pivoted, motifs)
            classes_func(scores_pivoted)
            sns.scatterplot(data=scores_pivoted, x='1', y='2', hue='annotation', ax=ax, hue_order=classes_order)
            ax.set_xlabel(f"{ylabel} DddA edits in accessible region (rep 1)\n(peak {peak_i} = {genomic_coords[0]}:{genomic_coords[1] + peak_start:,}-{genomic_coords[1] + peak_end:,})")
            ax.set_ylabel(f"{ylabel} DddA edits in accessible region (rep 2)\n(peak {peak_i} = {genomic_coords[0]}:{genomic_coords[1] + peak_start:,}-{genomic_coords[1] + peak_end:,})")
            xrange = ax.get_xlim()
            yrange = ax.get_ylim()
            equal_range = (min(xrange[0], yrange[0]), max(xrange[1], yrange[1]))
            ax.set_xlim(equal_range)
            ax.set_ylim(equal_range)
            ax.plot(equal_range, equal_range, color='gray', linestyle='--')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            fig.savefig(os.path.join(_plot_dir, "scatterplot.pdf"), bbox_inches='tight')
            plt.close(fig)
