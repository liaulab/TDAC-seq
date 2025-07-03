import pandas as pd
import pickle
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import numpy as np
from tdac_seq.ddda_dataset import ddda_dataset
import matplotlib.pyplot as plt
from tdac_seq.analysis.utils import prepare_matplotlib
prepare_matplotlib()
from tdac_seq.utils import smooth
import scipy.stats
import pyBigWig

plots_dir = os.path.join('plots', 'correlate_reps', 'HJR306')
os.makedirs(plots_dir, exist_ok=True)

reps = pd.read_csv('data/reps_HJR306.csv')
reps['ugi'] = reps['ugi'].map({True: '+', False: '−'})

regions = pd.read_csv('data/regions.tsv', sep='\t', index_col="name")
locus = 'LCR_new'
genomic_coords = regions.loc[locus][['chr', 'start', 'end']].values

def get_edits(dataset_id: str):
    with open(os.path.join('output', dataset_id, 'ddda_data.pkl'), 'rb') as f:
        ddda_data: ddda_dataset = pickle.load(f)
    deduped_inds = getattr(ddda_data, 'deduped_inds', dict()).get(locus, None)
    if deduped_inds is None:
        logging.warning(f'No deduped_inds in {dataset_id}, so using all reads')
        deduped_inds = slice(None)
    return ddda_data.edit_dict[locus][deduped_inds]

atac = pyBigWig.open("https://www.encodeproject.org/files/ENCFF357GNC/@@download/ENCFF357GNC.bigWig")
atac_vals = np.array(atac.values(*genomic_coords))

for group, reps_group in reps.groupby('group'):
    logging.info(f'Processing group {group}')
    fig, (ax_atac, *axes) = plt.subplots(nrows=len(reps_group) + 1, ncols=1, figsize=(4, 0.5 * (len(reps_group) + 1)), sharex=True, gridspec_kw=dict(hspace=0.3))
    ax_atac.fill_between(x=np.arange(len(atac_vals)), y1=0, y2=atac_vals, color='gray', linewidth=0)
    ax_atac.set_ylim(bottom=0)
    ax_atac.set_ylabel('ATAC', rotation=0, va='center', ha='right')
    ymax = 0
    if reps_group['ugi'].nunique() == 1:
        label_format = '{enzyme:s} {um}μM / {min}min'
    else:
        label_format = '{enzyme:s} {um}μM / {min}min / UGI {ugi}'
    for i, (ax, pair) in enumerate(zip(axes, reps_group.itertuples())):
        tdac_profile = np.full((2, genomic_coords[2] - genomic_coords[1] + 1), np.nan)
        for j, (dataset_id, color) in enumerate(zip([pair.rep1, pair.rep2], [plt.cm.tab10(0), plt.cm.tab10(3)])):
            try:
                edits = get_edits(dataset_id)
            except Exception as e:
                logging.error(f'Failed to load edits for {dataset_id}:', exc_info=True)
                continue
            tdac_profile[j] = smooth(edits.mean(axis=0), how='sum')
            ax.fill_between(x=np.arange(edits.shape[1]), y1=0, y2=tdac_profile[j], color=color, alpha=0.4, linewidth=0)
        ax.annotate(label_format.format(**pair._asdict()), xy=(1, 0.9), xycoords='axes fraction', ha='right', va='top', fontsize=8)
        r, _ = scipy.stats.pearsonr(*tdac_profile)
        ax.annotate(f'Pearson r = {r:.4f}', xy=(1, 0.5), xycoords='axes fraction', ha='right', va='top', fontsize=6, color='gray')
        if i == len(reps_group) // 2:
            ax.set_ylabel('Num DddA edits in 100-bp window')
        ymax = max(ymax, tdac_profile.max())
        if i == len(reps_group) - 1:
            ax.set_xlabel(f'Position in locus\n{genomic_coords[0]}:{genomic_coords[1]}-{genomic_coords[2]}')
            ax.set_xlim(0, genomic_coords[2] - genomic_coords[1])
        else:
            ax.xaxis.set_visible(False)
    ymax = np.ceil(ymax * 1.05)
    for ax in axes:
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, ymax])
    plot_fname = os.path.join(plots_dir, f'conditions_{group}.pdf')
    fig.savefig(plot_fname, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'Saved plot to {plot_fname}')
