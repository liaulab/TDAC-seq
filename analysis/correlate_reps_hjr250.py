import pandas as pd
import itertools
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import matplotlib.pyplot as plt
plt.set_loglevel("WARNING")
from tdac_seq.analysis.utils import prepare_matplotlib
prepare_matplotlib()
from tdac_seq.utils import smooth
from tdac_seq.ddda_dataset import ddda_dataset
import pickle
import itertools
import scipy.stats

plots_dir = 'plots/correlate_reps/HJR250'
os.makedirs(plots_dir, exist_ok=True)

parent_working_dir_prefix = "output/"
regions = pd.read_csv('data/regions.tsv', sep='\t', index_col="name")
datasets = pd.read_csv('data/datasets.tsv', sep='\t', index_col="ID")
datasets['regions'] = datasets['regions'].str.split(',')
datasets['working_dir'] = parent_working_dir_prefix + datasets.index
assert set(itertools.chain.from_iterable(datasets['regions'])).issubset(regions.index), "Some regions are not in the regions.tsv file"

def _load_rep(id):
    data_info = datasets.loc[id].copy()
    working_dir = data_info.working_dir

    object_fname = os.path.join(working_dir, "ddda_data.pkl")
    logging.info(f"Loading pre-aligned data from {object_fname}")
    with open(object_fname, "rb") as f:
        ddda_data: ddda_dataset = pickle.load(f)

    assert len(ddda_data.ref_seq_dict) == 1

    locus, _ = next(iter(ddda_data.ref_seq_dict.items()))
    edits = ddda_data.edit_dict[locus]
    genomic_region: tuple[str, int, int] = ddda_data.region_dict[locus]

    deduped_inds = getattr(ddda_data, "deduped_inds", {locus: None})[locus]
    if deduped_inds is None:
        logging.warning("No deduped inds found. Using all reads")
        deduped_inds = slice(None)

    track = edits[deduped_inds]
    track = smooth(track.mean(axis=0), window=100, how='sum').squeeze()
    return track, locus, genomic_region

df_reps = pd.read_csv('data/reps_HJR250.csv')
fig, axes = plt.subplots(nrows=len(df_reps), figsize=(4, 1.5 * len(df_reps)), gridspec_kw=dict(hspace=1))
for reps, ax in zip(df_reps.itertuples(), axes):
    logging.info(f"Loading {reps.rep1} and {reps.rep2}")
    try:
        track1, locus1, genomic_region1 = _load_rep(reps.rep1)
        track2, locus2, genomic_region2 = _load_rep(reps.rep2)
        assert locus1 == locus2
        assert genomic_region1 == genomic_region2
        color_rep1 = 'tab:blue'
        color_rep2 = 'tab:red'
        ax.plot(track1, color=color_rep1)
        ax.set_ylabel(reps.rep1, color=color_rep1, rotation=0, ha='right', va='center')
        ax.tick_params(axis='y', labelcolor=color_rep1, color=color_rep1)
        ax2 = ax.twinx()
        ax2.plot(track2, color=color_rep2)
        ax2.set_ylabel(reps.rep2, color=color_rep2, rotation=0, ha='left', va='center')
        ax2.tick_params(axis='y', labelcolor=color_rep2, color=color_rep2)
        r, _ = scipy.stats.pearsonr(track1, track2)
        ax.set_title(f"{genomic_region1[0]}:{genomic_region1[1]:,}-{genomic_region1[2]:,}" + (f" ({locus1})" if not locus1.startswith('chr') else '') + f"\n(Pearson r={r:.3f})")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, integer=True))
        ax.set_xlim(0, len(track1))
        ax2.set_xlim(0, len(track2))
        ax.set_ylim(0, track1.max() * 1.05)
        ax2.set_ylim(0, track2.max() * 1.05)
    except Exception as e:
        logging.error(f"Error loading {reps.rep1} and {reps.rep2}", exc_info=e)
        continue
fig.savefig(os.path.join(plots_dir, 'individual_tracks.pdf'), bbox_inches='tight')
plt.close(fig)
