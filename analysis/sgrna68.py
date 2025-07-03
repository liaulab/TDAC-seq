import pandas as pd
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import numpy as np
import matplotlib.pyplot as plt
plt.set_loglevel("WARNING")
import os
from tdac_seq.analysis.utils import parse_guides_and_props, prepare_matplotlib
prepare_matplotlib()
from tdac_seq.refseq import extract_ref_seq_from_fasta
import pickle
from tdac_seq.utils import smooth
import scipy.sparse
import seaborn as sns
from Bio.Seq import complement
import scipy.stats

hybrid_start = 4136
hybrid_end = 9064
reverse = True
smoothing_window = 150
NUM_GENOTYPES_TO_PLOT = range(3, 7)

parent_working_dir_prefix = "output/"
working_dir = os.path.join(parent_working_dir_prefix, "sgrna68")
os.makedirs(working_dir, exist_ok=True)

regions = pd.read_csv('data/regions.tsv', sep='\t', index_col="name")
datasets = pd.read_csv('data/datasets.tsv', sep='\t', index_col="ID")
datasets['regions'] = datasets['regions'].str.split(',')
datasets['working_dir'] = parent_working_dir_prefix + datasets.index

plot_dir = f"plots/sgrna68"
os.makedirs(plot_dir, exist_ok=True)

locus = "HBG"
sgrna_info = "sgRNA68:ACTGAATCGGAACAAGGCAA"
ref_seq = extract_ref_seq_from_fasta('data/ref.fa',locus)[locus]
seq_len = len(ref_seq)
genomic_coords = regions.loc[locus, ['chr', 'start', 'end']].values

def _get_profile(ddda_edits: scipy.sparse.csr_array, deletion: slice = None, hybrid: bool = False) -> np.ndarray:
    # assert ddda_edits[:, deletion].sum() == 0
    # get reads with top 10% editing
    num_edits = ddda_edits.sum(axis=1)
    top_reads = num_edits > np.percentile(num_edits, 90)
    mean_edits = smooth(ddda_edits[top_reads].mean(axis=0), how='sum', window=smoothing_window)
    if hybrid:
        mean_edits_ = np.zeros(seq_len)
        mean_edits_[:hybrid_start] = mean_edits[:hybrid_start]
        mean_edits_[hybrid_end:] = mean_edits[hybrid_start:]
        mean_edits = mean_edits_
    if deletion is not None:
        mean_edits[deletion] = np.nan
    return mean_edits

def load_data(id: str, sgrna68: bool = False):
    if sgrna68:
        locus = "HBG_sgRNA68"
    else:
        locus = "HBG"
    data_info = datasets.loc[id].copy()
    with open(os.path.join(data_info['working_dir'], "ddda_data.pkl"), "rb") as f:
        ddda_data = pickle.load(f)
    ddda_edits = ddda_data.edit_dict[locus]
    read_ids = ddda_data.read_ids[locus]
    if sgrna68:
        assert ddda_edits.shape[1] == seq_len - (hybrid_end - hybrid_start)
    else:
        assert ddda_edits.shape[1] == seq_len
    return ddda_edits, read_ids

for id, wt_id in [
    ("HJR288_sgRNA68_D8", "HJR288_sgLuc_D8"),
    ("HJR288_sgRNA68_D10_1", "HJR288_sgLuc_D10_1"),
    ("HJR288_sgRNA68_D10_2", "HJR288_sgLuc_D10_2"),
    (["HJR288_sgRNA68_D10_1", "HJR288_sgRNA68_D10_2"], ["HJR288_sgLuc_D10_1", "HJR288_sgLuc_D10_2"]),
    ("HJR288_sgRNA68_D10_NoDddA", "HJR288_sgLuc_D10_NoDddA"),
    ("HJR288_sgRNA68_D12_1", "HJR288_sgLuc_D12_1"),
    ("HJR288_sgRNA68_D12_2", "HJR288_sgLuc_D12_2"),
    (["HJR288_sgRNA68_D12_1", "HJR288_sgRNA68_D12_2"], ["HJR288_sgLuc_D12_1", "HJR288_sgLuc_D12_2"]),
    ("HJR288_sgRNA68_D12_NoDddA", "HJR288_sgLuc_D12_NoDddA"),
    (["HJR322_3", "HJR322_4"], ["HJR322_1", "HJR322_2"]),
    ]:
    if isinstance(id, str):
        id = [id]
    if isinstance(wt_id, str):
        wt_id = [wt_id]

    logging.info(f"Processing {'_'.join(id)}")

    _plot_dir = os.path.join(plot_dir, f"{'_'.join(id)}")
    os.makedirs(_plot_dir, exist_ok=True)

    # Load sgRNA sequences
    guides = parse_guides_and_props(sgrna_info, ref_seq)

    profiles_wt = np.stack([_get_profile(load_data(_id)[0]) for _id in wt_id])
    assert profiles_wt.shape[1] == seq_len
    assert profiles_wt.shape[0] == len(wt_id)

    profiles_68 = np.full((max(NUM_GENOTYPES_TO_PLOT), len(id), seq_len), np.nan)
    for j, id_68 in enumerate(id):
        ddda_edits_68, read_ids_68 = load_data(id_68, sgrna68=True)

        # Get CRISPR deletion genotype
        aln_fname = os.path.join(parent_working_dir_prefix, id_68, "del", "aln.tsv")
        logging.info(f"Loading {aln_fname}")
        scores_df = pd.read_csv(aln_fname, sep='\t', index_col=[0, 1, 2], usecols=[0, 1, 2])
        assert list(scores_df.index.names) == ['read_id', 'del_start', 'del_len']
        assert all((scores_df.index.get_level_values('del_len') == 0) == (scores_df.index.get_level_values('del_start') == 0))

        # Aggreagte reads by genotype
        idx = pd.IndexSlice
        scores_agg = scores_df.groupby(['del_start', 'del_len']).size().sort_values(ascending=False).rename('count').to_frame()
        scores_agg['freq'] = scores_agg['count'] / scores_agg['count'].sum()
        scores_agg.reset_index(inplace=True)

        # Aggreagate DddA editing across reads
        for i, genotype in enumerate(scores_agg.head(max(NUM_GENOTYPES_TO_PLOT)).itertuples()):
            logging.info(f"Genotype {i+1}: {genotype.del_len}-bp deletion at {genotype.del_start}")
            reads = scores_df.loc[idx[:, genotype.del_start, genotype.del_len], :]
            read_ids = set(reads.index.get_level_values('read_id').to_series().str.lstrip('@')) & set(read_ids_68)
            read_idx = [read_ids_68.index(read_id) for read_id in read_ids]
            profiles_68[i, j] = _get_profile(ddda_edits_68[read_idx], deletion=slice(genotype.del_start, genotype.del_start + genotype.del_len), hybrid=True)

    if len(id) == 2:
        max_num_genotypes = max(NUM_GENOTYPES_TO_PLOT)
        fig, axes = plt.subplots(ncols=max_num_genotypes, figsize=(max_num_genotypes * 3, 3))
        for i, (ax, _profile_68, genotype) in enumerate(zip(axes, profiles_68, scores_agg.itertuples())):
            ax.scatter(*_profile_68, marker='o', s=1, linewidth=0, alpha=0.5, rasterized=True)
            r, _ = scipy.stats.pearsonr(*_profile_68[:, ~np.isnan(_profile_68).any(axis=0)])
            ax.set_title(f"Genotype {i + 1}:\n{genotype.del_len}-bp deletion\n({genotype.count:,} reads {genotype.freq:.1%})\nPearson r = {r:.4f}")
            ax.set_xlabel("Rep 1")
            ax.set_ylabel("Rep 2")
            # make the axes square
            max_val = np.nanmax(_profile_68)
            ax.set_xlim(-0.05 * max_val, 1.05 * max_val)
            ax.set_ylim(-0.05 * max_val, 1.05 * max_val)
            # make ticks integers
            ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
        sns.despine(fig=fig)
        fig.savefig(os.path.join(_plot_dir, 'correlation_reps.pdf'), bbox_inches='tight')
        plt.close(fig)

    for num_genotypes in NUM_GENOTYPES_TO_PLOT:
        logging.info(f"Plotting top {num_genotypes} genotypes")
        fig, axes = plt.subplots(nrows=num_genotypes, ncols=3, figsize=(12, 3))
        max_yval = max(np.nanmax(profiles_wt), np.nanmax(profiles_68[:num_genotypes]))
        for (i, genotype), (ax_allele1, ax_allele2, ax) in zip(enumerate(scores_agg.head(num_genotypes).itertuples()), axes):
            for wt_profile, alpha in zip(profiles_wt, np.linspace(0.5 if len(wt_id) > 1 else 1, 1, len(wt_id))):
                ax.plot(wt_profile, color="gray", alpha=alpha)
            for d68_profile, alpha in zip(profiles_68[i], np.linspace(0.5 if len(id) > 1 else 1, 1, len(id))):
                ax.plot(d68_profile, color="orange", alpha=alpha)
            ax.set_xlim(0, seq_len)
            window2 = slice(4124, 4146)
            window1 = slice(9055, 9078)
            hspace_ratio = 0.3
            ax_allele1.barh(y=0, width=genotype.del_len, left=genotype.del_start - 0.5, color="orange", height=hspace_ratio)
            ax_allele1.set_xlim(window1.start - 0.5, window1.stop - 0.5)
            ax_allele1.set_ylim(-0.5, 0.5)
            ax_allele2.barh(y=0, width=genotype.del_len, left=genotype.del_start - 0.5, color="orange", height=hspace_ratio)
            ax_allele2.set_xlim(window2.start - 0.5, window2.stop - 0.5)
            ax_allele2.set_ylim(-0.5, 0.5)
            for x, c in enumerate(ref_seq if not reverse else complement(ref_seq)):
                if x in range(genotype.del_start, genotype.del_start + genotype.del_len):
                    alpha = 0.3
                else:
                    alpha = 1
                if x in range(window1.start, window1.stop):
                    ax_to_plot = ax_allele1
                elif x in range(window2.start, window2.stop):
                    ax_to_plot = ax_allele2
                else:
                    continue
                ax_to_plot.text(x, 0, c, ha='center', va='center', alpha=alpha)
            ax_allele1.set_yticks([0])
            ax_allele1.set_yticklabels([f"{genotype.count:,} reads ({genotype.freq:.1%}): {genotype.del_len}-bp deletion"])
            ax_allele2.yaxis.set_visible(False)
            _genomic_coords = genomic_coords.copy()
            if reverse:
                ax.invert_xaxis()
                ax_allele1.invert_xaxis()
                ax_allele2.invert_xaxis()
                _genomic_coords = (genomic_coords[0], genomic_coords[2], genomic_coords[1])
            sns.despine(ax=ax)
            sns.despine(ax=ax_allele1, left=True)
            sns.despine(ax=ax_allele2, left=True)
            if genotype.Index < num_genotypes - 1:
                if genotype.Index == 0:
                    ax.set_title(f"Num DddA edits in\n{smoothing_window}-bp sliding window")
                ax_allele1.xaxis.set_visible(False)
                ax_allele2.xaxis.set_visible(False)
                ax.set_xticklabels([])
                ax_allele1.spines['bottom'].set_visible(False)
                ax_allele2.spines['bottom'].set_visible(False)
            else:
                guide_ticks1 = [(guide.cutsite - 0.5, guide.Index) for guide in guides.itertuples() if guide.cutsite in range(window1.start, window1.stop)]
                guide_ticks2 = [(guide.cutsite - 0.5, guide.Index) for guide in guides.itertuples() if guide.cutsite in range(window2.start, window2.stop)]
                ax_allele1.set_xticks(tuple(zip(*guide_ticks1))[0], minor=True)
                ax_allele1.set_xticklabels(tuple(zip(*guide_ticks1))[1], rotation=45, minor=True, ha='right', va='top')
                ax_allele1.tick_params(axis='x', which='minor', length=20)
                ax_allele2.set_xticks(tuple(zip(*guide_ticks2))[0], minor=True)
                ax_allele2.set_xticklabels(tuple(zip(*guide_ticks2))[1], rotation=45, minor=True, ha='right', va='top')
                ax_allele2.tick_params(axis='x', which='minor', length=20)
                ax.set_xlabel(f"Position in amplicon\n{_genomic_coords[0]}:{_genomic_coords[1]:,}-{_genomic_coords[2]:,}")
        for _, _, ax in axes:
            ax.set_ylim(0, 1.05 * max_yval)
        plot_fname = os.path.join(_plot_dir, f"top_{num_genotypes}_genotypes.pdf")
        fig.savefig(plot_fname, bbox_inches='tight')
        plt.close(fig)

        logging.info(f"Done making plot {plot_fname}")
