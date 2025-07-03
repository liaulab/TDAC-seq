import pandas as pd
from tdac_seq.ddda_dataset import ddda_dataset
import pickle
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
from tdac_seq.dedup import UmiIterator
import numpy as np
from seaborn.utils import relative_luminance
import os
import sys
import matplotlib.pyplot as plt
from tdac_seq.analysis.utils import prepare_matplotlib
prepare_matplotlib()
import matplotlib.colors
import matplotlib.patches

def main(dataset_id: int):
    datasets = pd.read_csv('data/datasets.tsv', sep='\t', index_col="ID")

    data_info = datasets.iloc[dataset_id]
    id = data_info.name
    logging.info(f"Processing dataset {id}")

    if data_info['sgRNA'] == 'none':
        logging.error('Not CRISPR treated')
        return
    elif data_info['ABE']:
        logging.error('ABE not supported')
        return

    plots_dir = os.path.join('plots', id)
    os.makedirs(plots_dir, exist_ok=True)

    with open(f'output/{id}/ddda_data.pkl', 'rb') as f:
        ddda_data: ddda_dataset = pickle.load(f)
    assert len(ddda_data.ref_seq_dict) == 1
    locus = list(ddda_data.ref_seq_dict.keys())[0]
    ref_seq = ddda_data.ref_seq_dict[locus]
    edits = ddda_data.edit_dict[locus]
    read_inds = np.arange(10000)

    threshold = int(0.01 * sum(1 for c in ref_seq[500:-500] if c in 'CG'))

    umi_groups = UmiIterator(
        edits=edits,
        read_inds=read_inds,
        threshold=threshold,
        )

    df_ = {}
    for read_ind, umi_group in umi_groups:
        if read_ind is None:
            break
        df_[read_ind] = umi_group

    df = pd.DataFrame.from_dict(df_, orient='index', columns=['umi_group']).rename_axis(index='read')
    # read_ids = np.array(ddda_data.read_ids[locus])
    # df['read_id'] = read_ids[df.index]
    df = df.join(pd.read_csv(f'output/{id}/genotypes.tsv', sep='\t', index_col='read'), how='left', on='read')

    max_umi_count = df.groupby('umi_group').value_counts().groupby('umi_group').max()
    all_umi_count = df.groupby('umi_group').value_counts().groupby('umi_group').sum()

    max_count = 10
    big_umi_mask = (max_umi_count > max_count) | (all_umi_count > max_count)
    big_umi_i = max_umi_count.index[big_umi_mask]
    assert len(big_umi_i) == 1
    big_umi_i = big_umi_i[0]

    for suffix in ['', '_all']:
        xcenters = np.arange(1, max_count + 1)
        ycenters = np.arange(1, max_count + 1)
        if suffix == '_all':
            xcenters[-1] = max_umi_count[big_umi_i]
            ycenters[-1] = all_umi_count[big_umi_i]
        hist = np.histogram2d(
            x=max_umi_count,
            y=all_umi_count,
            bins=(np.append(xcenters, xcenters[-1] + 1), np.append(ycenters, ycenters[-1] + 1)),
            )[0]

        fig, ax = plt.subplots(figsize=(len(xcenters) / 3, len(xcenters) / 3))
        norm = matplotlib.colors.LogNorm()
        cmap = plt.cm.Greens
        ax.pcolormesh(hist.T, cmap=cmap, norm=norm)
        ax.set_xlabel('Number of reads of the most\nabundant CRISPR genotype per UMI')
        ax.set_ylabel('Number of reads per UMI')
        for i, x in enumerate(xcenters):
            for j, y in enumerate(ycenters):
                if hist[i, j] > 0:
                    ax.text(i + 0.5, j + 0.5, f"{hist[i, j]:.0f}",
                        ha='center',
                        va='center',
                        color='black' if relative_luminance(cmap(norm(hist[i, j]))) > 0.5 else 'white',
                        fontsize=8,
                    )
                if x == y:
                    # red box around entries on the diagonal
                    ax.add_patch(matplotlib.patches.Rectangle(
                        xy=(i, j),
                        width=1,
                        height=1,
                        fill=False,
                        edgecolor='red',
                        linewidth=1,
                        clip_on=False,
                        zorder=99,
                        ))
        ax.set_xlim(0, len(xcenters))
        ax.set_ylim(0, len(xcenters))
        ax.set_xticks(np.arange(len(xcenters)) + 0.5)
        ax.set_yticks(np.arange(len(ycenters)) + 0.5)
        ax.set_xticklabels(xcenters)
        ax.set_yticklabels(ycenters)
        fig.savefig(os.path.join(plots_dir, f'umi_crispr{suffix}.pdf'), bbox_inches='tight')
        plt.close(fig)

    in_big_umi = df['umi_group'] == big_umi_i
    num_edits = edits.sum(axis=1)[read_inds]
    bins = np.histogram_bin_edges(num_edits, bins=100)
    fig, ax = plt.subplots()
    ax.hist(num_edits[in_big_umi], bins=bins, alpha=0.5, label=f'Reads in the single\nproblematic UMI ({in_big_umi.mean():.1%})')
    ax.hist(num_edits[~in_big_umi], bins=bins, alpha=0.5, label=f'All other reads ({(~in_big_umi).mean():.1%})')
    ax.set_xlabel('Number of DddA edits per read')
    ax.set_ylabel('Number of reads')
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(plots_dir, 'umi_problematicone.pdf'), bbox_inches='tight')
    plt.close(fig)

    logging.info(f"Finished processing dataset {id}. Plots saved to {plots_dir}")

if __name__ == "__main__":
    assert len(sys.argv) == 2
    main(int(sys.argv[1]))
