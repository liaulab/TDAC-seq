import plotly.io
plotly.io.templates.default = "plotly_white"
import plotly.express as px
import pandas as pd
import os
import matplotlib.pyplot as plt
from tdac_seq.analysis.utils import prepare_matplotlib
prepare_matplotlib()
import scipy.stats
import numpy as np
from tdac_seq.constants import get_motifs
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

gfi1b_motifs = get_motifs('GFI1B')
for rep1, rep2, is_abe, motifs in [
    ('HJR312_MGY_rep1', 'HJR312_MGY_rep2', False, {motif_name: gfi1b_motifs['motif'][motif_name] for motif_name in gfi1b_motifs['motif_highlight']}),
    ('HJR244_1', 'HJR244_2', False, {
        'AP-1': (1488, 1498),
    }),
    ('HJR244_3', 'HJR244_4', True, {
        'FOXK2': (1410, 1411),
        'AP-1': (1492, 1493),
    }),
    ]:
    plot_dir = os.path.join('plots', 'correlate_reps', rep1 + '_' + rep2)
    os.makedirs(plot_dir, exist_ok=True)

    logging.info('Loading volcano data')
    df_volcano = pd.concat([pd.read_csv(os.path.join('output', rep, 'volcano(raw).tsv'), sep='\t', index_col='abe_edits' if is_abe else ['deletion start', 'deletion length']) for rep in [rep1, rep2]], keys=[rep1, rep2], names=['reps'])
    if is_abe:
        df_volcano.reset_index(inplace=True)
        df_volcano['abe_edits'] = df_volcano['abe_edits'].str.slice(1, -1).str.split(',').apply(lambda x: tuple(int(i) for i in x if i != ''))
        df_volcano.set_index(['reps', 'abe_edits'], inplace=True)
    df_volcano = df_volcano.unstack('reps')['diff edits']

    logging.info('Loading abundance data')
    df_abundance = pd.concat([pd.read_csv(os.path.join('output', rep, 'genotypes.tsv'), sep='\t', index_col='read') for rep in [rep1, rep2]], keys=[rep1, rep2], names=['reps'])
    df_abundance.drop(df_abundance.index[~df_abundance['del_only']], inplace=True)
    df_abundance = df_abundance.groupby(['reps', 'del_start', 'del_len']).size().rename('count')
    df_abundance = df_abundance.unstack('reps', fill_value=0)
    df_abundance.rename_axis(index={'del_start': 'deletion start', 'del_len': 'deletion length'}, inplace=True)
    df_abundance = df_abundance.loc[df_abundance.index.isin(df_volcano.index)]

    if is_abe:
        wt_genotype = tuple()
    else:
        wt_genotype = (0, 0)

    for label, axis_label in [('abundance', 'Num reads'), ('volcano', 'Diff DddA edits in accessible region')]:
        logging.info(f'Plotting {label} for {rep1} and {rep2}')
        if label == 'abundance':
            df = df_abundance
        elif label == 'volcano':
            df = df_volcano
        else:
            raise ValueError(f"Unknown label: {label}")
        logging.info(f'Loaded {df.shape[0]} genotypes')
        wt = df.loc[[wt_genotype]]
        df = df.drop(wt_genotype).reset_index()
        if is_abe:
            df['abe edit position (median)'] = df['abe_edits'].apply(lambda x: np.median(x))
        df['annotation'] = 'Other'
        _motifs = {f'{motif_name} ({motif_range[0]}-{motif_range[1]})': motif_range for motif_name, motif_range in motifs.items()}
        for motif_name, (start, end) in _motifs.items():
            if is_abe:
                mask = df['abe_edits'].apply(lambda x: any([start <= pos < end for pos in x]))
            else:
                mask = (df['deletion start'] <= start) & (df['deletion start'] + df['deletion length'] >= end)
            df.loc[mask, 'annotation'] = motif_name
        motifs_ordered = sorted(_motifs.keys(), key=_motifs.get) + ['Other']
        df['annotation'] = pd.Categorical(df['annotation'], categories=motifs_ordered)
        df.sort_values('annotation', inplace=True, ascending=False) # so that other is plotted at the bottom zorder

        for plot_suffix, color_kwargs in {
            "": dict(color='abe edit position (median)' if is_abe else 'deletion start', color_continuous_scale='rainbow'),
            "_by_motif": dict(color='annotation', color_discrete_sequence=px.colors.qualitative.Plotly[:len(motifs_ordered) - 1] + ['lightgray'], category_orders={'annotation': motifs_ordered}),
            }.items():
            fig = px.scatter(
                df,
                x=rep1,
                y=rep2,
                hover_data=['abe_edits', 'annotation'] if is_abe else ['deletion start', 'deletion length', 'annotation'],
                trendline="ols",
                trendline_scope="overall",
                trendline_color_override='black',
                labels={'1': f'{axis_label} (rep 1)', '2': f'{axis_label} (rep 2)'},
                **color_kwargs,
            )
            if label != 'abundance':
                fig.add_trace(px.scatter(
                    x=wt[rep1],
                    y=wt[rep2],
                    hover_name=['wildtype'],
                    color_discrete_sequence=['black'],
                ).update_traces(marker=dict(size=20)).data[0])
            fig.write_html(os.path.join(plot_dir, f'scatter_{label}{plot_suffix}.html'))

            fig, ax = plt.subplots(figsize=(4, 4))
            if plot_suffix == '':
                cmap = plt.cm.rainbow
                norm = plt.Normalize()
                colors = cmap(norm(df['abe edit position (median)' if is_abe else 'deletion start']))
                fig.colorbar(
                    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax,
                    label='deletion start',
                    orientation='vertical',
                )
            else:
                cmap = color_kwargs['color_discrete_sequence']
                colors = [cmap[motifs_ordered.index(motif)] for motif in df['annotation']]
                # legend
                if label != 'abundance':
                    ax.scatter([], [], color='black', marker='*', label='Wildtype')
                for motif in motifs_ordered:
                    legend_label = motif
                    if label == 'abundance':
                        mask = df['annotation'] == motif
                        legend_label += f'\n({mask.sum()} genotypes, {df.loc[mask, [rep1, rep2]].sum().sum():,d} reads)'
                    ax.scatter([], [], color=cmap[motifs_ordered.index(motif)], label=legend_label)
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
                if is_abe:
                    for _, row in df.iterrows():
                        if row['annotation'] == 'Other':
                            continue
                        ax.annotate(
                            '+'.join(map(str, row['abe_edits'])),
                            xy=(row[rep1], row[rep2]),
                            xytext=(-1, 1),
                            textcoords='offset points',
                            fontsize=4,
                            color=cmap[motifs_ordered.index(row['annotation'])],
                            rotation=-45,
                            ha='right',
                            va='bottom',
                        )
            ax.scatter(
                df[rep1],
                df[rep2],
                c=colors,
                linewidth=0.1,
                edgecolor='white',
                s=5,
                marker='o',
            )
            r, _ = scipy.stats.pearsonr(
                df.dropna()[rep1],
                df.dropna()[rep2],
            )
            ax.set_title(f'Pearson r = {r:.2f}')
            ax.set_xlabel(f'{axis_label} (rep 1)')
            ax.set_ylabel(f'{axis_label} (rep 2)')
            if label != 'abundance':
                ax.scatter(
                    wt[rep1],
                    wt[rep2],
                    marker='*',
                    linewidth=0.1,
                    edgecolor='white',
                    s=50,
                    color='black',
                )
            else:
                ax.set_xscale('log')
                ax.set_yscale('log')
            fig.savefig(os.path.join(plot_dir, f'scatter_{label}{plot_suffix}.pdf'), bbox_inches='tight')
            plt.close(fig)
