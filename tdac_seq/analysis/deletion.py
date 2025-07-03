import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import numpy as np
import matplotlib.pyplot as plt
plt.set_loglevel("WARNING")
import matplotlib.colors
import seaborn as sns
from tdac_seq.plots import UnibindTrack, ConservationTrack, INCHES_PER_SQUARE, INCHES_PER_ALLELE, _annotate_cutsites
from tdac_seq.ddda_dataset import ddda_dataset
from tdac_seq.utils import diff_DddA_helper
import scipy.signal
from Bio.Seq import complement
from typing import Literal
from tdac_seq.dad import call_dads_hmm_bias
import itertools
import scipy.stats

def plot_accessibility_heatmap_per_deletion(plot_fname: str, window: slice, genomic_coords: tuple[str, int, int], scores: np.ndarray, cmap = None, norm = None, ref_seq: str = None, target_sites: pd.Series = None, wt: float = None, minimal_motif: list[slice] = [], accessible_region: tuple[int, int] = None) -> None:
    _genomic_coords = (genomic_coords[0], genomic_coords[1] + window.start, genomic_coords[1] + window.stop)

    # setup figure
    main_width = INCHES_PER_SQUARE * (window.stop - window.start)
    cbar_width = 0.3
    cbar_wspace = 0.6
    unibind_track = UnibindTrack("", _genomic_coords)
    conservation_track = ConservationTrack("", _genomic_coords)
    fig, ((ax_unibind, ax_empty1), (ax_conservation, ax_empty2), (ax, cax)) = plt.subplots(
        figsize=(main_width + cbar_wspace + cbar_width, unibind_track.height + conservation_track.height + INCHES_PER_SQUARE * scores.shape[1]),
        ncols=2,
        nrows=3,
        gridspec_kw=dict(
            width_ratios=[main_width, cbar_width],
            wspace=cbar_wspace / (main_width + cbar_wspace + cbar_width),
            hspace=0.02,
            height_ratios=[unibind_track.height, conservation_track.height, INCHES_PER_SQUARE * scores.shape[1]],
            ),
        )
    ax_empty1.remove()
    ax_empty2.remove()

    if cmap is None:
        cmap = plt.cm.coolwarm
    if norm is None:
        vmin = min(np.nanpercentile(scores, 5), wt - 0.01)
        vmax = max(np.nanpercentile(scores, 95), wt + 0.01)
        vrange = max(vmax - wt, wt - vmin)
        norm = matplotlib.colors.CenteredNorm(vcenter=wt, halfrange=vrange)

    x = np.arange(window.start, window.stop)
    ax.pcolormesh(x, np.arange(scores.shape[1]), scores.T, cmap=cmap, norm=norm)
    ax.set_ylim(bottom=4.5)
    ax.invert_yaxis()
    ax.set_xlabel("Start of CRISPR deletion\n{}:{:,}-{:,}".format(*_genomic_coords))
    ax.set_ylabel("Length of CRISPR deletion")
    clabel = "Num DddA edits"
    if accessible_region is not None:
        clabel += f" in {genomic_coords[0]}:{genomic_coords[1] + accessible_region[0]:,}-{genomic_coords[1] + accessible_region[1]:,}"
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label=clabel, cax=cax)
    # Optionally annotate things
    if wt is not None:
        cbar.set_ticks([wt], labels=["WT"], minor=True)
        # set minor tick length to same as major
        cbar.ax.tick_params(which='minor', length=plt.rcParams['ytick.major.size'])
    if ref_seq is not None:
        ax.set_xticks(x, labels=[ref_seq[i] for i in range(window.start, window.stop)])
    if target_sites is not None:
        _annotate_cutsites(ax, target_sites)
    for _minimal_motif in minimal_motif:
        for tk in ax.get_xticklabels():
            if tk.get_position()[0] >= _minimal_motif.start and tk.get_position()[0] < _minimal_motif.stop:
                tk.set_color('red')
    ax.set_xlim(window.start - 0.5, window.stop - 0.5)

    unibind_track.plot(fig, ax_unibind)
    conservation_track.plot(fig, ax_conservation)

    fig.savefig(plot_fname, bbox_inches="tight")
    plt.close(fig)

def plot_accessibility_alleletable(plot_fname: str, window: list[slice], genomic_coords: tuple[str, int, int], ref_seq: str, target_sites: pd.Series, scores_agg, min_coverage: int, scores_df, is_abe: bool = False, ddda_data: ddda_dataset = None, reverse_sequence: bool = False, method: Literal["smooth", "dedup+smooth", "dad", "smooth+exclude"] = "smooth", sort_by: Literal["coverage", "allele", "mean"] = "coverage", allele_table: bool = True, ddda_threshold: float | int = None, alleles_wysiwyg: bool = False, extra_info_col: str = None, highlight_dels: bool = False) -> None:
    """
    scores_agg is dataframe where each row is a genotype. It should also have column 'count' for the number of reads with that genotype.
    scores_df is dataframe where each row is a read. It should also have column 'num_edits' for the number of DddA edits in that read.
    If ddda_data is provided, then the heatmap will be plotted instead of boxplot using scores_df.
    """
    if isinstance(window, slice):
        window = [window]

    if alleles_wysiwyg:
        df_plot = scores_agg.reset_index(drop=False)
    else:
        df_plot = scores_agg.sort_values('count', ascending=False).reset_index(drop=False)
    if not is_abe:
        # delete deletions that are outside of the window
        mask = df_plot['del_len'] > 0
        for _window in window:
            mask &= ((df_plot['del_start'] >= _window.stop) | (df_plot['del_start'] + df_plot['del_len'] < _window.start))
        df_plot.drop(df_plot.index[mask], inplace=True)
    df_plot.reset_index(drop=True, inplace=True)
    # sort by num ddda edits. important to reset index counter which is used as row number when plotting
    if not alleles_wysiwyg and sort_by == "mean":
        df_plot.set_index('abe_edits' if is_abe else ['del_start', 'del_len'], inplace=True)
        df_plot['num_edits_mean'] = scores_df.groupby('abe_edits' if is_abe else ['del_start', 'del_len']).mean()
        df_plot = df_plot.drop(index=scores_agg.index[scores_agg['count'] < min_coverage], errors='ignore')
        df_plot = df_plot.sort_values("num_edits_mean", ascending=False).reset_index()
    # subset genotypes to show
    if not alleles_wysiwyg:
        df_plot = df_plot.head(max(5, min(20, sum(scores_agg['count'] >= min_coverage))))

    # sort by mutation. important to reset index counter which is used as row number when plotting
    if not alleles_wysiwyg and sort_by == "allele":
        if is_abe:
            # get positions of all bases edited
            edited_bases = sorted(set(itertools.chain.from_iterable(df_plot['abe_edits'])))
            if not reverse_sequence:
                edited_bases = edited_bases[::-1]
            for _edited_base in edited_bases:
                df_plot[f"edited_base_{_edited_base}"] = df_plot['abe_edits'].apply(lambda x: _edited_base in x)
            df_plot = df_plot.sort_values([f"edited_base_{_edited_base}" for _edited_base in edited_bases]).reset_index(drop=True)
        else:
            df_plot = df_plot.sort_values(['del_start', 'del_len']).reset_index(drop=True)

    # figure sizing and setup
    if ddda_data is not None:
        bar_width = 6
    else:
        bar_width = 3
    if allele_table:
        unibind_tracks = []
        conservation_tracks = []
        _ref_seqs = []
        _target_sites = []
        _genomic_coordss = []
        for _window in window:
            _genomic_coords = (genomic_coords[0], genomic_coords[1] + _window.start, genomic_coords[1] + _window.stop)
            _genomic_coordss.append(_genomic_coords)
            unibind_tracks.append(UnibindTrack("", _genomic_coords))
            conservation_tracks.append(ConservationTrack("", _genomic_coords))
            _ref_seqs.append(ref_seq[_window])
            _target_sites.append(target_sites[(target_sites >= _window.start) & (target_sites < _window.stop)] - _window.start)
        main_width = sum(unibind_track.width for unibind_track in unibind_tracks)
        unibind_track_height = max(unibind_track.height for unibind_track in unibind_tracks)
        conservation_track_height = max(conservation_track.height for conservation_track in conservation_tracks)
        fig, axes = plt.subplots(
            ncols=len(window) + 1,
            nrows=3,
            figsize=(main_width + bar_width, unibind_track_height + conservation_track_height + INCHES_PER_ALLELE * len(df_plot)),
            gridspec_kw=dict(width_ratios=[unibind_track.width for unibind_track in unibind_tracks] + [bar_width], wspace=0.02, hspace=0.02, height_ratios=[unibind_track_height, conservation_track_height, INCHES_PER_ALLELE * len(df_plot)]),
            )
        ax_empty1 = axes[0, -1]
        ax_empty2 = axes[1, -1]
        ax_bar = axes[2, -1]
        ax_empty1.remove()
        ax_empty2.remove()
    else:
        unibind_tracks = [None] * len(window)
        conservation_tracks = [None] * len(window)
        _ref_seqs = [None] * len(window)
        _target_sites = [None] * len(window)
        _genomic_coordss = [None] * len(window)
        fig, ax_bar = plt.subplots(figsize=(bar_width, INCHES_PER_ALLELE * len(df_plot)))

    # accessibility data
    if ddda_data is not None:
        _plot_accessibility_heatmap(fig, ax_bar, df_plot, scores_df, ddda_data=ddda_data, is_abe=is_abe, method=method, ddda_threshold=ddda_threshold)
        if reverse_sequence:
            ax_bar.invert_xaxis()
    else:
        _plot_accessibility_boxplot(fig, ax_bar, df_plot, scores_df, is_abe=is_abe)

    for i, (_window, unibind_track, conservation_track, _ref_seq, _target_sites, _genomic_coords) in enumerate(zip(window, unibind_tracks, conservation_tracks, _ref_seqs, _target_sites, _genomic_coordss)):
        if allele_table:
            ax_allele = axes[2, i]
            ax_unibind = axes[0, i]
            ax_conservation = axes[1, i]
            _plot_allele_table(ax_allele, df_plot, _ref_seq, cutsites=_target_sites, window_start=_window.start, is_abe=is_abe, reverse_sequence=reverse_sequence, extra_info_col=extra_info_col, highlight_dels=highlight_dels)
            unibind_track.plot(fig, ax_unibind)
            conservation_track.plot(fig, ax_conservation)
            if reverse_sequence:
                ax_allele.set_xlabel(f"{_genomic_coords[0]}:{_genomic_coords[2]:,}-{_genomic_coords[1]:,}")
                ax_allele.invert_xaxis()
                ax_unibind.invert_xaxis()
                ax_conservation.invert_xaxis()
            else:
                ax_allele.set_xlabel(f"{_genomic_coords[0]}:{_genomic_coords[1]:,}-{_genomic_coords[2]:,}")
            if i > 0:
                ax_allele.set_yticks([])
                ax_conservation.set_yticks([])
                ax_conservation.set_ylabel("")
                ax_conservation.spines['left'].set_visible(False)
        else:
            ax_bar.set_yticks(range(len(df_plot)))
            if is_abe:
                ax_bar.set_yticklabels([f'{"+".join(ref_seq[editpos] + str(editpos) for editpos in row['abe_edits'])}' if len(row['abe_edits']) > 0 else 'Wildtype' for _, row in df_plot.iterrows()])
            else:
                ax_bar.set_yticklabels([f'∆{row["del_start"]}-{row["del_start"] + row["del_len"]}' if row['del_len'] > 0 else 'Wildtype' for _, row in df_plot.iterrows()])
            # label read coverage on the right y axis
            ax_bar_right = ax_bar.twinx()
            ax_bar_right.set_yticks(range(len(df_plot)))
            ax_bar_right.set_yticklabels([f'{row["freq"]:.1%} ({row["count"]:.0f} read{"s" if row["count"] != 1 else ""})' + row.get(extra_info_col, '') for _, row in df_plot.iterrows()])
            ax_bar_right.set_ylim(ax_bar.get_ylim())

    fig.savefig(plot_fname, bbox_inches="tight")
    plt.close(fig)

def plot_accessibility_alleletable_by_motif(plot_fname: str, window: slice, genomic_coords: tuple[str, int, int], ref_seq: str, target_sites: pd.Series, scores_agg, min_coverage: int, scores_df, motif: tuple[int, int], is_abe: bool = False, reverse_sequence: bool = False, plot_kind: Literal['box', 'bar'] = 'box', highlight_dels: bool = False) -> None:
    """
    scores_agg should be a dataframe where each row is a genotype. It should have columns 'count', 'freq', and if is_abe, 'abe_edits' otherwise 'del_start' and 'del_len'.
    scores_df should be a series where each row is a read. Each entry is number of DddA edits. First level of index is read_id, the remaining levels are 'abe_edits' if is_abe, otherwise 'del_start' and 'del_len'.
    """
    if not isinstance(window, slice):
        if len(window) != 1:
            raise NotImplementedError("Only single window is supported")
        window = window[0]

    _genomic_coords = (genomic_coords[0], genomic_coords[1] + window.start, genomic_coords[1] + window.stop)
    _ref_seq = ref_seq[window]
    _target_sites = target_sites[(target_sites >= window.start) & (target_sites < window.stop)] - window.start

    def _get_df_plot(_scores_agg) -> pd.DataFrame:
        df_plot = _scores_agg.sort_values('count', ascending=False).reset_index(drop=False)
        if not is_abe:
            df_plot.drop(df_plot.index[((df_plot['del_start'] >= window.stop) | (df_plot['del_start'] + df_plot['del_len'] < window.start)) & (df_plot['del_len'] > 0)], inplace=True)
        df_plot.reset_index(drop=True, inplace=True)
        df_plot = df_plot.head(max(2, min(8, sum(_scores_agg['count'] >= min_coverage))))
        return df_plot
    
    scores_agg = scores_agg.reset_index()
    if is_abe:
        motif_deleted = scores_agg['abe_edits'].apply(lambda abe_edits: any(abe_edit >= motif[0] and abe_edit < motif[1] for abe_edit in abe_edits))
    else:
        motif_deleted = (scores_agg['del_start'] <= motif[0]) & (scores_agg['del_start'] + scores_agg['del_len'] >= motif[1])
    df_plot1 = _get_df_plot(scores_agg[motif_deleted])
    df_plot2 = _get_df_plot(scores_agg[~motif_deleted])

    # figure sizing and setup
    unibind_track = UnibindTrack("", _genomic_coords)
    conservation_track = ConservationTrack("", _genomic_coords)
    bar_width = 3
    main_width = unibind_track.width
    fig, ((ax_unibind, ax_empty1), (ax_conservation, ax_empty2), (ax_allele, ax_bar), (ax_allele2, ax_bar2)) = plt.subplots(
        ncols=2,
        nrows=4,
        figsize=(main_width + bar_width, unibind_track.height + conservation_track.height + INCHES_PER_ALLELE * len(df_plot1) + INCHES_PER_ALLELE * len(df_plot2)),
        gridspec_kw=dict(width_ratios=[main_width, bar_width], wspace=0.02, hspace=0.02, height_ratios=[unibind_track.height, conservation_track.height, INCHES_PER_ALLELE * len(df_plot1), INCHES_PER_ALLELE * len(df_plot2)]),
        )
    ax_empty1.remove()
    ax_empty2.remove()

    idx = pd.IndexSlice
    def _get_pvals(_df_plot):
        if is_abe:
            vals_wt = scores_df.loc[idx[:, tuple()]]
        else:
            vals_wt = scores_df.loc[idx[:, 0, 0]]
        pvals = []
        for allele in _df_plot.itertuples():
            if is_abe:
                vals_mut = scores_df.loc[idx[:, allele.abe_edits]]
            else:
                vals_mut = scores_df.loc[idx[:, allele.del_start, allele.del_len]]
            res = scipy.stats.ttest_ind(vals_wt, vals_mut, equal_var=False)
            log10_pval = (np.log(2) + scipy.stats.norm.logsf(np.abs(res.statistic))) / np.log(10)
            pvals.append((allele.Index, log10_pval))
        return pd.Series(dict(pvals), name='log10_pval')

    # allele table
    df_plot1['log10_pval'] = _get_pvals(df_plot1).apply(' log10_pval = {:.1f}'.format)
    _plot_allele_table(ax_allele, df_plot1, _ref_seq, cutsites=_target_sites, window_start=window.start, is_abe=is_abe, reverse_sequence=reverse_sequence, extra_info_col='log10_pval', highlight_dels=highlight_dels)
    ax_allele.set_xticks([])
    ax_allele.set_xticks([], minor=True)
    ax_allele.spines['bottom'].set_visible(True)
    df_plot2['log10_pval'] = _get_pvals(df_plot2).apply(' log10_pval = {:.1f}'.format)
    _plot_allele_table(ax_allele2, df_plot2, _ref_seq, cutsites=_target_sites, window_start=window.start, is_abe=is_abe, reverse_sequence=reverse_sequence, extra_info_col='log10_pval', highlight_dels=highlight_dels)
    # accessibility data
    _plot_accessibility_boxplot(fig, ax_bar, df_plot1, scores_df, is_abe=is_abe, plot_kind=plot_kind)
    ax_bar.set_xticks([])
    _plot_accessibility_boxplot(fig, ax_bar2, df_plot2, scores_df, is_abe=is_abe, plot_kind=plot_kind)
    xlim1 = ax_bar.get_xlim()
    xlim2 = ax_bar2.get_xlim()
    xlim = (min(xlim1[0], xlim2[0]), max(xlim1[1], xlim2[1]))
    ax_bar.set_xlim(xlim)
    ax_bar2.set_xlim(xlim)

    if not (motif[1] - motif[0] == 1 and is_abe):
        ax_allele.axvspan(motif[0] - window.start - 0.5, motif[1] - window.start - 0.5, color='red', alpha=0.2)
        ax_allele2.axvspan(motif[0] - window.start - 0.5, motif[1] - window.start - 0.5, color='red', alpha=0.2)

    # unibind track
    unibind_track.plot(fig, ax_unibind)

    # conservation track
    conservation_track.plot(fig, ax_conservation)

    fig.savefig(plot_fname, bbox_inches="tight")
    plt.close(fig)

def _plot_allele_table(ax: plt.Axes, df_plot: pd.DataFrame, ref_seq: str, cutsites: pd.Series = None, window_start: int = 0, is_abe: bool = False, reverse_sequence: bool = False, extra_info_col: str = None, highlight_dels: bool = False) -> None:
    """
    df_plot should have columns 'del_start', 'del_len', 'freq', 'count' for the abundance of each deletion allele
    """
    y = np.arange(len(df_plot))
    if not is_abe:
        bar_params = dict(y=y, width=df_plot['del_len'] - 0.2, left=df_plot['del_start'] - 0.4 - window_start)
        if highlight_dels:
            ax.barh(**bar_params, height=0.3, color='yellow')
        else:
            ax.barh(**bar_params, height=0.3, color='black')
    if reverse_sequence:
        ref_seq = complement(ref_seq)
    for i, row in df_plot.iterrows():
        if is_abe:
            for pos, base_ref in enumerate(ref_seq):
                if pos + window_start in row['abe_edits']:
                    color = 'black'
                    if base_ref == "A":
                        base = "G"
                    elif base_ref == "T":
                        base = "C"
                    else:
                        raise ValueError(f"Unexpected base {base_ref}")
                    # draw a box around the edited base
                    ax.add_patch(plt.Rectangle((pos - 0.5, i - 0.5), 1, 1, edgecolor='red', facecolor='red', alpha=0.2))
                else:
                    color = 'gray'
                    base = base_ref
                ax.text(pos, i, base, ha='center', va='center', color=color)
        else:
            del_start, del_len = int(row['del_start'] - window_start), int(row['del_len'])
            if del_start < 0:
                del_len += del_start
                del_start = 0
            allele = ref_seq
            if not highlight_dels:
                allele = allele[:del_start] + "-" * del_len + allele[del_start + del_len:]
            for pos, base in enumerate(allele):
                if base != "-":
                    ax.text(pos, i, base, ha='center', va='center', color='gray')
    ax.set_xlim(-0.5, len(ref_seq) - 0.5)
    if cutsites is not None and len(cutsites) > 0:
        _annotate_cutsites(ax, cutsites)
    sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{row["freq"]:.1%} ({row["count"]:.0f} read{"s" if row["count"] != 1 else ""})' + row.get(extra_info_col, '') for _, row in df_plot.iterrows()])
    ax.tick_params(axis='y', which='both', size=0)
    ax.set_ylim(len(df_plot) - 0.5, -0.5)

def _plot_accessibility_boxplot(fig: plt.Figure, ax: plt.Axes, df_plot: pd.DataFrame, scores_df: pd.Series, is_abe: bool = False, plot_kind: Literal['box', 'bar'] = 'box') -> None:
    """
    df_plot should have columns 'del_start', 'del_len' for the deletion alleles that will be plotted in that order.
    scores_df should have columns 'del_start', 'del_len', 'num_edits' for the number of DddA edits for each deletion allele.
    del_start should be in plot coordinates, i.e. relative to the window.start
    If raw_edits is provided, then the heatmap will be plotted instead of boxplot using scores_df.
    """
    if not is_abe:
        df_raw = scores_df.rename("num_edits").reset_index(['del_start', 'del_len'])
        mapper = {tuple(key) if key['del_len'] != 0 else 0: val for val, key in df_plot[['del_start', 'del_len']].iterrows()}
        df_raw['allele_id'] = df_raw[['del_start', 'del_len']].apply(lambda x: mapper.get((x['del_start'], x['del_len']) if x['del_len'] != 0 else 0, -1), axis="columns")
        df_raw.drop(index=df_raw[df_raw['allele_id'] == -1].index, inplace=True)
    else:
        df_raw = scores_df.rename("num_edits").reset_index('abe_edits')
        mapper = {abe_edits: allele_id for allele_id, abe_edits in df_plot['abe_edits'].items()}
        df_raw['allele_id'] = df_raw['abe_edits'].apply(mapper.get)
    df_raw.dropna(subset='allele_id', inplace=True)
    df_raw['allele_id'] = df_raw['allele_id'].astype(int)
    if plot_kind == 'box':
        sns.boxplot(data=df_raw, orient='h', ax=ax, y="allele_id", x="num_edits", showfliers=False, boxprops=dict(facecolor='lightgray'), showmeans=True)
    elif plot_kind == 'bar':
        sns.barplot(data=df_raw, orient='h', ax=ax, y="allele_id", x="num_edits", color='lightgray')
    else:
        raise ValueError(f"Unknown plot_kind {plot_kind}")
    ax.set_xlabel("Num DddA edits")
    for i in range(len(df_plot)):
        ax.axhline(i, color='black', linewidth=0.3, alpha=0.2, zorder=0)
    ax.set_yticks(np.arange(len(df_plot)))
    ax.set_yticklabels('')
    ax.set_ylabel('')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(len(df_plot) - 0.5, -0.5)

def _plot_accessibility_heatmap(fig: plt.Figure, ax: plt.Axes, df_plot: pd.DataFrame, scores_df: pd.Series, ddda_data: ddda_dataset, is_abe: bool = False, method: Literal["smooth", "dedup+smooth", "dad", "smooth+exclude"] = "smooth", ddda_threshold: float | int = None) -> None:
    """
    df_plot should have columns 'del_start', 'del_len' for the deletion alleles that will be plotted in that order.
    scores_df should have columns 'del_start', 'del_len', 'num_edits' for the number of DddA edits for each deletion allele.
    del_start should be in plot coordinates, i.e. relative to the window.start
    If raw_edits is provided, then the heatmap will be plotted instead of boxplot using scores_df.
    """
    SMOOTHING_WINDOW = 100
    if not is_abe:
        df_raw = scores_df.rename("num_edits").reset_index(['del_start', 'del_len'])
        mapper = {tuple(key) if key['del_len'] != 0 else 0: val for val, key in df_plot[['del_start', 'del_len']].iterrows()}
        df_raw['allele_id'] = df_raw[['del_start', 'del_len']].apply(lambda x: mapper.get((x['del_start'], x['del_len']) if x['del_len'] != 0 else 0, -1), axis="columns")
    else:
        df_raw = scores_df.rename("num_edits").reset_index('abe_edits')
        mapper = {abe_edits: allele_id for allele_id, abe_edits in df_plot['abe_edits'].items()}
        df_raw['allele_id'] = df_raw['abe_edits'].apply(mapper.get)
    df_raw.dropna(subset='allele_id', inplace=True)
    df_raw['allele_id'] = df_raw['allele_id'].astype(int)
    if not is_abe:
        wt_allele_id = (df_plot['del_len'] == 0).idxmax()
    else:
        wt_allele_id = (df_plot['abe_edits'] == tuple()).idxmax()
    wt_reads_index = df_raw[df_raw['allele_id'] == wt_allele_id].index
    locus = list(ddda_data.edit_dict.keys())
    assert len(locus) == 1
    locus = locus[0]
    heatmap_diff = np.full((len(df_plot), ddda_data.edit_dict[locus].shape[1]), np.nan)
    if method == "dedup+smooth":
        for allele_id, reads in df_raw.groupby('allele_id'):
            A_pos_results = diff_DddA_helper(
                ddda_data,
                reads.index,
                wt_reads_index,
                locus,
                down_sample_n=5000,
            )
            if A_pos_results is None:
                continue
            # Calculate effect of sgRNA on DddA edits
            CT_ABE_edited_DddA_edits = A_pos_results["C_to_T"]["fg_DddA_edits"]
            CT_ABE_unedited_DddA_edits = A_pos_results["C_to_T"]["bg_DddA_edits"]
            GA_ABE_edited_DddA_edits = A_pos_results["G_to_A"]["fg_DddA_edits"]
            GA_ABE_unedited_DddA_edits = A_pos_results["G_to_A"]["bg_DddA_edits"]
            diff_edits = (CT_ABE_edited_DddA_edits - CT_ABE_unedited_DddA_edits + GA_ABE_edited_DddA_edits - GA_ABE_unedited_DddA_edits) / 2
            heatmap_diff[allele_id] = np.convolve(diff_edits, np.ones(SMOOTHING_WINDOW), mode="same") / SMOOTHING_WINDOW
        maxval = 0.0075
    elif method in ("smooth", "smooth+exclude", "smooth+groupnorm"):
        ddda_edits = ddda_data.edit_dict[locus]
        if method == "smooth+exclude":
            total_ddda_edits = np.array(ddda_edits.sum(axis=1)).squeeze()
            assert ddda_threshold is not None
            if ddda_threshold < 1:
                threshold_ddda_edits = np.percentile(total_ddda_edits, 100 * ddda_threshold)
            else:
                threshold_ddda_edits = ddda_threshold
            reads_ddda_pos = np.nonzero(total_ddda_edits > threshold_ddda_edits)[0]
        else:
            reads_ddda_pos = np.arange(ddda_edits.shape[0])
        for allele_id, reads in df_raw.groupby('allele_id'):
            heatmap_diff[allele_id] = ddda_edits[np.intersect1d(reads.index, reads_ddda_pos)].mean(axis=0)
        # normalize to wt
        heatmap_diff -= ddda_edits[np.intersect1d(wt_reads_index, reads_ddda_pos)].mean(axis=0)
        heatmap_diff = scipy.signal.convolve2d(heatmap_diff, np.full((1, SMOOTHING_WINDOW), 1 / SMOOTHING_WINDOW), mode="same")
        if method == "smooth+exclude":
            maxval = max(np.percentile(np.abs(heatmap_diff), 99), 0.001)
        else:
            maxval = 0.0075
    elif method == "dad":
        ref_seq = ddda_data.ref_seq_dict[locus]
        edits = ddda_data.edit_dict[locus]
        strands = ddda_data.read_strands[locus]
        for allele_id, reads in df_raw.groupby('allele_id'):
            heatmap_diff[allele_id] = call_dads_hmm_bias(edits[reads.index][:1000].toarray(), ref_seq, strands[reads.index][:1000]).mean(axis=0)
        heatmap_diff -= call_dads_hmm_bias(edits[wt_reads_index][:1000].toarray(), ref_seq, strands[wt_reads_index][:1000]).mean(axis=0)
        maxval = max(np.percentile(np.abs(heatmap_diff), 95), 0.001)
    else:
        raise ValueError(f"Unknown method {method}")
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(-maxval, maxval)
    ax.pcolormesh(np.arange(heatmap_diff.shape[1]), np.arange(heatmap_diff.shape[0]), heatmap_diff, cmap=cmap, norm=norm, rasterized=True)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label="DddA accessibility change", ax=ax, pad=0.3)
    genomic_coords = ddda_data.region_dict[locus]
    ax.set_xlabel(f"Position in read\n{genomic_coords[0]}:{genomic_coords[1]:,}-{genomic_coords[2]:,}")
    ax.set_yticks(np.arange(len(df_plot)))
    ax.set_yticklabels('')
    ax.set_ylabel('')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(len(df_plot) - 0.5, -0.5)
