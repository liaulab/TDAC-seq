import pandas as pd
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import numpy as np
import matplotlib.pyplot as plt
plt.set_loglevel("WARNING")
from tqdm import trange
import matplotlib.colors
from tdac_seq.analysis.utils import prepare_matplotlib, set_xtick_genomic_coords
prepare_matplotlib()
import scipy.stats
import plotly.io
plotly.io.templates.default = "plotly_white"
import plotly.graph_objects as go
import matplotlib.patheffects

def minimal_motif_search(
        scores: pd.DataFrame,
        motif_search_fname: str,
        searchrange_start: tuple[int, int],
        searchrange_size: tuple[int, int] = (1, 15),
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    scores is a DataFrame with columns ['del_start', 'del_len', 'num_edits'] where each row is a read.
    Returns the heatmap of p-values for the minimal motif search and minimum number of reads in the +/- deletion clusters per window.
    """
    if os.path.exists(motif_search_fname):
        logging.info(f"Loading motif search results from {motif_search_fname}")
        motif_search = np.load(motif_search_fname)
        heatmap = motif_search['heatmap']
        heatmap_coverage = motif_search['heatmap_coverage']
    else:
        logging.info(f"Searching for minimal motif")
        heatmap = np.full((searchrange_size[1], searchrange_start[1] - searchrange_start[0]), np.nan)
        heatmap_coverage = np.zeros((searchrange_size[1], searchrange_start[1] - searchrange_start[0]), dtype=int)
        for motif_size in trange(*searchrange_size, desc="Motif size"):
            for motif_start in trange(*searchrange_start, leave=False, desc="Motif start"):
                _motif = slice(motif_start, motif_start + motif_size)
                motif_deleted = (scores['del_start'] <= _motif.start) & (scores['del_start'] + scores['del_len'] > _motif.stop)
                group = motif_deleted & (scores['del_len'] <= motif_size + 10)
                res = scipy.stats.ttest_ind(scores.loc[group, 'num_edits'], scores.loc[~group, 'num_edits'], equal_var=False)
                # normal approximation to t-distribution because df is large
                pval = np.log(2) + scipy.stats.norm.logsf(np.abs(res.statistic))
                heatmap[motif_size, motif_start - searchrange_start[0]] = pval
                heatmap_coverage[motif_size, motif_start - searchrange_start[0]] = min(sum(group), sum(~group))
        np.savez_compressed(motif_search_fname, heatmap=heatmap, heatmap_coverage=heatmap_coverage)
    return heatmap, heatmap_coverage

def plot_motif_search(
        plot_dir: str,
        heatmap: np.ndarray,
        genomic_coords: tuple[str, int, int],
        searchrange_start: tuple[int, int],
        searchrange_size: tuple[int, int] = (1, 15),
        heatmap_coverage: np.ndarray = None,
        min_coverage: int = 1000,
        motif_annotations: list[tuple[int, int]] | dict[str, tuple[int, int]] = None,
        pval_thresh: float = -100,
    ) -> None:
    heatmap_plot = heatmap.copy() / np.log(10)
    if heatmap_coverage is not None:
        heatmap_plot[heatmap_coverage < min_coverage] = np.nan

    xlabel = f"Start position of deletion window\nAmplicon={genomic_coords[0]}:{genomic_coords[1]:,}-{genomic_coords[2]:,}\nVisible={genomic_coords[0]}:{genomic_coords[1] + searchrange_start[0]:,}-{genomic_coords[1] + searchrange_start[1]:,}"
    ylabel = "Length of deletion window"

    fig, ax = plt.subplots(figsize=(0.05 * heatmap_plot.shape[1], 0.2 * heatmap_plot.shape[0]))
    cmap = plt.cm.terrain_r
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=pval_thresh)
    ax.pcolormesh(*np.meshgrid(
        np.arange(heatmap_plot.shape[1]) + searchrange_start[0],
        np.arange(heatmap_plot.shape[0]) + searchrange_size[0],
        ), heatmap_plot, cmap=cmap, norm=norm)
    set_xtick_genomic_coords(ax, xlim=searchrange_start, genomic_region=genomic_coords)
    ax.set_xlabel(xlabel)
    if motif_annotations is None:
        _motif_annotations = {}
    elif isinstance(motif_annotations, list):
        _motif_annotations = {f"Δ{_motif[0]}-{_motif[1]}": _motif for _motif in motif_annotations}
    else:
        _motif_annotations = {f"{motif_name} (Δ{_motif[0]}-{_motif[1]})": _motif for motif_name, _motif in motif_annotations.items()}
    for _motif_name, _motif in _motif_annotations.items():
        kwargs = dict(xy=(_motif[0] - 1, _motif[1] - _motif[0]), xytext=(20, 20), textcoords="offset points")
        ax.annotate("", **kwargs, arrowprops=dict(arrowstyle='->', ))
        ax.annotate(_motif_name, **kwargs,
                    ha='left', va='bottom', color='black',
                    # path_effects=[matplotlib.patheffects.withStroke(linewidth=1, foreground='white')],
                    )
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label="log10 P-value", ax=ax)
    # get two slope ticks
    ticks0 = matplotlib.ticker.AutoLocator().tick_values(norm.vmin, norm.vcenter)
    ticks1 = matplotlib.ticker.AutoLocator().tick_values(norm.vcenter, norm.vmax)
    ticks = np.concatenate([
        ticks0[-3:1:-2], # skip the first and last, center is going to be the last, so skip that one, also skip every other
        ticks1[2:-1:2], # skip the first and last, center is going to be the first, so skip that one, also skip every other
        [norm.vcenter],
        ])
    cbar.set_ticks(ticks)
    fig.savefig(os.path.join(plot_dir, "minimalmotif_agg.pdf"), bbox_inches="tight")
    plt.close(fig)

    data = pd.DataFrame(heatmap_plot, index=pd.RangeIndex(searchrange_size[0], searchrange_size[1] + 1, name='Length of deletion window'), columns=pd.RangeIndex(searchrange_start[0], searchrange_start[1], name='Start position of deletion window'))
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=data.columns,
        y=data.index,
        colorscale='earth',
        zmax=0,
        zmid=pval_thresh,
        zmin=-2000,
        colorbar=dict(title="log10 P-value"),
        hovertemplate="Start position of deletion window: %{x}<br>Length of deletion window: %{y}<br>P-value: %{z:.2e}<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        )
    fig.write_html(os.path.join(plot_dir, "minimalmotif_agg.html"))
