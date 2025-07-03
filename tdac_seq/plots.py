import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.figure
import numpy as np
from collections.abc import Sequence, Iterable
from dataclasses import dataclass
import networkx as nx
from tdac_seq.utils import get_unibind
import pyBigWig
import seaborn as sns
import pandas as pd

class GenomeTrack:
    def __init__(self, track_name: str):
        self.track_name = track_name
        self.height: int = 1

    def plot(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, axvline: int | None = None, xlim: tuple[int, int] | None = None):
        if axvline is not None:
            ax.axvline(axvline, color='red', linewidth=0.5)
        if xlim is not None:
            ax.set_xlim(xlim)

class BigWigTrack(GenomeTrack):
    def __init__(self, track_name: str, data: Iterable[float]):
        super().__init__(track_name)
        self.data = np.array(data)
        self.height = 1

    def plot(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, **kwargs):
        ax.fill_between(np.arange(len(self.data)), self.data, 0, linewidth=0.1)
        ax.set_yticks([self.data.min(), self.data.max()])
        ax.set_ylim(self.data.min(), self.data.max())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        super().plot(fig, ax, **kwargs)

class OverlaidBigWigTrack(GenomeTrack):
    def __init__(self, track_name: str, data: np.ndarray):
        super().__init__(track_name)
        self.data = data
        self.height = 1

    def plot(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, **kwargs):
        for row in self.data:
            ax.plot(row)
        ax.set_yticks([self.data.min(), self.data.max()])
        ax.set_ylim(self.data.min(), self.data.max())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        super().plot(fig, ax, **kwargs)

def plot_hic_map(data: np.ndarray, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, norm = plt.Normalize(-1, 1), cmap=plt.cm.bwr, cbar_kwargs: dict = {}):
    xy = np.stack(np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1])), axis=-1)
    rotation_matrix = np.array([[1, 1], [-1, 1]])
    xy = np.matmul(xy, rotation_matrix)
    data[np.tril_indices_from(data, k=-1)] = np.nan
    h = ax.pcolormesh(xy[..., 1], xy[..., 0], data, norm=norm, cmap=cmap, rasterized=True)
    ax.xaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylim(bottom=0, top=xy[..., 0].max() + 1)
    ax.set_xlim(left=-1, right=xy[..., 1].max() + 1)
    ax.set_yticks([])
    bbox = list(ax.get_position().bounds)
    bbox[0] -= 0.02
    bbox[2] = 0.02
    ax_cbar = fig.add_axes(bbox)
    fig.colorbar(h, cax=ax_cbar, orientation="vertical", location='left', **cbar_kwargs)

class HiCTrack(GenomeTrack):
    def __init__(self, track_name: str, data: Iterable[Iterable[float]], **kwargs):
        super().__init__(track_name)
        self.data = np.array(data)
        self.height = 2 # half of the width of the ax
        self.kwargs = kwargs

    def plot(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, xlim: tuple[int, int] = None, **kwargs):
        if xlim:
            plot_hic_map(self.data[xlim[0]:xlim[1], xlim[0]:xlim[1]], fig, ax, **self.kwargs)
        else:
            plot_hic_map(self.data, fig, ax, **self.kwargs)
        super().plot(fig, ax)

@dataclass
class BedEntry:
    start: int
    end: int
    name: str
    score: float

class BedTrack(GenomeTrack):
    def __init__(self, track_name: str, data: Iterable[BedEntry]):
        super().__init__(track_name)
        self.data = list(data)
        self.height = 1
        self.cm = plt.cm.Blues

    def plot(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, **kwargs):
        norm = plt.Normalize(vmin=min(entry.score for entry in self.data), vmax=max(entry.score for entry in self.data))
        np.random.seed(2020)
        for entry in self.data:
            y = np.random.randint(10)
            ax.barh(y=y, width=entry.end - entry.start, left=entry.start, height=1, color=self.cm(norm(entry.score)))
            ax.annotate(entry.name, ((entry.start + entry.end) / 2, y), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=5, alpha=max(0.2, norm(entry.score)))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.xaxis.set_visible(False)
        super().plot(fig, ax, **kwargs)

INCHES_PER_SQUARE = 0.2
INCHES_PER_ALLELE = 0.3

class UnibindTrack(GenomeTrack):
    def __init__(self, track_name: str, genomic_coords: tuple[str, int, int]):
        super().__init__(track_name)
        self.genomic_coords = genomic_coords
        self.unibind_tfs = get_unibind(*self.genomic_coords)
        self._y = self._assign_tf_to_y()
        self.height = INCHES_PER_ALLELE * (self._y.max() + 1) if len(self._y) > 0 else 0
        self.width = INCHES_PER_SQUARE * (self.genomic_coords[2] - self.genomic_coords[1])

    def _assign_tf_to_y(self) -> np.ndarray:
        MIN_BUFFER = 6 / INCHES_PER_SQUARE
        y = np.arange(len(self.unibind_tfs))
        for i, row in self.unibind_tfs.reset_index().iterrows():
            for j in range(i):
                # check if the any tf overlaps on track j with the current tf
                if (j not in y[:i]) or row['chromStart'] >= self.unibind_tfs.iloc[np.nonzero(y[:i] == j)[0]]['chromEnd'].max() + MIN_BUFFER:
                    y[i] = j
                    break
        return y

    def plot(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, **kwargs):
        unibind_track = dict(
            y=self._y,
            width=self.unibind_tfs['chromEnd'] - self.unibind_tfs['chromStart'] - 0.2,
            left=self.unibind_tfs['chromStart'] - self.genomic_coords[1] - 0.4,
            height=0.5,
        )
        ax.barh(**unibind_track,
            color=self.unibind_tfs['color'],
            )
        ax.barh(**unibind_track,
            hatch=self.unibind_tfs['strand'].map({'+': '/', '-': '\\', '.': 'x'}),
            fill=False,
            edgecolor=self.unibind_tfs['color'].apply(_adjust_color_lightness),
            )
        for (_, row), y in zip(self.unibind_tfs.reset_index().iterrows(), self._y):
            ax.annotate(
                text=row['tf'],
                xy=(row['chromStart'] - self.genomic_coords[1] - 0.4, y),
                xytext=(-3, 0),
                textcoords="offset points", ha='right', va='center', color='black')
        ax.axis('off')
        ax.set_xlim(-0.5, self.genomic_coords[2] - self.genomic_coords[1] - 0.5)
        ax.set_ylim(-0.5, (self._y.max() if len(self._y) > 0 else 0) + 0.5)
        super().plot(fig, ax, **kwargs)

class ConservationTrack(GenomeTrack):
    def __init__(self, track_name: str, genomic_coords: tuple[str, int, int]):
        super().__init__(track_name)
        self.genomic_coords = genomic_coords
        self.height = INCHES_PER_ALLELE
        self.width = INCHES_PER_SQUARE * (self.genomic_coords[2] - self.genomic_coords[1])

    def plot(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, **kwargs):
        phyloP_range = (-0.5, 5)
        phyloP = np.array(pyBigWig.open("http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw").values(*self.genomic_coords))
        phyloP_plot = np.clip(phyloP, *phyloP_range)
        cmap = plt.cm.coolwarm
        norm = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=phyloP_range[0], vmax=phyloP_range[1])
        ax.bar(np.arange(len(phyloP_plot)), phyloP_plot, color=[cmap(norm(x)) for x in phyloP_plot], width=1)
        ax.set_ylim(*phyloP_range)
        ax.set_xlim(-0.5, len(phyloP_plot) - 0.5)
        ax.set_ylabel("PhyloP", rotation=0, ha='right')
        sns.despine(ax=ax, left=False, bottom=True, right=True, top=True)
        ax.set_yticks(phyloP_range)
        ax.set_xticks([])
        super().plot(fig, ax, **kwargs)

class ArchesTrack(GenomeTrack):
    def __init__(self, track_name: str, data: Iterable[BedEntry], above: bool = True, vmin: float = 0, vmax: float = 1, ylim: tuple[float, float] | None = None):
        super().__init__(track_name)
        self.data = list(data)
        self.height = 1
        self.above = above
        self.vmin = vmin if vmin is not None else min(entry.score for entry in self.data)
        self.vmax = vmax if vmax is not None else max(entry.score for entry in self.data)
        self.ylim = ylim

    def plot(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, **kwargs):
        vrange_diff = self.vmax - self.vmin
        norm = plt.Normalize(self.vmin - 0.1 * vrange_diff, self.vmax + 0.1 * vrange_diff)
        for entry in self.data:
            # plot arch
            significance = norm(entry.score)
            distance = entry.end - entry.start
            x = np.linspace(entry.start, entry.end, 10)
            y = distance * np.sin(np.linspace(0, np.pi, 10))
            ax.plot(x, y, color=plt.cm.Blues(significance), linewidth=0.5, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        if self.ylim:
            ax.set_yticks(self.ylim)
            ax.set_ylim(*self.ylim)
        if not self.above:
            ax.invert_yaxis()
        super().plot(fig, ax, **kwargs)

def plot_genome_tracks(tracks: Sequence[GenomeTrack], width: float = 4, axvline: int | None = None, xlim: tuple[int, int] | None = None) -> matplotlib.figure.Figure:
    VSPACE = 0.2
    MAXHEIGHT = sum(track.height + VSPACE for track in tracks) - VSPACE
    fig = plt.figure(dpi=600, figsize=(width, MAXHEIGHT))
    current_figy = MAXHEIGHT
    for track in tracks:
        current_figy -= track.height
        ax = fig.add_axes([0, current_figy / MAXHEIGHT, 1, track.height / MAXHEIGHT])
        current_figy -= VSPACE
        track.plot(fig, ax, axvline=axvline, xlim=xlim)
        fig.text(-0.15, (ax.get_position().y0 + ax.get_position().y1) / 2, track.track_name, ha='right', va='center')
    return fig

class HiCTrackSquare(GenomeTrack):
    def __init__(self, track_name: str, data: Iterable[Iterable[float]], norm=plt.Normalize(-1, 1), cmap=plt.cm.RdGy_r):
        super().__init__(track_name)
        self.data = np.array(data)
        self.height = 2
        self.norm = norm
        self.cmap = cmap

    def plot(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, xlim: tuple[int, int] = None, **kwargs):
        h = ax.pcolormesh(self.data, norm=self.norm, cmap=self.cmap, rasterized=True)
        ax.xaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_ylim(bottom=self.data.shape[0], top=0)
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(left=0, right=self.data.shape[1])
        ax.set_yticks([])
        super().plot(fig, ax)

def plot_TF_track(TF_sites, chr, start, end, ax):

    import pyranges as pr

    # Keep TF sites in the specified region
    region_range = pr.PyRanges(
        chromosomes=chr, 
        starts=[start], 
        ends=[end])
    region_TF_sites = TF_sites.intersect(region_range).df
    region_TF_sites = region_TF_sites.iloc[np.argsort(region_TF_sites["Start"]),:]

    # Merge overlapping motifs of the same TF
    region_TF_site_dict = {}
    for i, row in region_TF_sites.iterrows():
        Chromosome, Start, End, TF = row[["Chromosome", "Start", "End", "TF"]]
        if TF not in region_TF_site_dict:
            region_TF_site_dict[TF] = [[Start, End]]
        else:
            # If the current motif overlaps with the previous motif of the same TF, merge them
            if Start <= region_TF_site_dict[TF][-1][1]:
                region_TF_site_dict[TF][-1][1] = max(region_TF_site_dict[TF][-1][1], End)
            else:
                region_TF_site_dict[TF].append([Start, End])

    # Visualize the motif match track
    from dna_features_viewer import GraphicFeature, GraphicRecord
    features = []
    for TF in region_TF_site_dict:
        for site in region_TF_site_dict[TF]:
            features.append(GraphicFeature(
                start=site[0] - region_start - plot_start, 
                end=site[1] - region_start - plot_start, 
                color=generate_low_saturation_color(), 
                label=TF))
    record = GraphicRecord(sequence_length=plot_end - plot_start, features=features)
    record.plot(ax=ax)

def plot_hmm(transmat, emissionprob_open, emissionprob_closed, ax):
    # transmat = np.array([
    #     # closed   , open
    #     [0.99683402, 0.00316598], # closed
    #     [0.09315898, 0.90684102], # open
    #     ])
    # emissionprob_open = np.array([
    #     # unedited , edited
    #     [0.52864081, 0.47135919], # TC
    #     [0.72864081, 0.27135919], # GC
    #     [0.82864081, 0.17135919], # CC
    #     [0.92864081, 0.07135919], # AC
    #     ])
    # emissionprob_closed = np.array([1, 0])

    G = nx.DiGraph()
    G.add_node("Closed", pos=(1, 2), color="orangered")
    G.add_node("Open", pos=(0, 2), color="orangered")
    G.add_node("TC", pos=(-0.45, 1.2))
    G.add_node("CC", pos=(-0.15, 1.2))
    G.add_node("GC", pos=( 0.15, 1.2))
    G.add_node("AC", pos=( 0.45, 1.2))
    G.add_node("Closed, Edited", pos=(1, 0), label="Edited", color="deepskyblue")
    G.add_node("TC, Edited", pos=(-0.45, 0), label="Edited", color="deepskyblue")
    G.add_node("CC, Edited", pos=(-0.15, 0), label="Edited", color="deepskyblue")
    G.add_node("GC, Edited", pos=( 0.15, 0), label="Edited", color="deepskyblue")
    G.add_node("AC, Edited", pos=( 0.45, 0), label="Edited", color="deepskyblue")
    G.add_edge("Open", "TC", label="", color="skyblue")
    G.add_edge("Open", "CC", label="", color="skyblue")
    G.add_edge("Open", "GC", label="", color="skyblue")
    G.add_edge("Open", "AC", label="", color="skyblue")
    G.add_edge("Closed", "Open", label=f"{transmat[0, 1]:.2%}", connectionstyle="arc3,rad=0.1", color="lightcoral")
    G.add_edge("Open", "Closed", label=f"{transmat[1, 0]:.2%}", connectionstyle="arc3,rad=0.1", color="lightcoral")
    G.add_edge("Closed", "Closed", label=f"{transmat[0, 0]:.2%}", color="lightcoral")
    G.add_edge("Open", "Open", label=f"{transmat[1, 1]:.2%}", color="lightcoral")
    G.add_edge("Closed", "Closed, Edited", label=f"{emissionprob_closed[1]:.2%}", color="skyblue")
    G.add_edge("TC", "TC, Edited", label=f"{emissionprob_open[0, 1]:.2%}", color="skyblue")
    G.add_edge("CC", "CC, Edited", label=f"{emissionprob_open[2, 1]:.2%}", color="skyblue")
    G.add_edge("GC", "GC, Edited", label=f"{emissionprob_open[1, 1]:.2%}", color="skyblue")
    G.add_edge("AC", "AC, Edited", label=f"{emissionprob_open[3, 1]:.2%}", color="skyblue")

    pos = nx.get_node_attributes(G, "pos")
    node_labels = {lbl: lbl for lbl in G.nodes}
    node_labels |= nx.get_node_attributes(G, "label")
    node_colors = nx.get_node_attributes(G, "color")
    node_colors = [node_colors.get(node, (0, 0, 0, 0)) for node in G.nodes]
    node_fontcolors = nx.get_node_attributes(G, "font_color")
    edge_colors = nx.get_edge_attributes(G, "color")

    node_size = 1400
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_size)
    for node in G.nodes:
        nx.draw_networkx_labels(G, pos, labels={node: node_labels[node]}, font_color=node_fontcolors.get(node), ax=ax)
    for edge in G.edges(data=True):
        connectionstyle = edge[2].get("connectionstyle", "arc3")
        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=edge_colors.get((edge[0], edge[1]), "grey"), connectionstyle=connectionstyle, node_size=node_size, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(edge[0], edge[1]): edge[2].get("label", "")}, connectionstyle=connectionstyle, node_size=node_size, label_pos=0.3, bbox={"alpha": 0}, ax=ax)
    ax.set_axis_off()

def plot_atac_peaks(atac_vals: np.ndarray, peaks: list[tuple[int, int]], genomic_coords: tuple[str, int, int] = None) -> plt.figure:
    figheight_atac = 2
    figheight_perbar = 0.2
    figwidth_perbp = 0.0015
    fig, (ax_peaks, ax_atac) = plt.subplots(nrows=2, figsize=(len(atac_vals) * figwidth_perbp, figheight_atac + figheight_perbar * len(peaks)), gridspec_kw=dict(height_ratios=[figheight_perbar * len(peaks), figheight_atac], hspace=0))
    ax_atac.fill_between(np.arange(len(atac_vals)), atac_vals, linewidth=0, color=plt.cm.tab10(0))
    if genomic_coords is not None:
        ax_atac.set_xlabel(f"Position in {genomic_coords[0]}:{genomic_coords[1]:,}-{genomic_coords[2]:,}")
    else:
        ax_atac.set_xlabel("Position in the locus")
    for peak_i, (peak_start, peak_end) in enumerate(peaks):
        ax_peaks.barh(peak_i, peak_end - peak_start, left=peak_start, color='gray')
        ax_peaks.annotate(f"{peak_start:,}-{peak_end:,}", (peak_start, peak_i), xytext=(-2, 0), textcoords='offset points', ha='right', va='center')
    ax_peaks.invert_yaxis()
    ax_peaks.axis('off')
    ax_peaks.set_xlim(0, len(atac_vals))
    ax_atac.set_xlim(0, len(atac_vals))
    ax_atac.set_ylim(bottom=0)
    ax_atac.spines['right'].set_visible(False)
    ax_atac.spines['top'].set_visible(False)
    ax_atac.spines['left'].set_visible(False)
    ax_atac.set_yticks([])
    return fig

def _adjust_color_lightness(c: tuple[float, float, float]):
    import colorsys
    c = colorsys.rgb_to_hls(*c)
    if c[1] < 0.5:
        amount = 0.6
    else:
        amount = 1.4
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def _annotate_cutsites(ax: plt.Axes, cutsites: pd.Series):
    ax.set_xticks(cutsites - 0.5, minor=True, labels=cutsites.index.to_series().apply(lambda x: x if x.startswith("sg") else "sg" + x), rotation=45, ha='right')
    ax.tick_params(axis='x', which='minor', direction='out', top=False, bottom=True, labelbottom=True, labeltop=False, length=20)

def plot_crispr_coverage(ax: plt.Axes, crispr_coverage: np.ndarray, threshold: float = None, cutsites: pd.Series = None):
    ax.plot(np.arange(len(crispr_coverage)), crispr_coverage)
    ax.set_xlabel("Position in the locus")
    if threshold is not None:
        ax.axhline(y=threshold, color="red", linestyle="dashed")
    if cutsites is not None:
        _annotate_cutsites(ax, cutsites + 0.5)

def plot_impact_score(genomic_region: tuple[str, int, int], is_abe: bool, crispr_coverage: Sequence[float], cutsites: Sequence[int], impact_score: Sequence[float], subwindow: tuple[int, int] = None) -> matplotlib.figure.Figure:
    if subwindow is None:
        subwindow = (0, genomic_region[2] - genomic_region[1])
    genomic_coords = (genomic_region[0], genomic_region[1] + subwindow[0], genomic_region[1] + subwindow[1])
    xlim_min, xlim_max = subwindow
    unibind_track = UnibindTrack("", genomic_coords)
    conservation_track = ConservationTrack("", genomic_coords)
    fig, (ax_unibind, ax_conservation, ax_coverage, ax) = plt.subplots(figsize=(12, 6 + unibind_track.height + conservation_track.height), nrows=4, gridspec_kw=dict(height_ratios=[unibind_track.height, conservation_track.height, 3, 3], hspace=0.02))
    unibind_track.plot(fig, ax_unibind)
    conservation_track.plot(fig, ax_conservation)
    plot_crispr_coverage(ax_coverage, crispr_coverage)
    ax_coverage.set_xlim(xlim_min, xlim_max)
    ax_coverage.set_xticks([])
    ax.stem(cutsites, impact_score)
    # ax.set_xticks(cut_sites_plot, minor=True)
    # ax.set_xticklabels([f"sg{x}" if not x.startswith('sg') and not is_abe else x for x in cut_sites_plot.index], rotation=45, ha="right", minor=True)
    if is_abe:
        ax_coverage.set_ylabel("ABE coverage / background")
        ax.set_ylabel("ABE impact score")
    else:
        ax_coverage.set_ylabel("Deletion coverage")
        ax.set_ylabel("Impact score of deletion")
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_xlabel(f"Position in the locus\namplicon = {genomic_region[0]}:{genomic_region[1]:,}-{genomic_region[2]:,}\nvisible = {genomic_coords[0]}:{genomic_coords[1]:,}-{genomic_coords[2]:,}")
    ax.set_ylim(bottom=0)
    return fig

def plot_schematic(motifs: dict[str, tuple[int, int]], ref_seq: str, genomic_coords: tuple[str, int, int], window_in_amplicon: tuple[int, int]):
    genomic_coords_window = (genomic_coords[0], genomic_coords[1] + window_in_amplicon[0], genomic_coords[1] + window_in_amplicon[1])
    unibind_track = UnibindTrack("", genomic_coords_window)
    conservation_track = ConservationTrack("", genomic_coords_window)
    fig, (ax_unibind, ax_conservation, ax) = plt.subplots(figsize=(12, unibind_track.height + conservation_track.height + 0.3), nrows=3, gridspec_kw=dict(height_ratios=[unibind_track.height, conservation_track.height, 0.3], hspace=0.02), sharex=True)
    unibind_track.plot(fig, ax_unibind)
    conservation_track.plot(fig, ax_conservation)
    for i, c in enumerate(ref_seq[genomic_coords_window[1] - genomic_coords[1]:genomic_coords_window[2] - genomic_coords[1] + 1]):
        ax.text(i, 0, c, ha='center', va='center', fontsize=8)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_visible(False)
    ax_conservation.xaxis.set_visible(False)
    for motif_name, motif in motifs.items():
        ax.axvspan(motif[0] - (genomic_coords_window[1] - genomic_coords[1]) - 0.5, motif[1] - (genomic_coords_window[1] - genomic_coords[1]) - 0.5, color='red', alpha=0.2)
    xticks = np.arange(0, len(ref_seq), 10)
    xticks = xticks[xticks >= (genomic_coords_window[1] - genomic_coords[1])]
    xticks = xticks[xticks <= (genomic_coords_window[2] - genomic_coords[1])]
    ax.set_xticks(xticks - genomic_coords_window[1] + genomic_coords[1])
    ax.set_xticklabels(xticks)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(f'Position in amplicon\nAmplicon = {genomic_coords[0]}:{genomic_coords[1]}-{genomic_coords[2]}\nVisible = {genomic_coords_window[0]}:{genomic_coords_window[1]}-{genomic_coords_window[2]}')
    return fig
