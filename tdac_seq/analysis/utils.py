import re
import pandas as pd
from Bio.Seq import reverse_complement
import matplotlib.pyplot as plt
import logging
import matplotlib.ticker

def prepare_matplotlib():
    plt.style.use('data/style.mplstyle')
    logging.getLogger('fontTools').setLevel(logging.WARNING)

def parse_guides(input: str) -> pd.Series:
    if input == 'none':
        return pd.Series(name='spacer')
    elif m := re.match(r'^(?P<name>\w*):?(?P<spacer>[ACGT]{20})$', input):
        return pd.Series([m.group('spacer')], index=[m.group('name') or 'sgRNA'], name='spacer')
    elif input.startswith('!'):
        # get library from tsv file
        out = pd.read_csv(input[1:], sep='\t', index_col=0, header=None).squeeze('columns').str.upper()
        out.rename('spacer', inplace=True)
        return out
    else:
        raise ValueError(f"Invalid input: {input}")

def parse_guides_and_props(input: str, ref_seq: str) -> pd.DataFrame:
    """
    Will expand rows if guide targets multiple sites in the ref_seq, unlike find_cutsite and find_guide_strand.
    """
    guides = parse_guides(input)
    df = []
    for strand, cutsite_offset, operator in zip([True, False], [17, 3], [lambda x: x, reverse_complement]):
        for name, spacer in guides.items():
            for m in re.finditer(operator(spacer), ref_seq):
                df.append((
                    spacer,
                    m.start() + cutsite_offset,
                    strand,
                    name,
                ))
    return pd.DataFrame.from_records(df, columns=['spacer', 'cutsite', 'strand', 'name']).set_index('name')

def set_xtick_genomic_coords(ax, xlim: tuple[int, int], genomic_region: tuple[str, int, int]) -> None:
    xticks = matplotlib.ticker.AutoLocator().tick_values(*xlim)[1:-1]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{int(x):d}' for x in xticks], rotation=90)
    genomic_coords_ticks = matplotlib.ticker.AutoLocator().tick_values(genomic_region[1] + xlim[0], genomic_region[1] + xlim[1])[1:-1]
    ax.set_xticks(genomic_coords_ticks - genomic_region[1], minor=True)
    ax.set_xticklabels([f"{int(x):,}" for x in genomic_coords_ticks], minor=True, rotation=90, fontsize=8)
