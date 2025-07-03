from tdac_seq.utils import get_atac_peaks

def peaks_for_locus(locus: str, genomic_coords: tuple[str, int, int]) -> list[tuple[int, int]]:
    peaks = [(0, genomic_coords[2] - genomic_coords[1])]
    peaks.extend(get_atac_peaks(*genomic_coords))

    return peaks

def windows_for_locus(locus: str, editable_pos: set[int]) -> tuple[list[list[slice]], int, list[tuple[int | None]]]:
    if locus == "LCR_new":
        windows = [[slice(1390, 1430)], [slice(1470, 1510)]]
        min_coverage = 1000
        motifs = [(1410, 1411), (1492, 1493)]
    else:
        raise NotImplementedError()
    
    return windows, min_coverage, motifs

def get_motifs(locus: str):
    motif_search = {
        "GFI1B": dict(
            searchrange_start=(1300, 2000),
            motif={
                'GATA':   (1560, 1569),
                'AP-1':         (1608, 1623),
                'SPI1':         (1898, 1909),
                'Unannotated':  (1957, 1961),
            },
            motif_highlight=['GATA', 'AP-1', 'SPI1', 'Unannotated']
        ),
    }
    return motif_search.get(locus, {})
