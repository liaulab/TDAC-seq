from Bio import SeqIO
import pandas as pd
from collections.abc import Container

def extract_ref_seq(genome_file: str, region_dict: dict[str, tuple[str, int, int]], out_fasta_path: str):
    '''
    Extract reference sequences for the regions of interest
    :param genome_file: str, path to the reference genome file (e.g. hg38.fa)
    :param region_dict: dict, dictionary of regions to analyze. Format: {region_name: (chromosome, start, end)}
        Coordinates are 1-based and inclusive.
    :param out_fasta_path: str, path to the output fasta file
    '''

    assert all(" " not in region for region in region_dict.keys()), "Region names cannot contain spaces"

    # Extract the reference sequences for the regions of interest
    print("Loading reference genome")
    genome_seq = {}
    for record in SeqIO.parse(genome_file, "fasta"):
        genome_seq[record.id] = record.seq
    print("Extracting sequences for each locus")
    ref_seq_dict = {}
    with open(out_fasta_path, "w") as ref_fasta:
        for locus in region_dict.keys():
            chrom, region_start, region_end = region_dict[locus]
            ref_seq = str(genome_seq[chrom][(region_start - 1):(region_end)]).upper()
            ref_seq_dict[locus] = ref_seq
            ref_fasta.write(">" + locus + "\n")
            ref_fasta.write(ref_seq + "\n")
    return ref_seq_dict

def regions_df_to_dict(regions_df: pd.DataFrame) -> dict:
    return {key: val for key, val in zip(regions_df.index, zip(regions_df['chr'], regions_df['start'], regions_df['end']))}

def extract_ref_seq_from_fasta(ref_fasta_path: str, loci: Container[str] | str):
    if isinstance(loci, str):
        loci = {loci}
    ref_seq_dict = {}
    with open(ref_fasta_path, "rt") as ref_fasta:
        while (line := ref_fasta.readline()):
            assert line[0] == ">", "Invalid fasta file"
            locus = line[1:].strip()
            ref_seq = ref_fasta.readline().strip()
            if locus in loci:
                ref_seq_dict[locus] = ref_seq
    assert all(locus in ref_seq_dict for locus in loci), "Some regions are not in the reference fasta file"
    return ref_seq_dict
