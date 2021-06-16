import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import parse_ribomap as Pribo

"""
This is script will generate a reference fasta file from gbff file
gbff file :  GRCh38_p13_rna.gbff contain all human isoform
For each gb entry in gbff , we can extract is CDS range if it is a protein coding isoform.
We later truncate the isoform
"""

def main():
    gbff_path="/data/users/wergillius/reference/GRCh38_p13_rna/GRCh38.p13_rna.gbff"
    txt_path = "/data/users/wergillius/reference/GRCh38_p13_rna/GRCh38_p13_orf20UTR_cds_range.txt"
    gbff = SeqIO.parse(gbff_path,'gb')   # intotal : 163975 entry

    print("GBFF file loaded, extracting Transcript info ..... \n")
    print("This will take around 2 min")

    trunced_seqrecord = []
    
    with open(txt_path,'w') as f:

        for i in tqdm(range(163975)):
            # iterate over gbff 
            SeqRecord = next(gbff)
            try:
                # extract info
                gb_dict = Pribo.read_gb(SeqRecord)  # the Pribo func to process cds annotation
            except AssertionError:
                # skip entry with Assertion : NO CDS
                continue 

            seq = SeqRecord.seq
            start = gb_dict['start']
            end = gb_dict['end']

            if start >=20:
                # in case there are very short utr
                start -= 20
            else:
                continue # start from where it can
            if len(seq) >= end+20:
                end += 20
            else:
                continue  # end from where it can

            SeqRecord.seq=seq[start:end]
            # add to description to clarify the real end
            SeqRecord.description=SeqRecord.description+"[start=%s]"%start+"[end=%s]"%end
        
            trunced_seqrecord.append(SeqRecord)
            
            f.write("%s %s %s\n"%(SeqRecord.id,start,end))
            
    out="/data/users/wergillius/reference/GRCh38_p13_rna/GRCh38_p13_orf20UTR.fasta"
    SeqIO.write(trunced_seqrecord,out,'fasta')
    
    print("new ref saved to %s"%out)
        
if __name__  == "__main__":
    main()
    