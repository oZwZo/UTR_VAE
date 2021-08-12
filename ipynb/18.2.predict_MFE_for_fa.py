import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from Bio import SeqIO
import json
import RNA

start,end = sys.argv[1:]
print(start,end)
start = int(start)
end = int(end)

pcA_cds = list(SeqIO.parse('/data/users/wergillius/UTR_VAE/fa_dataset/human_yeast_sampled_seq.fasta',
            format='fasta'))
sub_df = pcA_cds[start:end]  # subset data to predict

mfe_dict = {}
ss_dict = {}

for SeqRecord in tqdm(sub_df, 'folding sequences from %d ...'%start):
    sequence = SeqRecord.seq.__str__()
    ss_string, mfe = RNA.fold(sequence)
    
    mfe_dict[SeqRecord.id] = mfe
    ss_dict[SeqRecord.id] = ss_string

with open(f'/data/users/wergillius/UTR_VAE/fa_dataset/mfe_json/subset_{start}_{end}.json','w') as out:
    json.dump(mfe_dict,out)
    out.close()
    
with open(f'/data/users/wergillius/UTR_VAE/fa_dataset/ss_json/subset_{start}_{end}.json','w') as out:
    json.dump(ss_dict,out)
    out.close()
