import os
import sys
import pysam
import PATH
import re
import numpy as np
import json
import pandas as pd
from scipy import stats
from Bio import SeqIO
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from importlib import reload
import parse_ribosome as Pribo

gencode_dir="/data/users/wergillius/reference/gencode/"
gtf_path=os.path.join(gencode_dir,'gencode.v37.annotation.gtf')

# structure the gtf into a  list : [ dict ]
startcodon_json = Pribo.gtf_2_json(gtf_path,'start_codon')
startcodon_json = [dic for dic in startcodon_json if dic['gene_type'] == 'protein_coding']
gtf_df = pd.json_normalize(startcodon_json)
start_codon_df = gtf_df[gtf_df.tag == 'CCDS']

# Exon
exon_json = Pribo.gtf_2_json(gtf_path,condition = 'exon')
# start codon -> exon
exon_w_start  = start_codon_df.exon_id.to_numpy()
# build dict for later KeyError-based filtering
exon_w_start_dict = {exon:1 for exon in exon_w_start}

exon_to_stay = []
for entry in tqdm(exon_json):
    # the level of entry are : Gene Transcrip or exon,
    #  this step will identify which are Exon
    try:
        exon_id = entry['exon_id']
    except KeyError:
        continue
    
    # the main step to identify json to include
    try:
        exon_w_start_dict[exon_id]
        exon_to_stay.append(entry)
    except KeyError:
        continue

print("convert json to df ....")
exon_w_start_df = pd.json_normalize(exon_to_stay)  # hierarchy : exon 

print("de duplicate ....")
exon_w_start_df = exon_w_start_df.drop_duplicates(subset=['exon_id'])

print("merging ....")
merge_df = start_codon_df.merge(exon_w_start_df[['exon_id','start','end']],left_on=['exon_id'],right_on=['exon_id'],how='right')
merge_df = merge_df.drop_duplicates(subset=['exon_id'])
         
print(merge_df.shape)



ribo_priasbl_sam = "/ssd/users/wergillius/public_ribo/STAR_out/cut_adpt/merge_Aligned.sort.bam"

ribo_sam = pysam.AlignmentFile(ribo_priasbl_sam,'rb',require_index=True)


# the chromosome number, start site , end site of all start codons
start_codon_c_s_e = merge_df.iloc[:,[0,2,3]].values       # start codon
exon_w_start_c_s_e = merge_df[['start_y','end_y']].values # exon 

reads_around_start = []
reads_withATG = []

for i in tqdm(range(len(start_codon_c_s_e))):
    
    c_s_e = start_codon_c_s_e[i]
    gene_cse = exon_w_start_c_s_e[i]

    target_reads = ribo_sam.fetch(*c_s_e)
    
    # include reads starts inside the exon , ignore those spliced junction 
    # i.e.`chr1 234 17M234523N13M`  cover  a long range
    reads_in_exons = [
        read for read in target_reads 
                       if read.reference_start in range(*gene_cse)]
    
    reads_around_start.append(reads_in_exons)
    reads_withATG.append([read for read in reads_in_exons if "ATG" in read.seq])
    
    
# Transcripts selection
#<font size=5 color='orange'><center> **Identify transcripts with reads enriched near 0**

start_codon_ls_w_enrich0 = []

for i in tqdm(range(len(start_codon_c_s_e))):
    # iter over start codon

    if len(reads_around_start[i]) > 10:
        anno_start_site =  start_codon_c_s_e[i][1]
        
        # all left site and read len
        left_site = np.array([read.reference_start for read in reads_around_start[i]]) - anno_start_site

        #  filtering
        left_site_selected = np.sum((left_site < -10)&(left_site > -20))

        start_codon_ls_w_enrich0.append((i,left_site_selected))
    
data=np.array(start_codon_ls_w_enrich0)

# x: start codon , y: exon
enrich0_df = merge_df.iloc[data[:,0]][["chromosome","hierarchy","start_x","end_x","start_y","end_y","gene_id","transcript_id","exon_id"]]

enrich0_df.loc[:,'index_of_startcodon'] = data[:,0]
enrich0_df.loc[:,'n_reads_enriched'] = data[:,1]
    
enrich0_df = enrich0_df[enrich0_df.n_reads_enriched != 0]


csv_path = ribo_priasbl_sam.replace("bam","t_enrich0.csv")

enrich0_df.to_csv("/ssd/users/wergillius/public_ribo/STAR_out/cut_adpt/merge_Aligned.sort.t_enrich0.csv", index=False)
print("finished")
print(f"result saved to {csv_path}")
    
    
    
    
    
    
    
