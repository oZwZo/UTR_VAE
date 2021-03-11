# This file is an executable script that is to manage the outputs from 
#    ribomap
#    salmon
#.   NCBI gbff annotation
#   
#.       

import os
import sys
import PATH
from Bio import SeqIO
import numpy as np
import pandas as pd
from tqdm import tqdm
import parse_ribomap as Pribo


def ID_2_renamed_DF(ID):
    # read DF , take columns and rename
    statspath =  os.path.join(out_dir,ID,ID+'_trimmed.stats')
    DF = Pribo.read_ribomap_stats(statspath)[['Transcipt_ID','Ribo_count']]
    DF.rename(columns={"Transcipt_ID":"Transcipt_ID",'Ribo_count':'RC_'+ID},inplace=True)  
    return DF

def ID_2_renamed_quant_df(ID):
    # read DF , take columns and rename
    path =  os.path.join(salmonout_path,ID,'quant.sf')
    DF = pd.read_table(path,sep='\t')[['Name','TPM', 'NumReads']]
    DF.rename(columns={"Name":"Name",'TPM':'TPM_'+ID,'NumReads':'TC_'+ID},inplace=True)  
    return DF

# root dir
GSE65778_dir = "/data/users/wergillius/UTR_VAE/public_ribo/GSE65778"

# salmon output
salmonout_path=os.path.join(GSE65778_dir,"salmon_out")
Sal_id = sorted(os.listdir(salmonout_path))


# ribo map output
out_dir = os.path.join(GSE65778_dir,'ribomap_out')
SRR_id = sorted(os.listdir(out_dir))



#####  merge all the stats result  #####
# initiate a DF 
print("\nloading ribomap samples :")
big_DF = ID_2_renamed_DF(SRR_id[0])
print("\t %s"%SRR_id[0])

# read all the stats file
for ID in SRR_id[1:]:
    print("\t %s"%ID)
    DF = ID_2_renamed_DF(ID) # just defined aboved
    big_DF = big_DF.merge(DF,left_on=['Transcipt_ID'],right_on=['Transcipt_ID'])
    
    
#####  merge all the quant.sf result  #####
# initiate a DF 
print("\n\nloading salmon sample :")
big_quant = pd.read_table(os.path.join(salmonout_path,Sal_id[0],'quant.sf'),sep='\t')
big_quant.columns = ['Name', 'Length', 'EffectiveLength', 'TPM_'+Sal_id[0], 'TC_'+Sal_id[0]]
print("\t %s"%Sal_id[0])

# read all the stats file
for ID in Sal_id[1:]:
    DF = ID_2_renamed_quant_df(ID) # just defined aboved
    big_quant = big_quant.merge(DF,left_on=['Name'],right_on=['Name'])
    print("\t %s"%ID)


#####  gbff file  #####
gbff_path="/data/users/wergillius/reference/GRCh38_p13_rna/GRCh38.p13_rna.gbff"
gbff_df = Pribo.gbff_DF(gbff_path)




# merge ribo-seq with mRNA-seq
print("\n \t merging .....")
tri_merge_df = big_DF.merge(big_quant,left_on=['Transcipt_ID'],right_on=['Name'])
# adding annotation
tri_merge_df = tri_merge_df.merge(gbff_df,left_on=['Transcipt_ID'],right_on=['ID'])

# convert `Bio.Seq` to normal `str`
tri_merge_df.loc[:,'utr'] = tri_merge_df.loc[:,'utr'].apply(lambda x: x.__str__())
tri_merge_df.loc[:,'seq'] = tri_merge_df.loc[:,'seq'].apply(lambda x: x.__str__())

# drop duplicate columns : Transcript ID , ID, Name
tri_merge_df = tri_merge_df.drop(labels=['ID','Name'],axis=1)


#####     Some  Quantification       #####

print("\n \t Calculating Mean ....")
RC_cols = list(filter(lambda x: 'RC_' in x,tri_merge_df.columns))
TC_cols = list(filter(lambda x: 'TC_' in x,tri_merge_df.columns))
TPM_cols = list(filter(lambda x: 'TPM_' in x,tri_merge_df.columns))

cols = [RC_cols,TC_cols,TPM_cols]
names = ['RC','TC','TPM']

for i in range(len(cols)):
    
    col = cols[i]
    col_name = names[i]
    tri_merge_df.loc[:,col_name+"_mean"] = tri_merge_df.loc[:,col].mean(axis=1)
    

print("\n \t Riboseq Count per Million Mean ....")
# for ribo-seq count , we do one more calculation : count per million
library_size = tri_merge_df.loc[:,RC_cols].sum(axis=0)               # shape (4,)
count_per_M = np.divide(tri_merge_df.loc[:,RC_cols],library_size)    # will broadcast 
tri_merge_df.loc[:,"RCPM_mean"] = 1e6*count_per_M.mean(axis=1)    
    
print("\n \t OPM Mean ....")
count_per_length = np.divide(tri_merge_df.loc[:,RC_cols].values,tri_merge_df.loc[:,'EffectiveLength'].values.reshape(-1,1))
total_c_p_l = np.sum(count_per_length,axis=0)
tri_merge_df.loc[:,'OPM'] = np.mean(1e6*count_per_length/total_c_p_l,axis=1)


# ignore the elicit values from individual samples 
brief_col =['Transcipt_ID', 'Length', 'EffectiveLength', 'Gene', 'GeneID','protein_id', 'seq', 'start', 'end', 'utr', 'RC_mean', 'TC_mean','TPM_mean','RCPM_mean','OPM']


#####    Save       #####
tri_merge_df.to_csv(os.path.join(GSE65778_dir,'Ribomap_salmon_out_annotated_full.csv'),index=False)
tri_merge_df[brief_col].to_csv(os.path.join(GSE65778_dir,'Ribomap_salmon_out_annotated_brief.csv'),index=False)

print("\nFULL data save to \n \t \t %s"%os.path.join(GSE65778_dir,'Ribomap_salmon_out_annotated_full.csv'))
print("BRIEF version save to \n \t \t %s"%os.path.join(GSE65778_dir,'Ribomap_salmon_out_annotated_brief.csv'))
print("\n \n Finished ! \(-o-)/")