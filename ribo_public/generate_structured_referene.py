import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
import argparse

# parser = argparse.ArgumentParser("A script to convert gbff file into csv")
# parser.add_argument("--gbff",type=str,default='/data/users/wergillius/reference/GRCh38_p13_rna/GRCh38.p13_rna.gbff')
# parser.add_argument("--convert",type=str,default="/data/users/wergillius/UTR_VAE/public_poly/GSE129651_ID_conversion.csv")
# args = parser.parse_args()

def read_gb(SeqRecord):
    """
    read gbff term , extract info of interest
    ...input : `SeqRecord` object from Bio.SeqIO.parse("gbff")
    """
    cds_index = None
    for i in range(len(SeqRecord.features)):
        if SeqRecord.features[i].type == 'CDS':
            cds_index = i
    # ignore terms that have no CDS annotation
    if cds_index is None:
        raise AssertionError("Not CDS !!")
        
    gb_dict={}
    
    # the term to extract
    gb_dict["ID"] = SeqRecord.id
    gb_dict['Gene'] = SeqRecord.features[cds_index].qualifiers['gene'][0]
    
    # there are several term
    for xref in SeqRecord.features[cds_index].qualifiers['db_xref']:
        if 'GeneID:' in xref:
            with_id = xref 
    gb_dict['GeneID'] = with_id.split('GeneID:')[-1]
    
    gb_dict['protein_id'] = SeqRecord.features[cds_index].qualifiers['protein_id'][0]
    
    gb_dict['seq'] = SeqRecord.seq
    gb_dict['start'] = SeqRecord.features[cds_index].location.nofuzzy_start
    gb_dict['end'] = SeqRecord.features[cds_index].location.nofuzzy_end
    
    gb_dict['utr'] = SeqRecord.seq[:gb_dict['start']]
    
    
    return gb_dict

def str_to_list(x):
    if type(x) ==str:
        x = [x]
    return x

def merging_ref_rna(convert_df):
    candidate_transcript = []
    for convert in convert_df.loc[:,'refseq.rna'].values:
        if type(convert) == str:
            candidate_transcript += [convert]
        elif type(convert) == list:
            candidate_transcript += convert
        elif pd.isna(convert):
            continue
        else:
            raise AssertionError("type %s"%type(convert))
    return candidate_transcript

#  ======= read candidte NCBI ID =========
# convert df is a DF containing `ensembl ID` and `NCBI ID`
convert_path="/data/users/wergillius/UTR_VAE/public_poly/GSE129651_ID_conversion.csv"
convert_df= pd.read_csv(convert_path)

# A Gene (`ensembl ID`) will correspond to more than one transcript ID `NM_00xxxxx.1 / XM_00xxxx.1`
candidate_transcript = merging_ref_rna(convert_df)

# ======= read gbff file ==========

gbff = SeqIO.parse(args.gbff)
# Bio.SeqIO can read gbff into a SeqRecord but most of the info are in the attr `features`

gbff_ls = []
for SeqRecord in tqdm(gbff):
    if SeqRecord.id in candidate_transcript:      # only extract a subset of gbff
        # ----- extract info ----
        try:
            gbff_ls.append(read_gb(SeqRecord))    # the core function here
        
        except AssertionError:                    # for SeqRecord that have no CDS
            continue

# list fo dict,json
# convert json to dataframe
gbff_df = pd.json_normalize(gbff_ls)

# some useful feature
gbff_df.loc[:,'utr_len'] = gbff_df.loc[:,'utr'].apply(lambda x: len(x))
gbff_df.loc[:,'seq_len'] = gbff_df.loc[:,'seq'].apply(lambda x: len(x))
gbff_df.loc[:,'with_uAUG'] = gbff_df.loc[:,'utr'].apply(lambda x: 'ATG' in x)

gbff_df.to_csv(os.path.join(reference_path,'GRCh38_p13_rna_structure_gbff.csv'))