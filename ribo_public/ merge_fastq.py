#! UTF-8
#! envs=pytorch
import os
import sys
from tqdm import tqdm
import argparse
from argparse import ArgumentError

# Arguments

parser = argparse.ArgumentParser()
parser.add_argument("--files",type=list,default=None,required=False,help='The list of abs path of fq to merge')
parser.add_argument("--dir",type=str,default=None,required=False,help="The fq files under this dir will be merge")
parser.add_argument("--out_name",type=str,default=None,required=False,help='the name of merged fq')
args = parser.parse_args()


if (args.files is None) & (args.dir is None):
    raise ArgumentError("--files and --dir can not be empty at the same time")
elif args.files is not None:
    # specify fq mode
    fq_list = args.files
elif args.dir is not None:
    # dir mode
    is_fq = lambda x : ('fastq' in x) | ('fq' in x)
    fq_list = [os.path.join(args.dir,file) for file in os.listdir(args.dir) if is_fq(file)]

if args.out_name is None:
    # get prefix of all fq files
    fq_prefix = [os.path.split(file)[1].split(".")[0] for file in fq_list]
    out_name = "/data/users/wergillius/UTR_VAE/public_ribo" + "+".join(fq_prefix) + ".fastq"
else:
    out_name = args.out_name
    
    
# start merging
with open(out_name,'w') as m:
    for i,fq in enumerate(fq_list):
        print(f"parsing the {i} fq : {fq}")
        with open(fq,'r') as f:
            for line in tqdm(f):
                m.write(line)
            f.close()
    m.close()
    
print("Finish !! ")
print(f"Final merged fq saved to {out_name}")