DIR=/ssd/users/wergillius/tool/meme/bin
input=/ssd/users/wergillius/Project/UTR_VAE/ipynb/network_interpretation/toenrich_sequences.fasta
# input=/ssd/users/wergillius/Project/UTR_VAE/ipynb/network_interpretation/Non_viral_pos.fasta
motif=$1
background=/ssd/users/wergillius/Project/UTR_VAE/ipynb/network_interpretation/UTRlib_background.fasta
# background=/ssd/users/wergillius/Project/UTR_VAE/ipynb/network_interpretation/Non_viral_background.fasta

$DIR/sea --verbosity 1 --text --qvalue -p $input -m $motif -n $background 

