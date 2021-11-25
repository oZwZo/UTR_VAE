echo "please enter motif txt file : xxx_cv"
read Query
read Set
motif_DIR=/ssd/users/wergillius/Project/UTR_VAE/ipynb/network_interpretation/save_motif
for i in {0..9} ;do 
    bash run_sea.sh ${motif_DIR}/${Query}_cv$i.txt > ${Query}_${Set}_cv$i.tsv&
    done
