echo "please enter the motif file in <xxx>_cv.txt"
read Query

motif_dir=/ssd/users/wergillius/Project/UTR_VAE/ipynb/network_interpretation/save_motif
for i in {0..9};do
    bash tomtom_HumanRBP.sh ${motif_dir}/${Query}_cv${i}.txt > ${Query}_RBP_cv${i}.tsv &
    done
