DIR=/ssd/users/wergillius/tool/meme/bin
Query=$1
Target=/ssd/users/wergillius/tool/meme/motif_databases/RNA/Ray2013_rbp_Homo_sapiens.dna_encoded.meme
$DIR/tomtom -no-ssc -oc . -text -verbosity 1 -min-overlap 5 -dist pearson -evalue -thresh 10.0 $Query $Target 