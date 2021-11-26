DIR=/ssd/users/wergillius/tool/meme/bin
Query=$1
Target=$2
$DIR/tomtom -no-ssc -oc . -text -verbosity 1 -min-overlap 5 -dist pearson -evalue -thresh 10.0 $Query $Target 