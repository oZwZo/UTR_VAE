echo "please enter config path"
read filename
for i in {3..5};do 
    out_name=$(echo $filename | cut -d "." -f 1)
    
    nohup python script/main_train.py --cuda 3 --config_file $filename --kfold_index $i > ${out_name}_cv${i}.out &
    pids=$!
    echo "PID: "$pids
    done