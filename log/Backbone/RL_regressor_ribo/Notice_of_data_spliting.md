# inappropriate spliting

Added By: Wergilius Z
Created: Feb 17, 2021 11:46 AM

# Inappropriate spliting

## starting:

- we found `test_run/big_capacity_slr.ini` perform very good but can't repeat at all.
- And we later found out that the training process of it, `random_split` is not specify with `torch.Generator().manual_seed(seed)`
- so that we

for model : RL_regressor

run : =`log/Backbone/RL_regressor_ribo/test_run/big_capacity_slr.ini`

### training set

![fig_of_md/Untitled.png](fig_of_md/Untitled.png)

### validation set

![fig_of_md/Untitled%201.png](fig_of_md/Untitled%201.png)

## It looks like very good generalization

## we would like to split in a per-mutated seed

test run = `log/Backbone/RL_regressor_ribo/test_run/bc_test.ini` which train on seed=42

### Seed=42

### train set

![fig_of_md/Untitled%202.png](fig_of_md/Untitled%202.png)

### val set

![fig_of_md/Untitled%203.png](fig_of_md/Untitled%203.png)

### Seed=51 : which will re-mix the train and val set

this will re-mix the train and val set

thus the performance will be a average of two 

### train set

![fig_of_md/Untitled%204.png](fig_of_md/Untitled%204.png)

### val set

![fig_of_md/Untitled%205.png](fig_of_md/Untitled%205.png)

### therefore , it's highly possible the high performance of  `big_capacity_slr` is caused by inconsistent spliting seed