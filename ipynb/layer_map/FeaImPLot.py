import os
import sys
import pickle
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# some function
model_path = sys.argv[1]


"""
```python
[in]  X_test.shape

[out] (25000, 256,29)
```
"""

def reload_RF_model(path):
#     path = os.path.join(os.path.abspath("."), path)
    with open(path, 'rb') as m:
        model = pickle.load(m)
    return model

def feat_imp_plot_2(MeanDim_imprt, MaxDim_imprt):
    sns.set_theme(style='ticks',font_scale=1)

    fig, axs = plt.subplots(1,2, figsize=(14,11),dpi=200)
    sns.barplot(x=MeanDim_imprt['values'].iloc[:30], y=MeanDim_imprt['channel'].iloc[:30],
                orient='h', ax=axs[0]  )
    axs[0].set_ylabel("")
    axs[0].set_xlabel("")
    axs[0].set_title('Mean Importance over sites',fontsize=16)

    sns.barplot(x=MaxDim_imprt['values'].iloc[:30], y=MaxDim_imprt['channel'].iloc[:30],
                orient='h', ax=axs[1])
    axs[1].set_ylabel("")
    axs[1].set_xlabel("")
    axs[1].set_title('Max Importance over sites',fontsize=16)
    fig.savefig(model_path.replace("model","plot").replace(".plot","_p2.png"), format='png', transparent=True)

def feat_imp_plot_h(MaxDim_imprt):
    sns.set_theme(style='ticks',font_scale=1)
    fig=plt.figure(figsize=(11,6),dpi=200)
    ax = fig.gca()
    sns.barplot(x=MaxDim_imprt['channel'].iloc[:30], y=MaxDim_imprt['values'].iloc[:30])
    plt.xticks(rotation=60, horizontalalignment='right');
    plt.ylabel('Feautre Importance', fontsize=16)
    plt.xlabel("")
    fig.savefig(model_path.replace("model","plot").replace(".plot","_ph.png"), format='png', transparent=True)
    
    
    
test_model = reload_RF_model(model_path)
test_impt = test_model.feature_importances_.reshape(256, -1)

MaxDim_imprt = pd.DataFrame([test_impt.max(axis=1), ["filter_%d"%i for i in range(256)]]).T
MaxDim_imprt.columns = ['values','channel']
MaxDim_imprt.sort_values('values', ascending=False, inplace=True)

MeanDim_imprt = pd.DataFrame([test_impt.mean(axis=1), ["filter_%d"%i for i in range(256)]]).T
MeanDim_imprt.columns = ['values','channel']
MeanDim_imprt.sort_values('values', ascending=False, inplace=True)

feat_imp_plot_2(MeanDim_imprt, MaxDim_imprt)
feat_imp_plot_h(MaxDim_imprt)