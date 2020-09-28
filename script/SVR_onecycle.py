import os
import sys
# add the top path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.ML_models import tunning_SVR
import utils

# read A549 csv
A549_df = utils.read_UTR_csv(cell_line = ['A549'])

# the Sequence one hot encoder
soh = utils.Seq_one_hot()

# 
A549_X = soh.d_transform(A549_df)
A549_y = A549_df.TEaverage.values

linear_svr = tunning_SVR(A549_X,A549_y,kernel='linear',log_transform=False);
linear_svr.plot_residual();
linear_svr.fig.savefig('../linear_C_{}.pdf'.format('_'.join(linear_svr.C_ls)),format='pdf')

linear_log = tunning_SVR(A549_X,A549_y,kernel='linear',log_transform=True);
linear_log.plot_residual();
linear_svr.fig.savefig('../linear_log_C_{}.pdf'.format('_'.join(linear_svr.C_ls)),format='pdf')