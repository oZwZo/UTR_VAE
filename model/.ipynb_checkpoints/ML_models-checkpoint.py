from sklearn.svm import SVR,LinearSVR
import numpy as np
import matplotlib.pyplot as plt

class tunning_SVR():
    def __init__(self,Dataset,C_ls=[1,5,10,20,50,100,500,1000],kernel='rbf',log_transform=False,**kwargs):
        
        # to accelerate the training
        self.model_ls = [SVR(C=c,kernel=kernel,epsilon=1e-4,**kwargs) for c in C_ls]
        self.log_ransform = log_transform
        self.C_ls = C_ls
        self.Dataset = Dataset
        if len(Dataset=2):
        self.X_train 
        self.y = np.log(y) if log_transform else y
        self.fig = plt.figure(figsize=(12,12))
        self.fit_model_list()
        
    def fit_model_list(self):
        for i in range(len(self.model_ls)):
            self.model_ls[i].fit(self.X,self.y)
            
    def predict_all(self):
        self.ypre_ls = [model.predict(self.X) for model in self.model_ls]
        return self.ypre_ls;
    
    def plot_residual(self,fig=None,n_row=3,n_col=3,**scatter_args):
        # compute all the y pre first
        ypre_ls = self.predict_all()
        
        if fig is None:
            fig = self.fig
        for i,c in enumerate(self.C_ls):
            ax = fig.add_subplot(n_row,n_col,1+i)
            ax.scatter(self.y,self.y-ypre_ls[i],**scatter_args)
            ax.set_title("%s" %c,fontsize=13)
        
        fig.show()
        
    def compute_metrics(self):
        return None