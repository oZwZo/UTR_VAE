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
        if len(Dataset) ==2:
            self.X_train,self.y_train = Dataset
            self.y_train = np.log(self.y_train) if log_transform else self.y_train
        elif len(Dataset) ==4:
            self.X_train,self.X_test,self.y_train,self.y_test = Dataset
            if log_transform:
                self.y_train = np.log(self.y_train)
                self.y_test = np.log(self.y_test)
        elif len(Dataset) ==6:
            self.X_train,self.X_val,self.X_test,self.y_train,self.y_val,self.y_test = Dataset
            if log_transform:
                self.y_train = np.log(self.y_train)
                self.y_val = np.log(self.y_val)
                self.y_test = np.log(self.y_test)
        
        self.fig = plt.figure(figsize=(12,12))
        self.fit_model_list()
        
    def fit_model_list(self):
        for i in range(len(self.model_ls)):
            self.model_ls[i].fit(self.X_train,self.y_train)
            
    
    def get_set(self,set):
        if set == 'train':
            return self.X_train, self.y_train
        elif set == 'test':
            return self.X_test, self.y_test
        elif set == 'val':
            return self.X_val , self.y_val
    
    def predict_all(self,set='test'):
        """
        set : train, val or test
        """
        try:
            X,_ = self.get_set(set)
        except:
            raise NameError("%s set is not defined" %set)
        self.ypre_ls = [model.predict(X) for model in self.model_ls]
        return self.ypre_ls;
    
    def plot_residual(self,fig=None,n_row=3,n_col=3,set='test',**scatter_args):
        try:
            X,y = self.get_set(set)
        except:
            raise NameError("%s set is not defined" %set)
        # compute all the y pre first
        ypre_ls = self.predict_all(set)
        
        if fig is None:
            fig = self.fig
        for i,c in enumerate(self.C_ls):
            ax = fig.add_subplot(n_row,n_col,1+i)
            ax.scatter(y,y-ypre_ls[i],**scatter_args)
            ax.set_title("%s" %c,fontsize=13)
        
        fig.show()
        
    def compute_metrics(self):
        return None