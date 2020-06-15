"""-----------------------------------------------------------------------------
--------------------------- Output Visualization -------------------------------
--------------------------------------------------------------------------------
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualize(object):
    """This class is used for visualization
    """ 
    def __init__(self, ofname):
        print ("initialize visualize class")
        self.df= pd.read_csv(ofname) 
        # preprocess dataframe
        self.preprocess()
        self.plot_convergence()
    
    def preprocess(self):
        """This function is used to preprocess the dataframe for plotting"""
        itercount=[i+1 for i in range(len(self.df))]
        self.df["iteration"]=itercount
        self.df["err"]=self.df["err"].apply(lambda x: x*-1)
        min_errval= float('inf')
        err_val=self.df["err"].values.tolist()
        min_err=[]
        for cur in err_val:
            if cur < min_errval:
                 min_errval=cur
            min_err.append(min_errval)
        self.df["min_err"]=min_err
        
    def plot_convergence(self):
       """This function is used to plot minimum value after each iteration"""
       sns.set_style("whitegrid",{"font.family": ["serif"],"font.serif" : "Arial"})
       g = sns.lineplot(x="iteration", y="min_err", markers=True,  
                        dashes=False, data=self.df)
       plt.show()

