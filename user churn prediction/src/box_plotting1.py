import seaborn as sns
import matplotlib.pyplot as plt

def box_plotting(data,nrows,ncols,figsize):
  fig,axes=plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)
  # fig, axes = plt.subplots(10, 2, figsize=figsize)
  fig.tight_layout(pad=5.0)  # Increase padding between subplots
  for i,col in enumerate(data.columns):
      ax=axes[i//2,i%2]
      sns.boxplot(x=data[col],ax=ax)
      ax.set_title(col)
  plt.show()