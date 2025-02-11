import numpy as np
from cxt.utils import mse
import matplotlib.pyplot as plt
import seaborn as sns

def plot_inference_scatter(
        yhats, ytrues, filename, subtitle=None,
          tool=r'$\mathbf{cxt(kit)}$', stackit=False):
    
    if stackit:
        ytrues_mean = np.stack(ytrues).mean(0)
        yhats_mean = np.stack(yhats).mean(0)
    else:
        ytrues_mean = ytrues
        yhats_mean = yhats
    error = mse(yhats_mean, ytrues_mean)
    plt.figure(figsize=(4, 4))
    plt.hexbin(ytrues_mean, yhats_mean, gridsize=100, cmap="viridis", mincnt=1)
    plt.plot([0, 15], [0, 15], c="black", linestyle="--")
    plt.title(r"""{} - MSE: {:.4f}
{}""".format(tool, error, subtitle), loc="left", fontsize=10)
    plt.xlabel("True log(Time) [generations]", fontsize=10)
    plt.ylabel("Predicted log(Time) [generations]", fontsize=10)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()