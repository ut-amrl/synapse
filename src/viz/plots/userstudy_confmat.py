import torchmetrics
import torch
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

val = torch.tensor([
    [76.85, 66.29, 67.74],
    [65.66, 73.75, 69.28],
    [65.46, 68.97, 73.82]
], dtype=torch.long)

fig2, ax2 = plot_confusion_matrix(conf_mat=val.numpy(), class_names=["user-1", "user-2", "user-3"], figsize=(12,12), fontcolor_threshold=0.9)

# display the figure
plt.show()

# fig2.savefig("/home/dynamo/AMRL_Research/repos/nspl/scripts/safety/plots/figs/userstudy_confusion_matrix.png", dpi=300)