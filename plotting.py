import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set_theme(style="dark")

# def plot_data_projections(sample, 
#                           title, 
#                           save, 
#                           num_stars=20000, 
#                           bin_size=None,
#                           xlim=[(-5, 5), (-5, 5), (-5, 5)], 
#                           ylim=[(-5, 5), (-5, 5), (-5, 5)], 
#                           figsize=(15, 5),
#                           cmap="magma"):

#     print("INFO: plotting -> {}".format(title))
    
#     # Extract x, y, and z coordinates
#     x = sample[:num_stars, 0].numpy()
#     y = sample[:num_stars, 1].numpy()
#     z = sample[:num_stars, 2].numpy()

#     # Create a list of tuples for each projection
#     projections = [((x, y), "x", "y"), ((x, z), "x", "z"), ((y, z), "y", "z")]

#     fig, axes = plt.subplots(1, 3, figsize=figsize)

#     for idx, ((data_x, data_y), xlabel, ylabel) in enumerate(projections):

#         if bin_size:
#             bin_edges_x = np.linspace(xlim[idx][0], xlim[idx][1], int((xlim[idx][1] - xlim[idx][0]) / bin_size) + 1)
#             bin_edges_y = np.linspace(ylim[idx][0], ylim[idx][1], int((ylim[idx][1] - ylim[idx][0]) / bin_size) + 1)
#             bins=(bin_edges_x, bin_edges_y)
#         else: 
#             bins=100

#         sns.kdeplot(x=data_x, y=data_y, levels=6, color="w", linewidths=1, ax=axes[idx])
#         sns.scatterplot(x=data_x, y=data_y, s=5, color=".15", ax=axes[idx])
#         sns.histplot(x=data_x, y=data_y, bins=bins, cmap=cmap, ax=axes[idx])

#         axes[idx].set_xlim(xlim[idx])
#         axes[idx].set_ylim(ylim[idx])
#         axes[idx].set_xlabel(xlabel)
#         axes[idx].set_ylabel(ylabel)
#         axes[idx].set_title(f"{xlabel} - {ylabel}")
#         plt.grid() 

#     fig.suptitle(title)
#     fig.tight_layout()
#     plt.savefig(save)



def plot_data_projections(sample, 
                          title, 
                          save, 
                          num_stars=20000, 
                          bin_size=None,
                          xlim=[(-5, 5), (-5, 5), (-5, 5)], 
                          ylim=[(-5, 5), (-5, 5), (-5, 5)], 
                          figsize=(15, 10),
                          label=["x (kpc)", "y (kpc)", "z (kpc)"],
                          cmap="magma"):

    print("INFO: plotting -> {}".format(title))
    
    # Extract x, y, and z coordinates
    x = sample[:num_stars, 0]#.numpy()
    y = sample[:num_stars, 1]#.numpy()
    z = sample[:num_stars, 2]#.numpy()

    # Create a list of tuples for each projection
    projections = [((x, y), label[0], label[1]), ((y, z), label[1], label[2]), ((z, x),label[2], label[0])]

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    for idx, ((data_x, data_y), xlabel, ylabel) in enumerate(projections):

        if bin_size:
            bin_edges_x = np.linspace(xlim[idx][0], xlim[idx][1], int((xlim[idx][1] - xlim[idx][0]) / bin_size) + 1)
            bin_edges_y = np.linspace(ylim[idx][0], ylim[idx][1], int((ylim[idx][1] - ylim[idx][0]) / bin_size) + 1)
            bins = (bin_edges_x, bin_edges_y)
        else: 
            bins = bin_edges_x = 100

        sns.histplot(x=data_x, bins=bin_edges_x, color='k', ax=axes[0, idx], element="step", kde = True, alpha=0.4)

        axes[0, idx].set_xlim(xlim[idx])
        axes[0, idx].set_xlabel(xlabel)
        axes[0, idx].set_ylabel('counts')
        axes[0, idx].set_title(f"{xlabel}")
        axes[0, idx].grid()


        sns.kdeplot(x=data_x, y=data_y, levels=6, color="w", linewidths=1, ax=axes[1, idx])
        sns.scatterplot(x=data_x, y=data_y, s=5, color=".15", ax=axes[1, idx])
        sns.histplot(x=data_x, y=data_y, bins=bins, cmap=cmap, ax=axes[1, idx])

        axes[1, idx].set_xlim(xlim[idx])
        axes[1, idx].set_ylim(ylim[idx])
        axes[1, idx].set_xlabel(xlabel)
        axes[1, idx].set_ylabel(ylabel)
        # axes[1, idx].set_title(f"{xlabel} - {ylabel}")
        axes[1, idx].grid()

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(save)







def plot_data(sample, title, save, num_stars=20000, xlim=[-5, 5], ylim=[-5, 5], figsize=(5, 5)):

    # sample = sample.cpu().detach()
    x = sample[:num_stars, 0].numpy()
    y = sample[:num_stars, 1].numpy()
    kde = stats.gaussian_kde(np.vstack([x, y]))
    xx, yy = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    z = np.reshape(kde(positions).T, xx.shape)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.kdeplot(x=x, y=y, levels=6, color="w", linewidths=1, ax=ax)
    sns.scatterplot(x=x, y=y, s=5, color=".15")
    sns.histplot(x=x, y=y, bins=50, cmap="mako")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    fig.tight_layout()
    plt.grid()  
    plt.savefig(save)


def plot_loss(train, valid, args):
    train_loss = train.loss_per_epoch
    valid_loss = valid.loss_per_epoch
    loss_min = valid.loss_min
    fig, ax = plt.subplots(figsize=(8,7))
    plt.plot(range(len(train_loss)), np.array(train_loss), color='b', lw=0.75)
    plt.plot(range(len(valid_loss)), np.array(valid_loss), color='r', lw=0.75, alpha=0.5)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("loss_min={}, epochs={}".format(round(loss_min,6),len(train_loss)))
    fig.tight_layout()
    plt.grid() 
    plt.savefig(args.workdir+'/loss.pdf')
