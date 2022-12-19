from cProfile import label
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as mtick
import matplotlib

import pandas as pd


def do_predictions(model:nn.Module, dataloader:torch.utils.data.DataLoader):
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch) # only one batch in the test set
        
        y_pred  = y_pred.detach().numpy()
        y_batch = y_batch.detach().numpy()
        
    model.train()
    return y_pred, y_batch


def runtime_analysis():
    raise NotImplementedError


def plot_mse(y_pred:np.ndarray, y_true:np.ndarray):
    # At the moment just returning the mse
    mse = mean_squared_error(y_true, y_pred)

    return mse


def plot_predictions(y_pred:np.ndarray, y_true:np.ndarray):
    s = None
    e = 2500
    plt.figure(figsize=(15,10))
    plt.plot(y_pred[s:e], label='Predicted risk', linewidth=3)
    plt.plot(y_true[s:e], label='True risk', linewidth=3)
    plt.legend(fontsize=30)
    plt.ylabel('Collision risk', fontsize=26)
    plt.xlabel('Time', fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()

def plot_multiple_predictions(y_pred:np.ndarray, y_true:np.ndarray, labels:list):
    
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    plt.rc('axes', labelsize=22)
    
    grays = matplotlib.cm.get_cmap('gist_gray')
    color = ['#fdbb84', '#ef6548', '#990000']
    color = ['#8073ac', '#35978f', '#990000']
    # color = ['#fc8d59', '#d7301f', '#990000']

    s = 2470
    e = 2470+1800

    s = 2470
    e = 2470+500


    plt.figure(figsize=(12,8))
    
    # for prediction in y_pred:
    plt.plot(y_pred[s:e,0], label=labels[0], linewidth=3, color=color[0])
    plt.plot(y_pred[s:e,1], label=labels[1], linewidth=3, color=color[1])
    plt.plot(y_pred[s:e,2], label=labels[2], linewidth=3, color=color[2])
    plt.plot(y_true[s:e], label='True risk', linewidth=3, linestyle='--', color='black')
    plt.legend(fontsize=22, labelcolor=grays(70))
    plt.ylim([0,1])
    plt.ylabel('Collision risk')
    plt.xlabel('Measurement')
    plt.xticks()
    plt.yticks(fontsize=20)
    

    # fig, axes = plt.subplots(3, 1, figsize=(15,12))

    # for i, ax in enumerate(axes):
    #     ax.plot(y_true[s:e], label='True risk', linewidth=2, linestyle='--', color='black')
    #     ax.plot(y_pred[s:e,i], label=labels[i], linewidth=4)
    #     ax.legend(fontsize=22)
    #     ax.set_ylim([-0.05,1.15])
    #     ax.set_ylabel('Collision risk')
    #     ax.set_xticks([])

    # axes[-1].set_xlabel('Time')
    
    # fig.savefig('../Plots/Performance_all_separate.eps', bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


def plot_loss(training_loss, validation_loss):
    
    epochs = np.arange(len(training_loss))

    plt.figure(figsize=(15,15))
    plt.plot(epochs, training_loss, label='Training_loss')
    plt.plot(epochs, validation_loss, label='Validation loss')
    plt.legend()
    plt.show()


def plot_evaluation_metrics_single_agent():
    '''
    3M training steps
    CNN    Progress COLAV CTE Time
    1conv_random 98.03 98 30 1018
    1conv_locked 94.16 98 26 1515
    1conv_unlocked 98.28 97 33 971
    3conv_random 97.06 100 182 1694
    3conv_locked 93.45 90 20 1010
    3conv_unlocked 94.19 99 113 1894
    Deep_random 44.75 95 774 8278
    Deep_locked 71.93 90 71 3210
    Deep_unlocked 85.28 100 471 3347

    1M trainingsteps
    CNN    Progress COLAV CTE Time
    Deep_unlocked 97.82 96 50 937
    '''

    df = pd.read_csv('results_RL/DRL_performance.txt', header=0, usecols=[1,2,3,4])
    # df = pd.read_csv('results_RL/DRL_performance_1M.txt', header=0, usecols=[1,2,3,4])

    metrics = ['Progress', 'COLAV', 'CTE', 'Time']
    ylabels = ['Avg. Progress', 'COLAV', 'CTE [m]', 'Time [s]']

    x = [0,0.5,1]
    bar_width = 0.45

    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=26)
    plt.rc('ytick', labelsize=26)
    plt.rc('axes', labelsize=26)
    # plt.axis('scaled')

    cmap  = matplotlib.cm.get_cmap('Reds')
    color = [cmap(int(1*(255/8))), cmap(int(4*(255/8))), cmap(int(6*(255/8)))]

    grays = matplotlib.cm.get_cmap('gist_gray')
    color = ['#fdbb84', '#ef6548', '#990000']

    fig, ax = plt.subplots(1, 4, figsize=(11,10))

    
    for i in range(4):
        ax[i].bar(x, df[metrics[i]][6:9].to_numpy(), align='center', width=bar_width, color=color)#, hatch=['xx', '..', '/'])
        ax[i].set_ylabel(ylabels[i])
        ax[i].set_xticks([])

    fig.tight_layout( rect=[0.04, 0.1, 1, 1])

    ax[0].set_ylim([0,100])
    # ax[1].set_ylim([50,100])
    ax[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter())
          
    labels = ['Random', 'Locked', 'Unlocked']
    handles = [plt.Rectangle((0,0),1,1, color=color[i]) for i in range(3)]

    bb = (fig.subplotpars.left, fig.subplotpars.bottom-0.12, fig.subplotpars.right-fig.subplotpars.left,-0.2)
    ax[0].legend(handles, labels, loc='lower center', bbox_to_anchor=bb, ncol=3, fontsize=26, mode="expand", borderaxespad=0., bbox_transform=fig.transFigure, labelcolor=grays(80))
 
    fig.savefig('1conv_metrics.pdf', bbox_inches='tight')

    plt.show()



def plot_evaluation_metrics_multiple_agents():
    '''
    CNN    Progress COLAV CTE Time
    1conv_random 98.03 98 30 1018
    1conv_locked 94.16 98 26 1515
    1conv_unlocked 98.28 97 33 971
    3conv_random 97.06 100 182 1694
    3conv_locked 93.45 90 20 1010
    3conv_unlocked 94.19 99 113 1894
    Deep_random 44.75 95 774 8278
    Deep_locked 71.93 90 71 3210
    Deep_unlocked 85.28 100 471 3347
    '''

    df = pd.read_csv('results_RL/DRL_performance.txt', header=0, usecols=[1,2,3,4])
    df = df.values

    agents = ['1conv', '3conv', 'DeepCNN']
    metrics = ['Progress', 'COLAV', 'CTE', 'Time']
    ylabels = ['Avg. Progress', 'COLAV', 'CTE [m]', 'Time [s]']

    x = np.arange(3)

    bar_width = 0.25
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    plt.rc('axes', labelsize=22)

  
    fig, ax = plt.subplots(4, 1, figsize=(9,10))

    cmap  = matplotlib.cm.get_cmap('viridis')

    # c = ['salmon', 'yellowgreen', 'skyblue']
    color = ['#fdbb84', '#ef6548', '#990000']
    grays = matplotlib.cm.get_cmap('gist_gray')
    
    for i in range(4):
        ax[i].bar(x-bar_width, df[0::3,i], align='center', width=bar_width, color=color[0], label='Random')
        ax[i].bar(x,           df[1::3,i], align='center', width=bar_width, color=color[1], label='Locked')
        ax[i].bar(x+bar_width, df[2::3,i], align='center', width=bar_width, color=color[2], label='Unlocked')
        ax[i].set_ylabel(ylabels[i])
        ax[i].set_xticks(x, agents)
        # ax[i].legend()
    ax[0].set_ylim([40,100])
    ax[1].set_ylim([40,100])
    ax[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter())

    ax[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, labelcolor=grays(80) , fontsize=22) #prop={'family':'seif'}

    fig.tight_layout()#rect=[0, 0.1, 1, 1])
    plt.show()

def plot_mse_histogram():
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    plt.rc('axes', labelsize=22)

    fig, ax = plt.subplots(1, 3, figsize=(14,6))

    grays = matplotlib.cm.get_cmap('gist_gray')

    mse_1conv = np.loadtxt('results_RL/_mse_1conv.txt')
    mse_3conv = np.loadtxt('results_RL/_mse_3conv.txt')
    mse_deep  = np.loadtxt('results_RL/_mse_deep.txt')

    mean_1conv = mse_1conv.mean()
    std_1conv = mse_1conv.std()
    mean_3conv = mse_3conv.mean()
    std_3conv = mse_3conv.std()
    mean_deep = mse_deep.mean()
    std_deep = mse_deep.std()

    # print(mean_deep, mean_1conv, mean_3conv)
    # 
    bins = np.linspace(0,0.125,26)
    print(bins)
    # [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14]
    # n, edges, bars = plt.hist(mse_1conv, 25, density=False, facecolor='#990000', edgecolor='black')
    _, _, n0 = ax[0].hist(mse_1conv, bins, density=False, facecolor='#990000', edgecolor='black')
    _, _, n1 = ax[1].hist(mse_3conv, bins, density=False, facecolor='#990000', edgecolor='black')
    _, _, n2 = ax[2].hist(mse_deep, bins, density=False, facecolor='#990000', edgecolor='black')

    
    for i in range(3):
        # ax[i].set_xlim([0,0.13])
        ax[i].set_ylim([0,30])
    
    ax[0].set_xlabel('MSE\n\n (a) 1conv')
    ax[1].set_xlabel('MSE\n\n (b) 3conv')
    ax[2].set_xlabel('MSE\n\n (c) DeepCNN')
    # ax[i].set_xticks([0, 0.025, 0.05, 0.075, 0.1, 0.0125])
    ax[0].set_title(f'   Mean: {mean_1conv:.2e}\nStandard deviation: {std_1conv:.2e}', color=grays(70), fontsize=20)
    ax[1].set_title(f'   Mean: {mean_3conv:.2e}\nStandard deviation: {std_3conv:.2e}', color=grays(70), fontsize=20)
    ax[2].set_title(f'   Mean: {mean_deep:.2e}\nStandard deviation: {std_deep:.2e}', color=grays(70), fontsize=20)

    
    # n0.datavalues = [n if n>0 else '' for n in n0.datavalues]
    # ax[0].bar_label(n0)
    # ax[1].bar_label(n1)
    # ax[2].bar_label(n2)

    fig.tight_layout()
    plt.show()
    
