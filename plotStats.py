import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_stats(df, label = None, var = 'rewards', window_size = 50, n_timesteps = 240000, xaxis = 'timesteps'):

   #rewards = df[var].values
   #print('rewards: ', rewards, 'len: ', len(rewards))

   # calculate the smoothed rewards using a moving average
   smoothed_rewards = df[var].rolling(window_size, center = True, min_periods=1).mean()

   # calculate the smooth standard deviation using a moving standard deviation
   smoothed_std = df[var].rolling(window_size, center = True, min_periods=1).std()
   

   #plt.plot(df['rewards'], label='Original')
   plt.figure(1)
   if xaxis == 'timesteps':
      timesteps = np.arange(len(df[var])) * n_timesteps / len(df[var])
      plt.plot(timesteps, smoothed_rewards,label = label) #label='Smoothed '+ var[:-1].capitalize() + ' ' + label)
      plt.fill_between(timesteps, smoothed_rewards-smoothed_std, smoothed_rewards+smoothed_std, alpha=0.2) #, label='Smoothed Std Dev ' + label)
      plt.xlabel('Timesteps')
   elif xaxis == 'episodes':
      plt.plot(smoothed_rewards,label = label)
      plt.fill_between(df.index, smoothed_rewards-smoothed_std, smoothed_rewards+smoothed_std, alpha=0.2) #, label='Smoothed Std Dev')
      plt.xlabel('Episodes')

   
   plt.ylabel('Average Episode ' + var[:-1].capitalize())
   plt.legend()



#dir_list = ['gamma_1', 'gamma_5', 'gamma_10', 'gamma_20', 'gamma_50', 'No_SF']
#dir_list = ['PSF', 'No_PSF']
#dir_list=['terminal_set', 'SF_good','realistic_model']
dir_list=['PSF', 'Lidar_detetction']

label_list = ['No_lidar', 'Lidar']
var_list = ['rewards', 'cross_track_errors', 'progresses', 'timesteps', 'durations','collisions']
var_index = 0

for i in range(len(dir_list)):
   path = os.path.join('/home/sveinjhu/Documents/Masteroppgave/logs/figures/RandomScenario0-v0', dir_list[i])
   df = pd.read_csv(os.path.join(path, 'stats.csv'))
   plot_stats(df, label = label_list[i], var = var_list[var_index], xaxis='episodes')

plt.show()