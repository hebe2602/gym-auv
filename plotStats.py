import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_stats(df, label = None, var = 'rewards', window_size = 100, n_timesteps = 2000000, xaxis = 'timesteps'):

   #rewards = df[var].values
   #print('rewards: ', rewards, 'len: ', len(rewards))

   # calculate the smoothed rewards using a moving average
   smoothed_rewards = df[var].rolling(window_size, center = True, min_periods=1).mean()

   # calculate the smooth standard deviation using a moving standard deviation
   smoothed_std = df[var].rolling(window_size, center = True, min_periods=1).std()
   

   #plt.plot(df['rewards'], label='Original')
   plt.figure(1, figsize=(8, 6))
   if xaxis == 'timesteps':
      timesteps = np.arange(len(df[var])) * n_timesteps / len(df[var])
      plt.plot(timesteps, smoothed_rewards,label = label) #label='Smoothed '+ var[:-1].capitalize() + ' ' + label)
      plt.fill_between(timesteps, smoothed_rewards-smoothed_std, smoothed_rewards+smoothed_std, alpha=0.2) #, label='Smoothed Std Dev ' + label)
      plt.xlabel('Timesteps',  fontsize=14)

   elif xaxis == 'episodes':
      plt.plot(smoothed_rewards,label = label)
      plt.fill_between(df.index, smoothed_rewards-smoothed_std, smoothed_rewards+smoothed_std, alpha=0.2) #, label='Smoothed Std Dev')
      plt.xlabel('Episodes',  fontsize=14)

   
   plt.ylabel('Average Episode ' + var[:-1].capitalize(),  fontsize=14)


   # set the plot to grayscale
   ax = plt.gca()
   ax.set_facecolor('lavender')

   # add grid to the plot
   ax.grid(True, linestyle='--', color='white', linewidth=0.5)

   legend = plt.legend(loc='best', fontsize=14)#, bbox_to_anchor=(1, 0.5))
   legend.get_frame().set_facecolor('lavender')



save_fig_path = os.path.join('/home/sveinjhu/Documents/Masteroppgave/logs/figures/RandomScenario1-v0/Plots')

dir_list = ['SF_3_lidar', 'SF_3_lidar_and_obst', 'SF_3_no_lidar']


label_list = ['PSF with lidar', 'PSF with lidar and moving obstacles', 'PSF without lidar']
var_list = ['rewards', 'cross_track_errors', 'progresses', 'timesteps', 'durations','collisions', ]
var_index = 4
window_size = 50
for var_index in range(6):
   for i in range(len(dir_list)):
      path = os.path.join('/home/sveinjhu/Documents/Masteroppgave/logs/figures/RandomScenario1-v0', dir_list[i])
      df = pd.read_csv(os.path.join(path, 'stats.csv'))
      plot_stats(df, label = label_list[i], var = var_list[var_index], xaxis='episodes', window_size=window_size) #xaxis='episodes'

   #plt.show()
   plt.savefig(os.path.join(save_fig_path, var_list[var_index]+'.png'))
   plt.clf()



# # Create a sample plot with three lines
# x = [1, 2, 3, 4, 5]
# y1 = [2, 4, 6, 8, 10]
# y2 = [1, 3, 5, 7, 9]
# y3 = [10, 8, 6, 4, 2]

# fig, ax = plt.subplots()
# ax.plot(x, y1, label=label_list[0])
# ax.plot(x, y2, label=label_list[1])


# # Create a separate figure with only the legends
# fig_legend, ax_legend = plt.subplots(figsize=(2, 0.5))
# ax_legend.axis('off')
# ax_legend.legend(*ax.get_legend_handles_labels(), loc='center', ncol=3)

# plt.show()


# vessel_speed_density = np.loadtxt('resources/speed_density.txt')[:30]
# speed_sum = np.sum(vessel_speed_density)
# normalized_speed_density = vessel_speed_density / speed_sum
# np.set_printoptions(precision=20)
# print(normalized_speed_density)

