
from turtle import heading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import rad2deg

# [speed_OS, heading_OS, rel_dist, rel_vel, rel_bearing, rel_course]
properties = ['Risk level',  'Relative distance',  'Relative speed', 'Relative bearing', 'Relative heading', 'Realtive course']

risk_intervals    = ['[0-0.2)', '[0.2-0.4)',  '[0.4-0.6)', '[0.6-0.8)', '[0.8-1]']
speed_intervals   = ['[0-0.5)',   '[0.5-1)',   '[1-1.5)', '[1.5-2)']
distance_interval = ['[0-50)',  '[50-100)', '[100-150)']
bearing_intervals = ['[-135--45)', '[-45-45)', '[45-135)', '[135--45)']
heading_interval  = ['[-135--45)', '[-45-45)', '[45-135)', '[135--45)']
course_interval   = ['[-135--45)', '[-45-45)', '[45-135)', '[135--45)']

all_intervals = [risk_intervals, distance_interval, speed_intervals, bearing_intervals, heading_interval ,course_interval]


index = []
for i, property in enumerate(properties):
    for interval in all_intervals[i]:
        index.append((property, interval))

multi_index = pd.MultiIndex.from_tuples(index)
data = np.zeros((len(index),2), dtype=int)
TABLE = pd.DataFrame(data=data, index=multi_index, columns=['Dynamic obstacles', 'Static obstacles'])


tot_num_dyn = 0
tot_num_stat = 0

def explore_dataset(data_path:str):
    r = 0
    while r < 6096:
        if r % 1000 == 0:
            print('Iteration ', r)
        scan = np.loadtxt(data_path, skiprows=r, max_rows=1)
        num_obst = (len(scan) - 2) // 5
        
        global tot_num_stat, tot_num_dyn

        for i in range(num_obst):
            heading_ts, dist, speed, bearing, course = scan[2+i*5:2+(i+1)*5]
            if heading_ts == 0 and speed == scan[0]:
                tot_num_stat += 1
                add_static_obst(dist, speed, bearing)
            else:
                tot_num_dyn += 1
                heading_os = scan[1]
                heading = heading_os - heading_ts
                add_dynamic_obst(dist, speed, bearing, heading, course)
       
        r += 1
    
            
def add_static_obst(dist, speed, bearing):

    property = properties[1]
    bucket = all_intervals[1]
    if dist < 50:
        TABLE.loc[property, bucket[0]]['Static obstacles'] += 1
    elif dist < 100:
        TABLE.loc[property, bucket[1]]['Static obstacles'] += 1
    else:
        TABLE.loc[property, bucket[2]]['Static obstacles'] += 1
    
    property = properties[2]
    bucket = all_intervals[2]
    if speed < 0.5:
        TABLE.loc[property, bucket[0]]['Static obstacles'] += 1
    elif speed < 1:
        TABLE.loc[property, bucket[1]]['Static obstacles'] += 1
    elif speed < 1.5:
        TABLE.loc[property, bucket[2]]['Static obstacles'] += 1
    else:
        TABLE.loc[property, bucket[3]]['Static obstacles'] += 1

    property = properties[3]
    bucket = all_intervals[3]
    bearing = np.rad2deg(bearing)
    if -135 <= bearing < -45:
        TABLE.loc[property, bucket[0]]['Static obstacles'] += 1
    elif -45 <= bearing < 45:
        TABLE.loc[property, bucket[1]]['Static obstacles'] += 1
    elif 45 <= bearing < 135:
        TABLE.loc[property, bucket[2]]['Static obstacles'] += 1
    else:
        TABLE.loc[property, bucket[3]]['Static obstacles'] += 1
    



def add_dynamic_obst(dist, speed, bearing, heading, course):

    property = properties[1]
    bucket = all_intervals[1]
    if dist < 50:
        TABLE.loc[property, bucket[0]]['Dynamic obstacles'] += 1
    elif dist < 100:
        TABLE.loc[property, bucket[1]]['Dynamic obstacles'] += 1
    else:
        TABLE.loc[property, bucket[2]]['Dynamic obstacles'] += 1
    
    property = properties[2]
    bucket = all_intervals[2]
    if speed < 0.5:
        TABLE.loc[property, bucket[0]]['Dynamic obstacles'] += 1
    elif speed < 1:
        TABLE.loc[property, bucket[1]]['Dynamic obstacles'] += 1
    elif speed < 1.5:
        TABLE.loc[property, bucket[2]]['Dynamic obstacles'] += 1
    else:
        TABLE.loc[property, bucket[3]]['Dynamic obstacles'] += 1

    property = properties[3]
    bucket = all_intervals[3]
    bearing = np.rad2deg(bearing)
    if -135 <= bearing < -45:
        TABLE.loc[property, bucket[0]]['Dynamic obstacles'] += 1
    elif -45 <= bearing < 45:
        TABLE.loc[property, bucket[1]]['Dynamic obstacles'] += 1
    elif 45 <= bearing < 135:
        TABLE.loc[property, bucket[2]]['Dynamic obstacles'] += 1
    else:
        TABLE.loc[property, bucket[3]]['Dynamic obstacles'] += 1
    
    property = properties[4]
    bucket = all_intervals[4]
    heading = np.rad2deg(heading)
    if -135 <= heading < -45:
        TABLE.loc[property, bucket[0]]['Dynamic obstacles'] += 1
    elif -45 <= heading < 45:
        TABLE.loc[property, bucket[1]]['Dynamic obstacles'] += 1
    elif 45 <= heading < 135:
        TABLE.loc[property, bucket[2]]['Dynamic obstacles'] += 1
    else:
        TABLE.loc[property, bucket[3]]['Dynamic obstacles'] += 1

    property = properties[5]
    bucket = all_intervals[5]
    course = np.rad2deg(course)
    if -135 <= course < -45:
        TABLE.loc[property, bucket[0]]['Dynamic obstacles'] += 1
    elif -45 <= course < 45:
        TABLE.loc[property, bucket[1]]['Dynamic obstacles'] += 1
    elif 45 <= course < 135:
        TABLE.loc[property, bucket[2]]['Dynamic obstacles'] += 1
    else:
        TABLE.loc[property, bucket[3]]['Dynamic obstacles'] += 1

    

explore_dataset('../data/metadata_MovingObstaclesNoRules.csv')

print()
TABLE['Static obstacles'] = TABLE['Static obstacles']/tot_num_stat
TABLE['Dynamic obstacles'] = TABLE['Dynamic obstacles']/tot_num_dyn
TABLE = TABLE.round(2)
TABLE['Static obstacles'][-8:] = '-'
print(TABLE)

# print(tot_num_ships)