import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

X = np.loadtxt('../data/simX.txt')
X_pred = np.loadtxt('../data/simX_pred.txt')
N = X.shape[0]
nx = 6
X_pred = X_pred.reshape(-1,nx,N)

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(-40, 40), ylim=(-40, 40))

#Represent robot as circle with radius = 1
robot = plt.Circle((0,0),1)
predicted_traj, = ax.plot(X_pred[:,0,0],X_pred[:,1,0],'y')
#At each time-step, we want to draw the position of the robot and the optimal predicted safe robot trajectory

def init_robot_and_predicted_traj():
    robot.center = (X[0,0],X[0,2])
    ax.add_patch(robot)

    return robot, predicted_traj,


#Creating function for animating trajectory
def animate(i):
    robot.center = (X[i,0],X[i,1])
    predicted_traj.set_xdata(X_pred[:,0,i])
    predicted_traj.set_ydata(X_pred[:,1,i])
    return robot, predicted_traj

anim = FuncAnimation(fig, animate,init_func=init_robot_and_predicted_traj,frames=N,interval=10)

 

#Color feasible region and terminal safe set
d = np.linspace(-200,200,400)
x1,y1 = np.meshgrid(d,d)

#Feasible region
feasible = (x1<=20) & (x1>=-20) & (y1<=20) & (y1>=-20)

#Terminal region
terminal = ((x1<=5) & (x1>=-5) & (y1<=5) & (y1>=-5))

plt.imshow( ((feasible).astype(int)) ,

                extent=(x1.min(),x1.max(),y1.min(),y1.max()),origin="lower", cmap="Greys", alpha = 0.3);

plt.imshow( ((terminal).astype(int)) ,

                extent=(x1.min(),x1.max(),y1.min(),y1.max()),origin="lower", cmap="Greens", alpha = 0.3);

#Adding labels, legends and display text for feasible region and terminal safe set
plt.xlabel('x')
plt.ylabel('y')

handles, labels = plt.gca().get_legend_handles_labels()

blue_circle = Line2D([0], [0], marker='o', color='w', label='Robot',
                        markerfacecolor='b', markersize=15)

yellow_line = Line2D([0], [0], label='Current safety trajectory', color='y')

handles.extend([blue_circle,yellow_line])

plt.legend(handles=handles)

fsize = 'small'
plt.text(-20+4,20-4,'Feasible region',fontsize=fsize)
plt.text(-5+4,5-4,'Terminal safe set',fontsize=fsize)

plt.show()