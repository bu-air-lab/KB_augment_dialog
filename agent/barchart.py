from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def plot_three():


	fig = plt.figure()
	ax1 = fig.add_subplot(111, projection='3d')

	xlabels=np.array(['','r0','','','r1','','','r2',''])
	#xpos = np.arange(xlabels.shape[0])
	xpos = [1,1,1,2,2,2,3,3,3]
	ypos=[1,2,3,1,2,3,1,2,3]
	ylabels=np.array(['','p0','','','p1','','','p2',''])
	#ypos = np.arange(ylabels.shape[0])
	num_elements = len(xpos)
	zpos = [0,0,0,0,0,0,0,0,0,0]
	dx = np.ones(9)*0.5
	dy = np.ones(9)*0.5
	dz = [0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11,0.11]


	colors = ['g','g','g','g','g','g','g','g','g']
	for i in range(9):
		ax1.bar3d(xpos[i], ypos[i], zpos[i], dx[i], dy[i], dz[i], color=colors[i])

	#ax1.w_xaxis.set_ticks(xpos + 0.5/2.)
	#ax1.w_yaxis.set_ticks(ypos + 0.5/2.)
	ax1.w_xaxis.set_ticklabels(xlabels)
	ax1.w_yaxis.set_ticklabels(ylabels)
	ax1.set_xlabel('Recipient')
	ax1.set_ylabel('Patient')
	ax1.set_zlabel('Belief')
	ax1.set_xlim(0.5, 3.5)
        ax1.set_ylim(0.5, 3.5)
        ax1.set_zlim(0.0, 0.2)
	plt.show()

def plot_four():


	fig = plt.figure()
	ax1 = fig.add_subplot(111, projection='3d')

	xlabels=np.array(['','r0','','r1','','r2','','r3','','','','','','','',''])
	#xpos = np.arange(xlabels.shape[0])
	xpos = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
	ypos=[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]
	ylabels=np.array(['','p0','','p1','','p2','','p3','','p4','','','','','',''])
	#ypos = np.arange(ylabels.shape[0])
	num_elements = len(xpos)

	zpos=[]
        for i in range(len(xpos)):
	        zpos.append(0)
	dx = np.ones(len(xpos))*0.5
	dy = np.ones(len(xpos))*0.5
	dz = [0.08,0.08,0.08,0.15,0.08,0.08,0.08,0.15,0.08,0.08,0.08,0.15,0.15,0.15,0.15,0.15]


	colors = ['g','g','g','r','g','g','g','r','g','g','g','r','r','r','r','r']
	for i in range(len(dz)):
		ax1.bar3d(xpos[i], ypos[i], zpos[i], dx[i], dy[i], dz[i], color=colors[i])

	#ax1.w_xaxis.set_ticks(xpos + 0.5/2.)
	#ax1.w_yaxis.set_ticks(ypos + 0.5/2.)
	ax1.w_xaxis.set_ticklabels(xlabels)
	ax1.w_yaxis.set_ticklabels(ylabels)
	ax1.set_xlabel('Recipient')
	ax1.set_ylabel('Patient')
	ax1.set_zlabel('Belief')
	ax1.set_xlim(0.5, max(xpos) +0.5)
        ax1.set_ylim(0.5, max(ypos) +0.5)
        ax1.set_zlim(0.0, 0.2)
	plt.show()




def main():

	plot_four()


if __name__ == '__main__':

	main()


