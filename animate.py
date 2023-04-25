import os
import numpy as np
import matplotlib.pyplot as plt


def animate(x, fn=None):
    if x.shape[-1] == 3:
        return animate3(x, fn or "magnet")
    if x.shape[-1] == 2:
        return animate2(x, fn or "gravity")
    

def animate3(x, fn="magnet"):
    os.system("rm /tmp/magnet*png")

    smallestx, biggestx = np.min(x[:,:,0]), np.max(x[:,:,0])
    smallesty, biggesty = np.min(x[:,:,1]), np.max(x[:,:,1])
    smallestz, biggestz = np.min(x[:,:,2]), np.max(x[:,:,2])

    T = x.shape[0]

    for t in range(T):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x[t,:,0], x[t,:,1], x[t,:,2])

        ax.set_xlim([smallestx, biggestx])
        ax.set_ylim([smallesty, biggesty])
        ax.set_zlim([smallestz, biggestz])

        plt.savefig("/tmp/magnet_%03d.png"%t)
        plt.close()
    
    os.system(f"gm convert -delay 5 -loop 0 /tmp/magnet_*.png /tmp/{fn}.gif")

def animate2(x, fn='gravity', other=None):
    os.system("rm /tmp/gravity*png")


    if fn == "drag3":
        smallest, biggest = np.min(x[:,0]), np.max(x[:,0])
    else:
        smallest, biggest = np.min(x), np.max(x)

    T = x.shape[0]

    if other is not None:
        other_smallest, other_biggest = np.min(other), np.max(other)
        smallest = min(smallest, other_smallest)
        biggest = max(biggest, other_biggest)
        T = min(T, other.shape[0])
    
    
    for t in range(T):
        plt.figure()
        plt.xlim([smallest, biggest])
        plt.ylim([smallest, biggest])

        plt.scatter(x[t,:,0], x[t,:,1], label="prediction")
        if other is not None:
            plt.scatter(other[t,:,0], other[t,:,1], c="r", label="actual")
            plt.legend(loc="upper left")
        
        plt.savefig("/tmp/gravity_%03d.png"%t)
        plt.close()
    
    os.system(f"gm convert -delay 5 -loop 0 /tmp/gravity_*.png /tmp/{fn}.gif")
