import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    """
    to plot the results at the end of the training.
    """

    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="blue")
    ax.set_xlabel("Training Steps", color="blue")
    ax.set_ylabel("Epsilon", color="blue")
    ax.tick_params(axis='x', colors="blue")
    ax.tick_params(axis='y', colors="blue")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="red")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="red")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="red")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


def loss_plot(losses, losses_file):
    plt.close()
    plt.plot(losses, label="Loss", color="blue")
    plt.legend()
    plt.savefig(losses_file)

def save_frames_as_gif(frames, episode, path = 'gifs/'):
    filename = 'episode_' +str(episode) +'.gif'

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1]/24, frames[0].shape[0]/24 ))#, dpi = 100)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=100)
    anim.save(path + filename, writer='pillow', fps=100)
    plt.close()


def clip_reward(reward):
    return np.sign(reward)
