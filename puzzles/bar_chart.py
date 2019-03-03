import matplotlib.pyplot as plt
import numpy as np

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

# Offset the right spine of par2.  The ticks and label have already been
# placed on the right by twinx above.
par2.spines["right"].set_position(("axes", 1.2))
# Having been created by twinx, par2 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(par2)
# Second, show the right spine.
par2.spines["right"].set_visible(True)

# p1, = host.plot([0, 1, 2], [0, 1, 2], "b-", label="Density")
# p2, = par1.plot([0, 1, 2], [0, 3, 2], "r-", label="Temperature")
# p3, = par2.plot([0, 1, 2], [50, 30, 15], "g-", label="Velocity")
p1 = host.bar(np.array([1,2,3]) - 0.2, [3, 3, 2], color="b", label="Density", width=0.2)
p2 = par1.bar(np.array([1,2,3]), [1, 2, 4], color="r", label="Temperature", width=0.2)
p3 = par2.bar(np.array([1,2,3]) + 0.2, [3, 1, 1], color="g", label="Velocity", width=0.2)

# host.set_xlim(0, 4)
host.set_ylim(0, 5)
par1.set_ylim(0, 5)
par2.set_ylim(1, 5)

host.set_xlabel("Distance")
host.set_ylabel("Density")
par1.set_ylabel("Temperature")
par2.set_ylabel("Velocity")

host.set_xticklabels(('a', 'b', 'c'))

host.yaxis.label.set_color(color="b")
par1.yaxis.label.set_color(color="r")
par2.yaxis.label.set_color(color="g")
#
# tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors="b")
par1.tick_params(axis='y', colors="r")
par2.tick_params(axis='y', colors="g")
# host.tick_params(axis='x', **tkw)
#
# lines = [p1, p2, p3]

# host.legend(lines, [l.get_label() for l in lines])

plt.show()
