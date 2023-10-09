import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Generate synthetic data for the variables
time = np.linspace(0, 24, 100)
variable1 = np.random.uniform(0, 1, size=len(time))
variable2 = np.sin(time) + np.random.normal(0, 0.1, size=len(time))

# Create the initial plot
fig, ax = plt.subplots()
bar = ax.bar([0], [0], width=0.5)  # Energy bar
line, = ax.plot([], [], 'r-')  # Line plot

# Set plot properties
ax.set_xlim(0, 24)
ax.set_ylim(0, 1.2)
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Value')
ax.set_title('Energy Model Animation')

# Animation update function
def update(frame):
    # Update variable values
    bar[0].set_height(variable1[frame])
    line.set_data(time[:frame+1], variable2[:frame+1])

    return bar, line

# Create the animation
animation = FuncAnimation(fig, update, frames=len(time), interval=200)

# Display the animation
plt.show()
