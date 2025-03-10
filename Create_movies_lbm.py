"""
Author: Ashesh Ghosh
Date: 12 Feb 2025
Description: Reads velocity field data from files and creates a movie showing how the velocity field evolves over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import glob
import re
import math

def load_velocity_data(filename):
    """
    Load velocity data from a file.
    Assumes the file has 5 columns: i, j, density, velocity_x, velocity_y.
    Returns:
      X, Y: Meshgrid coordinates (2D arrays)
      Vx, Vy: Velocity components as 2D arrays (with one value per grid cell)
    """
    data = np.loadtxt(filename)
    i_coords = data[:, 0].astype(int)
    j_coords = data[:, 1].astype(int)
    velocity_x = data[:, 3]
    velocity_y = data[:, 4]
    
    # Determine grid dimensions (assuming grid indices start at 0)
    nx = i_coords.max() + 1
    ny = j_coords.max() + 1
    
    Vx = np.zeros((ny, nx))
    Vy = np.zeros((ny, nx))
    for i, j, vx, vy in zip(i_coords, j_coords, velocity_x, velocity_y):
        Vx[j, i] = vx
        Vy[j, i] = vy
    
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    return X, Y, Vx, Vy

# Gather and sort velocity files by time extracted from the filename.
file_pattern = "active_nematic_velocity_*.dat"
files = sorted(glob.glob(file_pattern),
               key=lambda x: int(re.findall(r'active_nematic_velocity_(\d+)\.dat', x)[0]))
time_steps = [int(re.findall(r'active_nematic_velocity_(\d+)\.dat', f)[0]) for f in files]

# Load the first file to set up the grid and initial data.
X, Y, Vx, Vy = load_velocity_data(files[0])
magnitude = np.sqrt(Vx**2 + Vy**2)
ny, nx = magnitude.shape

fig, ax = plt.subplots(figsize=(8,6))
# Use imshow to show the velocity magnitude as a background.
im = ax.imshow(magnitude, origin='lower', cmap='viridis', 
               extent=[0, nx-1, 0, ny-1])
cb = fig.colorbar(im, ax=ax, label='Velocity magnitude')

# Overlay the quiver plot with one arrow per grid cell.
q = ax.quiver(X, Y, Vx, Vy, color='red', angles='xy', 
              scale_units='xy', scale=1)

ax.set_title(f"Velocity Field at time {time_steps[0]}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_aspect('equal')

def update(frame):
    # Load data for the current frame.
    X, Y, Vx, Vy = load_velocity_data(files[frame])
    magnitude = np.sqrt(Vx**2 + Vy**2)
    # Update the background image and quiver arrows.
    im.set_data(magnitude)
    q.set_UVC(Vx, Vy)
    ax.set_title(f"Velocity Field at time {time_steps[frame]}")
    return im, q

anim = FuncAnimation(fig, update, frames=len(files), interval=200, blit=False)

# Save the animation. Try using FFMpegWriter; if ffmpeg is not found, fall back to PillowWriter.
try:
    metadata = dict(title='Velocity Field Evolution', artist='Your Name')
    writer = FFMpegWriter(fps=5, metadata=metadata)
    anim.save("velocity_field.mp4", writer=writer)
    print("Movie saved as velocity_field.mp4")
except FileNotFoundError:
    print("ffmpeg not found, using Pillow writer to save as GIF.")
    writer = PillowWriter(fps=5)
    anim.save("velocity_field.gif", writer=writer)
    print("Movie saved as velocity_field.gif")

plt.show()