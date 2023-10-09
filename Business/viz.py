import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

df = pd.read_csv('Data/LithologySummary.csv', index_col='WELL')

zechstein_well_intersections = pd.read_csv('Data/Zechstein_WellIntersection.csv')

lith_names = list(df.columns)

# Set the number of columns for your subplot grid
num_cols = 3

# Get the number of wells (rows in the DataFrame)
num_wells = len(df)

# Calculate the number of rows needed for the subplot grid
num_rows = np.ceil(num_wells / num_cols).astype(int)


indexes = list(range(0, len(lith_names)))
width = 2*np.pi / len(lith_names)
angles = [element * width for element in indexes]

colours = ["#ae1241", "#5ba8f7", "#c6a000", "#0050ae", "#9b54f3", "#ff7d67", "#dbc227", "#008c5c"]

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(lith_names))


# Create a figure
fig = plt.figure(figsize=(20, num_rows * 7), linewidth=10,
                 edgecolor='#393d5c', 
                 facecolor='#25253c')

# Create a grid layout
grid = plt.GridSpec(num_rows, num_cols, wspace=0.5, hspace=0.5)


# Loop over each row in the DataFrame to create the radial bar charts per well
for i, (index, row) in enumerate(df.iterrows()):
    ax = fig.add_subplot(grid[i // num_cols, i % num_cols], projection='polar')
    bars = ax.bar(x=angles, height=row.values, width=width, 
                  edgecolor='white', zorder=2, alpha=0.8, color=colours[i])

    bars_bg = ax.bar(x=angles, height=100, width=width, color='#393d5c',
                     edgecolor='#25253c', zorder=1)
    
    # Set up labels, ticks and grid
    ax.set_title(index, pad=35, fontsize=22, fontweight='bold', color='white')
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.grid(color='#25253c')
    
    #Set up the lithology / category labels to appear at the correct angle
    for angle, height, lith_name in zip(angles, row.values, lith_names):
        rotation_angle = np.degrees(angle)
        if angle < np.pi:
            rotation_angle -= 90
        elif angle == np.pi:
            rotation_angle -= 90
        else:
            rotation_angle += 90
        ax.text(angle, 110, lith_name.upper(), 
                ha='center', va='center', 
                rotation=rotation_angle, rotation_mode='anchor', fontsize=12, 
                fontweight='bold', color='white')

# Add the scatter plot in the last subplot (subplot 9)
ax = fig.add_subplot(grid[num_rows - 1, num_cols - 1], facecolor='#393d5c')
ax.scatter(zechstein_well_intersections['X_LOC'], zechstein_well_intersections['Y_LOC'], c=colours, s=60)
ax.grid(alpha=0.5, color='#25253c')
ax.set_axisbelow(True)

# Set up the labels and ticks for the scatter plot
ax.set_ylabel('NORTHING', fontsize=12, 
                fontweight='bold', color='white')
ax.set_xlabel('EASTING', fontsize=12, 
                fontweight='bold', color='white')

ax.tick_params(axis='both', colors='white')
ax.ticklabel_format(style='plain')
ax.set_title('WELL LOCATIONS', pad=35, fontsize=22, fontweight='bold', color='white')

# Set the outside borders of the scatter plot to white
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white') 
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')

# Add a footnote to the scatter plot explaining missing well
ax.text(0.0, -0.2, 'Well 16/11-1 ST3 does not contain location information', ha='left', va='bottom', fontsize=10,
        color='white', transform=ax.transAxes)

# Set up and display well name labels
labels = []
for i, row in zechstein_well_intersections.iterrows():
    labels.append(ax.text(row['X_LOC'], row['Y_LOC'], row['WELL'], color='white', fontsize=14))

# Use adjust text to ensure text labels do not overlap with each other or the data points
adjust_text(labels, expand_points=(1.2, 1.2), expand_objects=(1.2, 1.2))

# Create a footnote explaining data source

footnote = """
Data Source: 
Bormann, Peter, Aursand, Peder, Dilib, Fahad, Manral, Surrender, & Dischington, Peter. (2020). FORCE 2020 Well well log and lithofacies dataset for
machine learning competition [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4351156

Figure Created By: Andy McDonald
"""

# Display overall infographic title and footnote
plt.suptitle('LITHOLOGY VARIATION WITHIN THE ZECHSTEIN GP.', size=36, fontweight='bold', color='white')
plot_sub_title = """CHARTS OF LITHOLOGY PERCENTAGES ACROSS 8 WELLS FROM THE NORWEGIAN CONTINENTAL SHELF"""

fig.text(0.5, 0.95, plot_sub_title, ha='center', va='top', fontsize=18, color='white', fontweight='bold')
fig.text(0.1, 0.01, footnote, ha='left', va='bottom', fontsize=14, color='white')


plt.show()