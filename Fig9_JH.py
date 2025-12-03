#!/usr/bin/python3
## 2-25-2025, revised Fig.9 with a cross section and additional marked points
from pykrige.ok import OrdinaryKriging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.interpolate import RegularGridInterpolator

# Function to calculate the area of contour polygons in square kilometers
def calculate_contour_area(contour):
    total_area = 0
    conversion_factor = 111 ** 2  # (111 km per degree) squared
    for collection in contour.collections:
        for path in collection.get_paths():
            try:
                # Create a polygon from the path and calculate its area
                poly = Polygon(path.vertices)
                # Convert the area from square degrees to square kilometers
                total_area += poly.area * conversion_factor
            except:
                continue
    return total_area

# File paths and data
file_path = 'Houston_ChEv_GWL_mapping_2019-2023_Shallow_Deep.txt'
counties = gpd.read_file('County.shp').to_crs("EPSG:4326")
loop610 = pd.read_csv('Houston_IH610_inner_loop.psxy', delimiter=' ', header=0)
org_df = pd.read_csv(file_path, sep='\t')

# Filter data
df = org_df[(org_df['WellDepth_BLS'] > 70) & (org_df['WellDepth_BLS'] < 700)]
lon = df['dec_long_va'].values
lat = df['dec_lat_va'].values

# Create grids for kriging
grid_lon = np.linspace(df['dec_long_va'].min(), df['dec_long_va'].max() + 0.1, 300)
grid_lat = np.linspace(df['dec_lat_va'].min(), df['dec_lat_va'].max(), 300)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# Set up figure and global settings
fig, axs = plt.subplots(2, 2, figsize=(10, 14), sharex=True, sharey=True)
global_min, global_max = -80, 80
levels = np.linspace(global_min, global_max, 17)

# Function to plot and calculate contour areas
def plot_and_calculate_area(ax, z_data, title):
    cs = ax.contourf(grid_lon, grid_lat, z_data, levels=levels, cmap='coolwarm_r', extend='both')
    ax.scatter(lon, lat, marker='o', c='gray', s=8, edgecolors='gray', linewidths=0.3)

    # Manually create contour lines with blue color and dotted style for specific levels
    contour_levels = [-60, -40, -20]
    contour_lines = []
    for level in contour_levels:
        contour_line = ax.contour(grid_lon, grid_lat, z_data, levels=[level], colors='blue', linewidths=1.0, linestyles='--')
        contour_lines.append(contour_line)
        
    # Annotate the contour lines
    for contour_line in contour_lines:
        ax.clabel(contour_line, inline=True, fontsize=11, fmt='%1.0f m')
    
    ax.set_title(title)
    
    # Calculate areas for each contour level
    area_20 = calculate_contour_area(ax.contour(grid_lon, grid_lat, z_data, levels=[-20], linewidths=0))
    area_40 = calculate_contour_area(ax.contour(grid_lon, grid_lat, z_data, levels=[-40], linewidths=0))
    area_60 = calculate_contour_area(ax.contour(grid_lon, grid_lat, z_data, levels=[-60], linewidths=0))
    
    # Annotate areas in the top-right corner of the plot
    annotation = f'Areas:\n-20 m: {area_20:.0f} km²\n-40 m: {area_40:.0f} km²\n-60 m: {area_60:.0f} km²'
    ax.text(0.95, 0.95, annotation, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', horizontalalignment='right', 
            bbox=dict(facecolor='white', alpha=0.6))

    return cs

# Generate data and plots for each subplot
z_measure, _ = OrdinaryKriging(df['dec_long_va'], df['dec_lat_va'], df['GW_Level_NAVD'], variogram_model='spherical').execute('grid', grid_lon[0, :], grid_lat[:, 0])
cs_measure = plot_and_calculate_area(axs[0, 0], z_measure, '(a) Median GWLs (2019-2023) (m, NAVD88)')

z_shallow, _ = OrdinaryKriging(df['dec_long_va'], df['dec_lat_va'], df['GWL_Shallow'], variogram_model='spherical').execute('grid', grid_lon[0, :], grid_lat[:, 0])
cs_shallow = plot_and_calculate_area(axs[0, 1], z_shallow, '(b) IVHG-Adjusted GWLs at -150 m (NAVD88)')

z_middle, _ = OrdinaryKriging(df['dec_long_va'], df['dec_lat_va'], df['GWL_Middle'], variogram_model='spherical').execute('grid', grid_lon[0, :], grid_lat[:, 0])
cs_middle = plot_and_calculate_area(axs[1, 0], z_middle, '(c) IVHG-Adjusted GWLs at -300 m (NAVD88)')

z_deep, _ = OrdinaryKriging(df['dec_long_va'], df['dec_lat_va'], df['GWL_Deep'], variogram_model='spherical').execute('grid', grid_lon[0, :], grid_lat[:, 0])
cs_deep = plot_and_calculate_area(axs[1, 1], z_deep, '(d) IVHG-Adjusted GWLs at -450 m (NAVD88)')

# (a) Add cross section line and markers on the first subplot (axs[0,0])
# Define cross section endpoints: A(-95.85, 30.05) and D(-94.85, 29.35)
ax_cs = axs[0, 0]
ax_cs.plot([-95.85, -94.85], [30.05, 29.35], 'r--', linewidth=2)  # Cross section line
ax_cs.plot(-95.85, 30.05, 'ro')  # Marker for point A
ax_cs.plot(-94.85, 29.35, 'ro')  # Marker for point D (formerly B)
ax_cs.text(-95.85, 30.05, ' A', fontsize=12, color='black', verticalalignment='bottom', horizontalalignment='right')
ax_cs.text(-94.85, 29.35, ' D', fontsize=12, color='black', verticalalignment='top', horizontalalignment='left')

# Add markers for additional points B and C along the cross section.
# Slightly adjust positions to lie on the cross section.
# Using linear interpolation between A and D:
# For point B, we choose: (-95.6, 29.875)
# For point C, we choose: (-95.3, 29.665)
ax_cs.plot(-95.6, 29.875, 'ro')
ax_cs.plot(-95.3, 29.665, 'ro')
ax_cs.text(-95.6, 29.875, ' B', fontsize=13, color='black', verticalalignment='bottom', horizontalalignment='right')
ax_cs.text(-95.3, 29.665, ' C', fontsize=13, color='black', verticalalignment='top', horizontalalignment='left')

# Add shared elements like county boundaries and colorbar
for ax_row in axs:
    for ax in ax_row:
        counties.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)
        ax.plot(loop610['Longitude'], loop610['Latitude'], color='green')
        ax.set_xlim([-96, -94.6])
        ax.set_ylim([28.95, 30.5])
        ax.text(0.05, 0.97, 'Chicot-Evangeline Aquifer\nTotal wells: 581', transform=ax.transAxes, fontsize=11, 
                verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.6))
        ax.text(0.38, 0.54, 'Houston\n Loop 610', transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', horizontalalignment='left')

# (b) Compute and output groundwater levels along the cross section for all four datasets.
# Define cross section endpoints: A(-95.85, 30.05) to D(-94.85, 29.35)
n_points = 100
cross_lon = np.linspace(-95.85, -94.85, n_points)
cross_lat = np.linspace(30.05, 29.35, n_points)
# Create coordinate pairs (note: interpolators expect (lat, lon))
cross_points = np.column_stack((cross_lat, cross_lon))

# Create interpolators for each groundwater level dataset (grid order: (lat, lon))
interp_measure = RegularGridInterpolator((grid_lat[:, 0], grid_lon[0, :]), z_measure)
interp_shallow = RegularGridInterpolator((grid_lat[:, 0], grid_lon[0, :]), z_shallow)
interp_middle = RegularGridInterpolator((grid_lat[:, 0], grid_lon[0, :]), z_middle)
interp_deep = RegularGridInterpolator((grid_lat[:, 0], grid_lon[0, :]), z_deep)

# Evaluate interpolators along the cross section
cross_measure = interp_measure(cross_points)
cross_shallow = interp_shallow(cross_points)
cross_middle = interp_middle(cross_points)
cross_deep = interp_deep(cross_points)

# Save the cross section data to a file for later plotting
df_cross = pd.DataFrame({
    'Longitude': cross_lon,
    'Latitude': cross_lat,
    'Median_GWL': cross_measure,
    'GWL_Shallow': cross_shallow,
    'GWL_Middle': cross_middle,
    'GWL_Deep': cross_deep
})
df_cross.to_csv('cross_section_gwl.txt', sep='\t', index=False)

# Create a shared colorbar
cbar_ax = fig.add_axes([0.15, 0.515, 0.75, 0.025])
cbar = fig.colorbar(cs_deep, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Groundwater Level (m, NAVD88)', bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
cbar.set_ticks(levels)

# Final adjustments and save the figure
plt.tight_layout()
plt.savefig('Fig9.pdf', dpi=300)
plt.savefig('Fig9.jpg', dpi=300)
plt.savefig('Fig9.png')
plt.show()

