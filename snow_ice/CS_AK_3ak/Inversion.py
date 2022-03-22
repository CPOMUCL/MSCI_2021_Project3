'''
The objective of this program is to provide the entire pipeline from directly consuming CPOM 
data and directly producing plots with the results
'''
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from itertools import chain
from mpl_toolkits.basemap import Basemap
import os



def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')


def main(fb_path1="data/CS2/TXTFILES/201504_CS2fb_1.5lon_x_0.5lat.txt",
         fb_path2="data/AK/TXTFILES/201504_AKfb_1.5lon_x_0.5lat.txt",
         verbose=False,
         minlat = 60, maxlat = 87.5,
         minlon = -180, maxlon = 180,
         render_map=False,
         render_matrix=True):

    if verbose:
        print("Starting inversion")
    
    '''
    Step 1: Data cleaning and adapting to the TransTessellate standard
    '''
    fb1 = np.loadtxt(fb_path1, skiprows=1)
    fb2 = np.loadtxt(fb_path2, skiprows=1)
    
    fb1 = fb1[~np.isnan(fb1).any(axis=1)]
    fb2 = fb2[~np.isnan(fb2).any(axis=1)]

    total_observations = fb1.shape[0] + fb2.shape[0]
    observations_shape = (total_observations, 5)
    observations_matrix = np.zeros(observations_shape)

    # Fill Observations matrix 
    i = 0
    for fb1_row in fb1:
        observations_matrix[i, 0] = fb1_row[0] # Longitude
        observations_matrix[i, 1] = fb1_row[1] # Latitude
        observations_matrix[i, 2] = 0                # Observation Category
        observations_matrix[i, 3] = fb1_row[2] # Freeboard measurement (m)
        observations_matrix[i, 4] = fb1_row[3] # Standard deviation 
        i += 1
    for fb2_row in fb2:
        observations_matrix[i, 0] = fb2_row[0] # Longitude
        observations_matrix[i, 1] = fb2_row[1] # Latitude
        observations_matrix[i, 2] = 1       # Observation Category
        observations_matrix[i, 3] = fb2_row[2] # Freeboard measurement (m)
        observations_matrix[i, 4] = fb2_row[3] # Standard deviation 
        i += 1

    data = pd.DataFrame(observations_matrix, columns=["Longitude", "Latitude", "Type", "Value", "StdDev"])

    # (TODO: Fill empty points, optionally)


    # Select a subset of the data only 
    data_subset = data[
        (data["Latitude"] > minlat) & (data["Latitude"] < maxlat) &
        (data["Longitude"] > minlon) & (data["Longitude"] < maxlon)]
    
    observations_matrix_subset = data_subset.values

    np.savetxt("observations.txt", observations_matrix_subset, '%5.1f %5.1f %d %5.5f %5.5f')

    # Add the total number of observations at the top of the file
    with open('observations.txt', 'r') as original: data = original.read()
    with open('observations.txt', 'w') as modified: modified.write(f"{observations_matrix_subset.shape[0]}\n" + data)

    '''
    Step 2: Performing inversion
    '''
    # Hyperparameters
    # number_of_processes = 1
    parametrization = 1 # 0 for Voronoi, 1 for Delaunay linear, 2 for Delaunay Clough-Tocher
    iterations_number = 50000
    verbosity = 5000

    # Run inversion
    subprocess.run(["./tideshmc", 
                "-i", "observations.txt", 
                "-o", "results/", 
                "-P", "priors/prior_snow_cs2.txt", "-P", "priors/prior_ice_cs2.txt", 
                "-P", "priors/prior_snow_ak.txt", "-P", "priors/prior_ice_ak.txt", 
                "-M", "priors/positionprior_snow_cs2.txt", "-M", "priors/positionprior_ice_cs2.txt",
                "-M", "priors/positionprior_snow_ak.txt", "-M", "priors/positionprior_ice_ak.txt",
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-A", str(parametrization), "-A", str(parametrization),
                "-A", str(parametrization), "-A", str(parametrization),
                "-t", str(iterations_number), 
                "-v", str(verbosity)])

    # Step 3: Compute means 
    parameter_W = 160
    parameter_H = 160

    fb1_filename = os.path.split(fb_path1)[1]
    fb2_filename = os.path.split(fb_path2)[1]

    fb1_file_snow = f"images/means/{fb1_filename}_snow"
    fb1_file_ice = f"images/means/{fb1_filename}_ice"
    fb2_file_snow = f"images/means/{fb2_filename}_snow"
    fb2_file_ice = f"images/means/{fb2_filename}_ice"


    subprocess.run(["./post_mean", "-i", 
                "results/ch.dat", "-o", fb1_file_snow,
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-A", str(parametrization), "-A", str(parametrization),
                "-A", str(parametrization), "-A", str(parametrization),
                "-W", str(parameter_W), "-H", str(parameter_H),
                "-I", str(0)])

    subprocess.run(["./post_mean", "-i", 
                "results/ch.dat", "-o", fb1_file_ice,
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-A", str(parametrization), "-A", str(parametrization),
                "-A", str(parametrization), "-A", str(parametrization),
                "-W", str(parameter_W), "-H", str(parameter_H),
                "-I", str(1)])            

    subprocess.run(["./post_mean", "-i", 
            "results/ch.dat", "-o", fb2_file_snow,
            "-x", str(minlon), "-X", str(maxlon),
            "-y", str(minlat), "-Y", str(maxlat),
            "-A", str(parametrization), "-A", str(parametrization),
            "-A", str(parametrization), "-A", str(parametrization),
            "-W", str(parameter_W), "-H", str(parameter_H),
            "-I", str(2)])            

    subprocess.run(["./post_mean", "-i", 
            "results/ch.dat", "-o", fb2_file_ice,
            "-x", str(minlon), "-X", str(maxlon),
            "-y", str(minlat), "-Y", str(maxlat),
            "-A", str(parametrization), "-A", str(parametrization),
            "-A", str(parametrization), "-A", str(parametrization),
            "-W", str(parameter_W), "-H", str(parameter_H),
            "-I", str(3)])            



    '''
    Step 4: Produce and save plots
    '''
    fb1_file_snow_mat = np.loadtxt(fb1_file_snow)
    fb1_file_ice_mat = np.loadtxt(fb1_file_ice)

    fb2_file_snow_mat = np.loadtxt(fb2_file_snow)
    fb2_file_ice_mat = np.loadtxt(fb2_file_ice)


    lon = np.linspace(minlon, maxlon, 160)
    lat = np.linspace(minlat, maxlat, 160)
    lon_g, lat_g = np.meshgrid(lon, lat)

    extent = [minlon, maxlon, minlat, maxlat]


    if render_matrix:
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
        
        img = ax[0, 0].imshow(fb1_file_snow_mat, cmap='seismic', aspect='auto', extent=extent, origin='lower', interpolation='None')
        ax[0, 0].set_title('CryoSat-2 Snow')
        plt.colorbar(img, ax=ax[0, 0])

        img = ax[0, 1].imshow(fb1_file_ice_mat, cmap='seismic', aspect='auto', extent=extent, origin='lower', interpolation='None')
        ax[0, 1].set_title('CryoSat-2 Ice')
        plt.colorbar(img, ax=ax[0, 1])

        img = ax[1, 0].imshow(fb2_file_snow_mat, cmap='seismic', aspect='auto', extent=extent, origin='lower', interpolation='None')
        ax[1, 0].set_title('AltiKa Snow')
        plt.colorbar(img, ax=ax[1, 0])

        img = ax[1, 1].imshow(fb2_file_ice_mat, cmap='seismic', aspect='auto', extent=extent, origin='lower', interpolation='None')
        ax[1, 1].set_title('AltiKa Ice')
        plt.colorbar(img, ax=ax[1, 1])

        plt.show()


    plt.figure(figsize=(16, 8))
    m = Basemap(projection='lcc', resolution=None, lat_0=-90, lon_0=0, lat_1=89.9, lon_1=180, width=1E7, height=0.5E7)
    draw_map(m)

    m.scatter(lon_g, lat_g, latlon=True, alpha=1, s=0.5, c=fb1_file_snow_mat, cmap="seismic")
    plt.colorbar(label=r'Freeboard thickness (m)')
    plt.title(fb1_filename)
    plt.savefig(f"images/maps/{fb1_filename}.png")

    if render_map:
        plt.show()

    plt.figure(figsize=(16, 8))
    m = Basemap(projection='lcc', resolution=None, lat_0=-90, lon_0=0, lat_1=89.9, lon_1=180, width=1E7, height=0.5E7)
    draw_map(m)

    m.scatter(lon_g, lat_g, latlon=True, alpha=1, s=0.5, c=fb2_file_snow_mat, cmap="seismic")
    plt.colorbar(label=r'Freeboard thickness (m)')
    plt.title(fb2_filename)
    plt.savefig(f"images/maps/{fb2_filename}.png")

    if render_map:
        plt.show()

    return fb1_file_snow_mat, fb2_file_snow_mat

if __name__ == "__main__":
    main()
