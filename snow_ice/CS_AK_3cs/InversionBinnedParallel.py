'''
The objective of this program is to provide the entire pipeline from directly consuming CPOM 
data and directly producing plots with the results. 

This time including support for multi-threading.
'''
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from itertools import chain
from mpl_toolkits.basemap import Basemap
import os
import warnings



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


def mask_observations(observations, inversion):
    new_inversion = np.copy(inversion)
    for i in range(len(observations)):
        for j in range(len(observations[0])):
            if np.isnan(observations[i][j]):
                new_inversion[i][j] = np.nan
    return new_inversion

def main(
    fb_path1="../data/CPOM/freeboard_daily_processed/CS2_CPOM/dailyFB_50km_2013-2014_season.pkl",
    fb_path2="../data/CPOM/freeboard_daily_processed/AK_CPOM/dailyFB_50km_2013-2014_season.pkl",
    verbose=False,
    minlat = 0, maxlat = 7950000.0,
    minlon = 0, maxlon = 7950000.0,
    sliding_window = 14,
    parametrization = 0,
    iterations_number = 100000,
    verbosity = 50000,
    independent_chains = 2,
    temperature_levels = 2,
    maximum_temperature = 5.0,
    iterations_between_tempering_attempts = 10,
    skipping = 50000,
    thinning = 100,
    render_map=True,
    render_matrix=True,
    render_observations=True
         ):

    if verbose:
        print("Starting inversion")
    
    '''
    Step 1: Data cleaning and adapting to the TransTessellate standard
    '''
    fb1 = np.load(fb_path1, allow_pickle=True)
    fb2 = np.load(fb_path2, allow_pickle=True)
    

    fb1_filename = os.path.split(fb_path1)[1]
    fb2_filename = os.path.split(fb_path2)[1]


    grid_x = np.load("WG_x.npy")
    grid_y = np.load("WG_y.npy")

    shape = (sliding_window, 160, 160)
    ak_map = np.empty(shape)
    cs2_map = np.empty(shape)

    for i, date in enumerate(list(fb1.keys())[50:50+sliding_window]):
        cs2_map[i] += fb1[date]
        ak_map[i] += fb2[date]


    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cs2_mean = np.nanmean(cs2_map, axis=0)
        ak_mean = np.nanmean(ak_map, axis=0)


    ak_observations = []
    cs2_observations = []

    shape = list(fb1.values())[0].shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if not np.isnan(cs2_mean[i][j]):
                cs2_observations.append([grid_x[i][j], grid_y[i][j], 0, cs2_mean[i][j], 1.0])                    
            if not np.isnan(ak_mean[i][j]):
                ak_observations.append([grid_x[i][j], grid_y[i][j], 1, ak_mean[i][j], 1.0])


    cs2_observations.extend(ak_observations)
    data = pd.DataFrame(cs2_observations, columns=["Longitude", "Latitude", "Type", "Value", "StdDev"])




    if render_observations:
        fig, ax = plt.subplots(1, 2, figsize=(15, 12))
        
        data[data['Type']==0].plot(kind='scatter', x='Longitude', y='Latitude', c='Value', cmap='seismic', ax=ax[0], title=f'{fb1_filename}')
        data[data['Type']==1].plot(kind='scatter', x='Longitude', y='Latitude', c='Value', cmap='seismic', ax=ax[1], title=f'{fb2_filename}')

        plt.show()

    # (TODO: Fill empty points, optionally)
    #minlat = min(data['Latitude']) * (0.9)
    #maxlat = max(data['Latitude']) * (1.1)
    #minlon = min(data['Longitude']) * (0.9)
    #maxlon = max(data['Longitude']) * (1.1)

    # Select a subset of the data only 
    #data_subset = data[
    #    (data["Latitude"] > minlat) & (data["Latitude"] < maxlat) &
    #    (data["Longitude"] > minlon) & (data["Longitude"] < maxlon)]
    
    observations_matrix_subset = data.values

    np.savetxt("observations.txt", observations_matrix_subset, '%5.1f %5.1f %d %5.5f %5.5f')

    # Add the total number of observations at the top of the file
    with open('observations.txt', 'r') as original: data = original.read()
    with open('observations.txt', 'w') as modified: modified.write(f"{observations_matrix_subset.shape[0]}\n" + data)

    '''
    Step 2: Performing inversion
    '''
    # Hyperparameters
    # number_of_processes = 1
    #parametrization = 0 # 0 for Voronoi, 1 for Delaunay linear, 2 for Delaunay Clough-Tocher
    #iterations_number = 100000
    #verbosity = 50000

    # Run inversion
    subprocess.run([
                "mpirun", "-np", str(independent_chains * temperature_levels),
                "./snow_icept", 
                "-i", "observations.txt", 
                "-o", "results/", 
                "-P", "priors/prior_snow.txt",
                "-P", "priors/prior_ice.txt", 
                "-P", "priors/prior_penetration.txt",
                "-M", "priors/positionprior_snow.txt", 
                "-M", "priors/positionprior_ice.txt",
                "-M", "priors/positionprior_penetration.txt",
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-A", str(parametrization), "-A", str(parametrization), "-A", str(parametrization),
                "-t", str(iterations_number), 
                "-v", str(verbosity),
                "-c", str(independent_chains),    # Independent chains to run at each temperature
                "-K", str(temperature_levels),    # Number of temperature levels for parallel tempering
                "-m", str(maximum_temperature),  # Maximum temperature for the parallel tempering log temperature
                "-e", str(iterations_between_tempering_attempts)    # Number of iterations between parallel tempering exchange attempts
                ])

    # Step 3: Compute means 
    parameter_W = 160
    parameter_H = 160

    file_snow = f"images/means/{fb1_filename}_snow"
    file_ice = f"images/means/{fb1_filename}_ice"
    file_cs2_penetration = f"images/means/{fb1_filename}_cs2_penetration"


    subprocess.run([
                "mpirun", "-np", str(2),
                "./post_mean_mpi", "-i", 
                "results/ch.dat", "-o", file_snow,
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-s", str(skipping),
                "-t", str(thinning),
                "-A", str(parametrization), "-A", str(parametrization),
                "-W", str(parameter_W), "-H", str(parameter_H),
                "-I", str(0)])

    subprocess.run([
                "mpirun", "-np", str(2),
                "./post_mean_mpi", "-i", 
                "results/ch.dat", "-o", file_ice,
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-s", str(skipping),
                "-t", str(thinning),
                "-A", str(parametrization), "-A", str(parametrization),
                "-W", str(parameter_W), "-H", str(parameter_H),
                "-I", str(1)])

    subprocess.run([
                "mpirun", "-np", str(2),
                "./post_mean_mpi", "-i", 
                "results/ch.dat", "-o", file_cs2_penetration,
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-s", str(skipping),
                "-t", str(thinning),
                "-A", str(parametrization), "-A", str(parametrization),
                "-W", str(parameter_W), "-H", str(parameter_H),
                "-I", str(2)])
    



    '''
    Step 4: Produce and save plots
    '''
    snow_mat = np.loadtxt(file_snow)
    ice_mat = np.loadtxt(file_ice)
    cs2_penetration_mat = np.loadtxt(file_cs2_penetration)

    snow_mat = mask_observations(cs2_mean, snow_mat)
    ice_mat = mask_observations(cs2_mean, ice_mat)
    cs2_penetration_mat = mask_observations(ak_mean, cs2_penetration_mat)


    lon = np.linspace(minlon, maxlon, 160)
    lat = np.linspace(minlat, maxlat, 160)
    lon_g, lat_g = np.meshgrid(lon, lat)

    lon_g = np.load("will_lons.npy")[:-1, :-1]
    lat_g = np.load("will_lats.npy")[:-1, :-1]

    extent = [minlon, maxlon, minlat, maxlat]


    if render_matrix:
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
        
        img = ax[0, 0].imshow(snow_mat, cmap='seismic', aspect='auto', extent=extent, interpolation='None')
        ax[0, 0].set_title('Snow thickness')
        plt.colorbar(img, ax=ax[0, 0])

        img = ax[0, 1].imshow(ice_mat, cmap='seismic', aspect='auto', extent=extent, interpolation='None')
        ax[0, 1].set_title('Ice thickness')
        plt.colorbar(img, ax=ax[0, 1])

        img = ax[1, 0].imshow(cs2_penetration_mat, cmap='seismic', aspect='auto', extent=extent, interpolation='None')
        ax[1, 0].set_title('CryoSat-2 penetration factor')
        plt.colorbar(img, ax=ax[1, 0])


        plt.show()

    if render_map:
        fig = plt.figure(figsize=(16, 20))
        ax = fig.add_subplot(221)

        m = Basemap(projection='lcc', resolution=None, lat_0=-90, lon_0=0, lat_1=89.9, lon_1=180, width=1E7, height=0.5E7)
        draw_map(m)
        m.scatter(lon_g, lat_g, latlon=True, alpha=1, s=0.5, c=snow_mat, cmap="seismic")
        plt.colorbar(label=r'Snow Thickness (m)')

        ax = fig.add_subplot(222)
        m = Basemap(projection='lcc', resolution=None, lat_0=-90, lon_0=0, lat_1=89.9, lon_1=180, width=1E7, height=0.5E7)
        draw_map(m)
        m.scatter(lon_g, lat_g, latlon=True, alpha=1, s=0.5, c=ice_mat, cmap="seismic")
        plt.colorbar(label=r'Ice Thickness (m)')

        ax = fig.add_subplot(223)
        m = Basemap(projection='lcc', resolution=None, lat_0=-90, lon_0=0, lat_1=89.9, lon_1=180, width=1E7, height=0.5E7)
        draw_map(m)
        m.scatter(lon_g, lat_g, latlon=True, alpha=1, s=0.5, c=cs2_penetration_mat, cmap="seismic")
        plt.colorbar(label=r'CryoSat-2 Penetration Factor')


        plt.show()


    return snow_mat, ice_mat, cs2_penetration_mat

if __name__ == "__main__":
    main()
