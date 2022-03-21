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

'''
CPOM DATA: 
fb_path1 = "../data/CPOM/freeboard_daily_processed/CS2_CPOM/dailyFB_50km_2019-2020_season.pkl"
fb_path2 = "../data/CPOM/freeboard_daily_processed/AK_CPOM/dailyFB_50km_2019-2020_season.pkl"

BRISTOL DATA:
fb_path1="../data/CPOM/freeboard_daily_processed/Bristol_LARM/CS2/freeboard/dailyFB_50km_2019-2020_season.pkl",
fb_path2="../data/CPOM/freeboard_daily_processed/Bristol_LARM/AK/dailyFB_50km_2019-2020_season.pkl",

'''
def main(
    observation_points = 600,
    model_1 = "Gaussian",
    model_2 = "PseudoFranke",
    model_3 = "EastWest",
    model_4 = "EastWest",
    noise = 0.1,
    verbose=False,
    minlat = 0, maxlat = 7950000.0/1000000,
    minlon = 0, maxlon = 7950000.0/1000000,
    parametrization = 2,
    iterations_number = 100000,
    verbosity = 5000,
    independent_chains = 4,
    temperature_levels = 1,
    maximum_temperature = 2.0,
    iterations_between_tempering_attempts = 10,
    skipping = 10000,
    thinning = 2,
    render_map=True,
    render_matrix=False,
    render_models = True,
    render_median = True,
    render_stddev = True,
    render_histogram = True
    ):

    if verbose:
        print("Starting inversion")
    
    '''
    Step 1: Create synthetic dataset
    '''
    # Create the double observation template
    subprocess.run([
        "python", "../../scripts/generatedualtemplatepoints.py",
        "-A", str(observation_points),
        "-B", str(observation_points),
        "-C", str(observation_points),
        "-D", str(observation_points),
        "-o", "synthetic/datatemplate.txt",
        "--xmin", str(minlon), "--xmax", str(maxlon),
        "--ymin", str(minlat), "--ymax", str(maxlat)
    ])

    # Create the observations file
    subprocess.run([
        "./mksynthetic", 
        "-m", str(model_1),
        "-m", str(model_2),
        "-m", str(model_3),
        "-m", str(model_4),
        "-i", "synthetic/datatemplate.txt",
        "-x", str(minlon), "-X", str(maxlon),
        "-y", str(minlat), "-Y", str(maxlat),
        "-o", "synthetic/synthetic_obs.txt",          # Observations with noise
        "-O", "synthetic/synthetic_franke.txt.true",  # Observations without noise
        "-n", str(noise),                              # Standard deviation of independent Gaussian noise 
        "-I", "synthetic/syntheticobs_franke.img",    # Image of the output
        "-W", str(160), "-H", str(160)
    ])

    if render_models:
        imgA = np.loadtxt("synthetic/syntheticobs_franke.img.A")
        imgB = np.loadtxt("synthetic/syntheticobs_franke.img.B")
        imgC = np.loadtxt("synthetic/syntheticobs_franke.img.C")
        imgD = np.loadtxt("synthetic/syntheticobs_franke.img.D")

        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
        
        img = ax[0, 0].imshow(imgA, cmap='seismic', aspect='auto', interpolation='None', origin='lower')
        ax[0, 0].set_title('Synthetic snow depth (m)')
        plt.colorbar(img, ax=ax[0, 0])

        img = ax[0, 1].imshow(imgB, cmap='seismic', aspect='auto', interpolation='None', origin='lower')
        ax[0, 1].set_title('Synthetic ice thickness (m)')
        plt.colorbar(img, ax=ax[0, 1])

        img = ax[1, 0].imshow(imgC, cmap='seismic', aspect='auto', interpolation='None', origin='lower')
        ax[1, 0].set_title('CryoSat-2 Penetration Factor')
        plt.colorbar(img, ax=ax[1, 0])

        img = ax[1, 1].imshow(imgD, cmap='seismic', aspect='auto', interpolation='None', origin='lower')
        ax[1, 1].set_title('AltiKa Penetration Factor')
        plt.colorbar(img, ax=ax[1, 1])




        plt.show()


    '''
    Step 2: Performing inversion
    '''
    # Run inversion
    subprocess.run([
                "mpirun", "-np", str(independent_chains * temperature_levels),
                "./snow_icept", 
                "-i", "synthetic/synthetic_obs.txt", 
                "-o", "results/", 
                "-P", "priors/synthetic/prior_snow.txt",
                "-P", "priors/synthetic/prior_ice.txt", 
                "-M", "priors/synthetic/positionprior_snow.txt", 
                "-M", "priors/synthetic/positionprior_ice.txt",
                "-H", "priors/synthetic/hierarchical_snow.txt", 
                "-H", "priors/synthetic/hierarchical_ice.txt", 
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-A", str(parametrization), "-A", str(parametrization),
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

    file_snow = f"images/snow"
    file_ice = f"images/ice"


    subprocess.run([
                "mpirun", "-np", str(independent_chains),
                "./post_mean_mpi", "-i", 
                "results/ch.dat", "-o", file_snow,
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-s", str(skipping),
                "-t", str(thinning),
                "-A", str(parametrization), "-A", str(parametrization),
                "-W", str(parameter_W), "-H", str(parameter_H),
                "-D", str(file_snow + "_stddev"),
                "-m", str(file_snow + "_median"),
#                "-g", str(file_snow + "_histogram"),
                "-I", str(0)])

    subprocess.run([
                "mpirun", "-np", str(independent_chains),
                "./post_mean_mpi", "-i", 
                "results/ch.dat", "-o", file_ice,
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-s", str(skipping),
                "-t", str(thinning),
                "-A", str(parametrization), "-A", str(parametrization),
                "-W", str(parameter_W), "-H", str(parameter_H),
                "-D", str(file_ice + "_stddev"),
                "-m", str(file_ice + "_median"),
#                "-g", str(file_ice + "_histogram"),
                "-I", str(1)])            
       
    '''
    Step 4: Produce and save plots
    '''
    snow_mat = np.loadtxt(file_snow)
    ice_mat = np.loadtxt(file_ice)

#    snow_mat = mask_observations(cs2_mean, snow_mat)
#    ice_mat = mask_observations(cs2_mean, ice_mat)


    lon = np.linspace(minlon, maxlon, 160)
    lat = np.linspace(minlat, maxlat, 160)
    lon_g, lat_g = np.meshgrid(lon, lat)

    lon_g = np.load("will_lons.npy")[:-1, :-1]
    lat_g = np.load("will_lats.npy")[:-1, :-1]

    extent = [minlon, maxlon, minlat, maxlat]


    if render_matrix:
        fig, ax = plt.subplots(1, 2, figsize=(15, 12))
        
        img = ax[0].imshow(snow_mat, cmap='seismic', aspect='auto', extent=extent, interpolation='None')
        ax[0].set_title('Snow thickness')
        plt.colorbar(img, ax=ax[0])

        img = ax[1].imshow(ice_mat, cmap='seismic', aspect='auto', extent=extent, interpolation='None')
        ax[1].set_title('Ice thickness')
        plt.colorbar(img, ax=ax[1])

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


        plt.show()

    snow_std = np.loadtxt(file_snow + "_stddev")
    ice_std = np.loadtxt(file_ice + "_stddev")

#    snow_std = mask_observations(cs2_mean, snow_std)
#    ice_std = mask_observations(cs2_mean, ice_std)


    if render_stddev:
        fig = plt.figure(figsize=(16, 20))
        ax = fig.add_subplot(221)

        m = Basemap(projection='lcc', resolution=None, lat_0=-90, lon_0=0, lat_1=89.9, lon_1=180, width=1E7, height=0.5E7)
        draw_map(m)
        m.scatter(lon_g, lat_g, latlon=True, alpha=1, s=0.5, c=snow_std, cmap="seismic")
        plt.colorbar(label=r'Snow Thickness Std (m)')

        ax = fig.add_subplot(222)
        m = Basemap(projection='lcc', resolution=None, lat_0=-90, lon_0=0, lat_1=89.9, lon_1=180, width=1E7, height=0.5E7)
        draw_map(m)
        m.scatter(lon_g, lat_g, latlon=True, alpha=1, s=0.5, c=ice_std, cmap="seismic")
        plt.colorbar(label=r'Ice Thickness Std (m)')


        plt.show()


    '''
    snow_hist = np.loadtxt(file_snow + "_histogram")
    ice_hist = np.loadtxt(file_ice + "_histogram")

    snow_hist = mask_observations(cs2_mean, snow_hist)
    ice_hist = mask_observations(cs2_mean, ice_hist)

    if render_histogram:
        fig = plt.figure(figsize=(16, 20))
        ax = fig.add_subplot(221)

        m = Basemap(projection='lcc', resolution=None, lat_0=-90, lon_0=0, lat_1=89.9, lon_1=180, width=1E7, height=0.5E7)
        draw_map(m)
        m.scatter(lon_g, lat_g, latlon=True, alpha=1, s=0.5, c=snow_hist, cmap="seismic")
        plt.colorbar(label=r'Snow Thickness (m)')

        ax = fig.add_subplot(222)
        m = Basemap(projection='lcc', resolution=None, lat_0=-90, lon_0=0, lat_1=89.9, lon_1=180, width=1E7, height=0.5E7)
        draw_map(m)
        m.scatter(lon_g, lat_g, latlon=True, alpha=1, s=0.5, c=ice_hist, cmap="seismic")
        plt.colorbar(label=r'Ice Thickness (m)')


        plt.show()
    '''

    snow_median = np.loadtxt(file_snow + "_median")
    ice_median = np.loadtxt(file_ice + "_median")

#    snow_median = mask_observations(cs2_mean, snow_median)
#    ice_median = mask_observations(cs2_mean, ice_median)


    if render_median:
        fig = plt.figure(figsize=(16, 20))
        ax = fig.add_subplot(221)

        m = Basemap(projection='lcc', resolution=None, lat_0=-90, lon_0=0, lat_1=89.9, lon_1=180, width=1E7, height=0.5E7)
        draw_map(m)
        m.scatter(lon_g, lat_g, latlon=True, alpha=1, s=0.5, c=snow_median, cmap="seismic")
        plt.colorbar(label=r'Snow Thickness Median (m)')

        ax = fig.add_subplot(222)
        m = Basemap(projection='lcc', resolution=None, lat_0=-90, lon_0=0, lat_1=89.9, lon_1=180, width=1E7, height=0.5E7)
        draw_map(m)
        m.scatter(lon_g, lat_g, latlon=True, alpha=1, s=0.5, c=ice_median, cmap="seismic")
        plt.colorbar(label=r'Ice Thickness Median (m)')


        plt.show()

    return snow_mat, ice_mat

if __name__ == "__main__":
    main()
