# MSCI_2021_Project3
Filling the gaps in spatio-temporal data coverage at high latitudes using state of the art inversion algorithms

Goal: use Bayesian trans-dimensional sampling techniques to retrieve sea level and sea ice information over extended regions and time periods

Data: You will use a series of altimetry radar (and laser) freeboards (CPOM@UCL repository) together with tide gauge data from the high latitudes (i.e. https://catalog-intaros.nersc.no/dataset/tide-gauge-data) 

Tools: You will test the reversible-jump Markov Chain MonteCarlo (RJ-MCMC) method described in https://github.com/rhyshawkins/TransTessellate2D but also some code developed in Earth Sciences

Bibliography: 

- Hawkins, Rhys, et al. "Virtual tide gauges for predicting relative sea level rise." Journal of Geophysical Research: Solid Earth 124.12 (2019): 13367-13391.
- Hawkins, Rhys, et al. "Trans‐dimensional surface reconstruction with different classes of parameterization." Geochemistry, Geophysics, Geosystems 20.1 (2019): 505-529.
- Gregory, William, Isobel R. Lawrence, and Michel Tsamados. "A Bayesian approach towards daily pan-Arctic sea ice freeboard estimates from combined CryoSat-2 and Sentinel-3 satellite observations." The Cryosphere Discussions (2021): 1-22.
- Tarantola, Albert. Inverse problem theory and methods for model parameter estimation. Society for Industrial and Applied Mathematics, 2005.

![image](https://user-images.githubusercontent.com/29431131/135485368-be2941c0-93b0-4874-8401-1d1108b041a1.png)

Schematic representation of laser or radar penetration into the snow from CryoSat-2 Ku-band (13.5 GHz) radar (a), AltiKa Ka-band (35 GHz) radar (b), laser reﬂection from the snow as seen from Operation IceBridge airborne measurements (c), and snow thickness measurement from snow radar on-board OIB (d). Credit: Isobel Lawrence
