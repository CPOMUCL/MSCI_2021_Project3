import numpy as np
import os
from mpl_toolkits.basemap import Basemap
from scipy import stats
import matplotlib as mpl
from scipy.interpolate import griddata
import pickle
import datetime
import glob
import matplotlib.pyplot as plt
import h5py

def save(dic,path):
    max_bytes = 2**31 -1
    bytes_out = pickle.dumps(dic)
    f = open(path,"wb")
    for idx in range(0, len(bytes_out), max_bytes):
        f.write(bytes_out[idx:idx+max_bytes])
    f.close()

def read_and_bin(months,days,grid_res):
    datapath = '/cpdata/SATS/LASER/ICESAT-2/ATL-10/versions/004/'
    bins = int(8e6/(grid_res*1000))
    FB = {}
    x = [] ; y = []
    k = 0
    for month in months:
        date = month.split('/')
        for day in range(days[k]):
            lon = [] ; lat = [] ; fb = []
            if int(date[0]+date[1]+str('%02d'%(day+1))) < 20181229: #yaw flip on 20181228
                gts = ['gt1r','gt2r','gt3r']
            else:
                gts = ['gt1l','gt2l','gt3l']
            files = sorted(glob.glob(datapath+month+'/ATL10-01_'+date[0]+date[1]+str('%02d'%(day+1))+'*.h5'))
            if len(files)>0:
                lon = [] ; lat = [] ; fb = []
                print(date[0]+date[1]+str('%02d'%(day+1)))
                for file in files:
                    data = h5py.File(file,'r')
                    for gt in gts:
                        lon.extend(data[gt]['freeboard_beam_segment']['beam_freeboard']['longitude'][:])
                        lat.extend(data[gt]['freeboard_beam_segment']['beam_freeboard']['latitude'][:])
                        fb.extend(data[gt]['freeboard_beam_segment']['beam_freeboard']['beam_fb_height'][:])
                 #       print(lon[0])
                  #      print(lat[0])
                   #     print(fb[0])
                #lon = np.array(lon) ; lat = np.array(lat) ; fb = np.array(fb)
                #valid = np.where((fb >= -0.37) & (fb <= 0.63) & (~np.isnan(fb)))                            #ID = np.where((~np.isnan(fb)) & (lat>=60) & (fb<10) & (fb>=0))
                #lon = lon[valid] ; lat = lat[valid] ; fb = fb[valid]
                #lon = lon[:] ; lat = lat[:] ; fb = fb[:]
                fb = np.array(fb) ; lon = np.array(lon) ; lat = np.array(lat)
                ID = np.where((~np.isnan(fb)) & (lat>=60) & (fb<10) & (fb>=0))
                lon = lon[ID] ; lat = lat[ID] ; fb = fb[ID]
                x_vec,y_vec = m(lon,lat)
                binned_FB = stats.binned_statistic_2d(x_vec,y_vec,fb,\
                                       statistic=np.nanmean,bins=bins,range=[[0, 8e6], [0, 8e6]])
                xi,yi = np.meshgrid(binned_FB[1],binned_FB[2])
                x.append(xi) ; y.append(yi)
#                print(date[0]+date[1]+str('%02d'%(day+1)),binned_FB[0].T.shape)
                FB[date[0]+date[1]+str('%02d'%(day+1))] = binned_FB[0].T

#scatter
#   plt.clf()
#   plt.close()
#   m.drawparallels(np.arange(60,90,10), linewidth = 0.25, linestyle='solid', zorder=8)
#   m.drawmeridians(np.arange(0.,360.,30.), linewidth = 0.25, zorder=8)
#   m.drawcoastlines(linewidth=0.5)
#   m.hexbin(lon[:], lat[:], C=fb[:], vmin=0, vmax=0.5, gridsize=500, cmap='RdBu_r')
#   m.colorbar(location = "bottom")
#   plt.title("IceSat-2 Scatter Map of Laser Freeboard: Arctic. "+date[0]+date[1]+str('%02d'%(day+1)))
#   plt.savefig('/home/cjn/OI_PolarSnow/freeboard_daily_processed/IS2/scatter_freeboard_map_calc_'+date[0]+date[1]+str('%02d'%(day+1))+'.png', dpi=300)

#pcolormesh
        plt.clf()
        plt.close()
        m.drawparallels(np.arange(60,90,10), linewidth = 0.25, linestyle='solid', zorder=8)
        m.drawmeridians(np.arange(0.,360.,30.), linewidth = 0.25, zorder=8)
        m.drawcoastlines(linewidth=0.5)
        m.pcolormesh(xi, yi, binned_FB[0].T, cmap='RdBu_r', vmin=0.0, vmax=0.5)
        m.colorbar()
        plt.title("IceSat-2 2D Histogram Map of Ice Thickness: Arctic "+date[0]+date[1]+str('%02d'%(day+1)))
        plt.savefig('/home/cjn/OI_PolarSnow/CS2_IS2/IS2/freeboard_daily_processed/pcolormesh_freeboard_map_calc_'+date[0]+date[1]+str('%02d'%(day+1))+'.png', dpi=300)



        k += 1

    save(FB,'/home/cjn/OI_PolarSnow/CS2_IS2/IS2/freeboard_daily_processed/dailyFB_'+str(grid_res)+'km_'+season+'_season.pkl')
    if os.path.exists('/home/cjn/OI_PolarSnow/CS2_IS2/IS2/freeboard_daily_processed/x_'+str(grid_res)+'km.npy')==False:
        np.save('/home/cjn/OI_PolarSnow/CS2_IS2/IS2/freeboard_daily_processed/x_'+str(grid_res)+'km.npy',np.array(x).transpose(1,2,0)[:,:,0])
        np.save('/home/cjn/OI_PolarSnow/CS2_IS2/IS2/freeboard_daily_processed/y_'+str(grid_res)+'km.npy',np.array(y).transpose(1,2,0)[:,:,0])

m = Basemap(projection='npstere',boundinglat=60,lon_0=0, resolution='l',round=True)
grid_res = int(input('specify grid resolution in km:\n'))
season = '2020-2021'
months = ['2020/08','2020/09','2020/10']
days = [31,30,5]

read_and_bin(months,days,grid_res)
