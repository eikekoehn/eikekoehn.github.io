#%%
#author: E. E. Koehn
#date: Aug 9, 2023

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage as spndimage

# %%
timevec = pd.date_range('2001-09-01','2003-02-01')
years = timevec.year
dayofyear = timevec.day_of_year
index = np.arange(len(timevec))

#%% GENERATE THE TEMPERATURE TIME SERIES
sst_mean = 15
seasonal_variability = 8

# Climatology
sst_clim = np.cos(index*2*np.pi/365)*seasonal_variability + sst_mean

# Instantaneous temperature
autocorr = 0.99
sst_anom = np.zeros_like(sst_clim)
for i in range(1,len(sst_anom)):
    sst_anom[i] = sst_anom[i-1]*autocorr + np.random.randn(1)*0.95
sst_inst = (sst_clim + sst_anom)+1
m = 15
sst_inst_smoothed = spndimage.convolve(sst_inst,np.ones(m),mode='reflect')/m

# Thresholds
sst_thresh90 = -1.2*(np.cos(index*2*np.pi/365)-1)+2+sst_clim
sst_thresh10 = 1.2*(np.cos(index*2*np.pi/365)-1)-2+sst_clim
diff_thresh90_clim = sst_thresh90-sst_clim
diff_thresh10_clim = sst_thresh10-sst_clim

# Calculate the continuous intensity index
int_index90 = (sst_inst_smoothed-sst_clim)/(sst_thresh90-sst_clim)
int_index10 = (sst_inst_smoothed-sst_clim)/(sst_thresh10-sst_clim)

#%%
facs = [1,2,3,4]
cmap_fracs = [0.25,0.375,0.625,0.875]
linewidths=[1.5,1,0.5,0.25]
colors = ['#444444','#888888','#AAAAAA','#CCCCCC']
cmap1 = plt.get_cmap('Oranges')
cmap2 = plt.get_cmap('Purples')
fontsize=15

plt.rcParams['font.size']=fontsize
fig,ax = plt.subplots(figsize=(15,7))
# plot the climatology
plt.plot(timevec,sst_clim,color='#222222',linewidth=2)
plt.text(timevec[-1],sst_clim[-1],' clim',ha='left',va='center',color='k')
# plot the thresholds
for fdx,fac in enumerate(facs):
    threshold10 = sst_clim+fac*diff_thresh10_clim
    threshold90 = sst_clim+fac*diff_thresh90_clim
    textcol = colors[fdx]
    plotcol = colors[fdx]
    plt.plot(timevec,threshold10,color=plotcol,linewidth=linewidths[fdx],linestyle='--')
    plt.plot(timevec,threshold90,color=plotcol,linewidth=linewidths[fdx],linestyle='--')
    print(fac)
    plt.text(timevec[-1],threshold90[-1],' clim.+{}'.format(fac)+r'$\epsilon_{p90}$',ha='left',va='center',color=textcol)
    plt.text(timevec[-1],threshold10[-1],' clim.+{}'.format(fac)+r'$\epsilon_{p10}$',ha='left',va='center',color=textcol)
    fac90_exceedance = sst_inst_smoothed > (sst_clim+(fac*diff_thresh90_clim))
    fac90_thresh = sst_clim+(fac*diff_thresh90_clim)
    fac90_thresh[np.logical_not(fac90_exceedance)]=np.NaN
    plt.fill_between(timevec,fac90_thresh,sst_inst_smoothed,color=cmap1(cmap_fracs[fdx]))
    fac10_exceedance = sst_inst_smoothed < (sst_clim+(fac*diff_thresh10_clim))
    fac10_thresh = sst_clim+(fac*diff_thresh10_clim)
    fac10_thresh[np.logical_not(fac10_exceedance)]=np.NaN
    plt.fill_between(timevec,fac10_thresh,sst_inst_smoothed,color=cmap2(cmap_fracs[fdx]))
# add an arrow between p90 and clim
plt.arrow(timevec[0],sst_clim[0],0,diff_thresh90_clim[2],shape='full',width=2,head_width=8,head_length=0.5,length_includes_head=True,color=cmap1(0.6))
plt.arrow(timevec[0],sst_clim[0],0,diff_thresh10_clim[2],shape='full',width=2,head_width=8,head_length=0.5,length_includes_head=True,color=cmap2(0.6))
plt.annotate(r'$\epsilon_{p90}$  ',[timevec[0],sst_clim[0]+diff_thresh90_clim[0]/2],ha='right',va='center',color=cmap1(0.6))
plt.annotate(r'$\epsilon_{p10}$  ',[timevec[0],sst_clim[0]+diff_thresh10_clim[0]/2],ha='right',va='center',color=cmap2(0.6))
ax.text(0.01,0.01,r'$\epsilon_{p90} = $90th percentile - climatology'+'\n'+'$\epsilon_{p10} = $10th percentile - climatology',transform=ax.transAxes,ha='left',va='bottom')


# plot the instantaneous temperature
plt.plot(timevec,sst_inst_smoothed,color='k',linewidth=5,label='temperature')
n=24
plt.annotate('instantaneous\ntemperature',[timevec[n],sst_inst_smoothed[n]],[30,60],textcoords='offset points',arrowprops=dict(facecolor='#888888',edgecolor='#888888', shrink=0.1),fontsize=fontsize+5)
n2 = 300
plt.annotate('MHW\ni.e., marine\nheatwave',[timevec[n2],sst_inst_smoothed[n2]],[-50,30],textcoords='offset points',arrowprops=dict(facecolor='#888888',edgecolor='#888888', shrink=0.05),fontsize=fontsize+10,ha='right')#,va='center')
n3 = 440
plt.annotate('MCS\ni.e., marine\ncoldspell',[timevec[n3],sst_inst_smoothed[n3]],[-60,-60],textcoords='offset points',arrowprops=dict(facecolor='#888888',edgecolor='#888888', shrink=0.05),fontsize=fontsize+10,ha='right',va='center')
# finalize plot
plt.ylabel('Temperature in °C',fontsize=fontsize+5)
plt.xlabel('Time',fontsize=fontsize+5)
ax.spines[['right', 'top']].set_visible(False)
plt.tight_layout()
plot_dir = '../assets/images/'
plt.savefig(plot_dir+'mhw_concept_sketch.png',dpi=300)
plt.show()

# %%
