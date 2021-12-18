#!/usr/bin/env python
# coding: utf-8
# [Coded: 03/11/2019] [Update-1: 24/02/2020] [Update-1: 11/07/2020] [Update-3:]

# In[5]:


import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd
import statistics 
from scipy.stats import variation 
import seaborn as sns


# In[6]:


pwd


# In[11]:


a = pd.read_csv('Datasets/DRY/Dry_all_bands.csv')
a


# In[4]:


del a['Name']
del a['Chave']


# In[5]:


from sklearn.preprocessing import StandardScaler

# Separating out the features
x = a.loc[:, :].values
# Separating out the target
a = pd.read_csv('Datasets/DRY/Dry_all_bands.csv')
y = a.loc[:,['Chave']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)


# In[6]:


x.shape


# In[7]:


from sklearn.decomposition import PCA
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4',
                         'principal component 5','principal component 6','principal component 7','principal component 8',
                         'principal component 9','principal component 10',])


# In[8]:


principalDf


# In[9]:


pca.explained_variance_ratio_


# In[10]:


A = pd.DataFrame(principalDf)

## save to xlsx file

filepath = 'Datasets/DRY/Principal_component.xlsx'

#A.to_excel(filepath, index=False)


# In[11]:


df = pd.read_csv('Datasets/DRY/Dry_all_insitu_obs_results_1.csv')


# In[12]:


df


# In[13]:


GAMM = np.zeros([106])


# In[14]:


in_situ = np.array(df['In-situ'])
knn = np.array(df['KNN'])
SVM = np.array(df['SVM'])
ANN = np.array(df['ANN'])
RF = np.array(df['RF'])
#GAMM = np.array(df['GAMM'])
#plot_no = np.array(df['Unnamed: 0'])


# In[15]:


Ensemble = []
coff_knn = np.corrcoef(in_situ,knn)
coff_svm = np.corrcoef(in_situ,SVM)
coff_ann = np.corrcoef(in_situ,ANN)
coff_rf = np.corrcoef(in_situ,RF)
coff_gamm = 0
for i in range(106):
    Ensemble.append((coff_knn*knn[i]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))[0,0]+
                            (coff_svm*SVM[i]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))[0,0]+
                             (coff_ann*ANN[i]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))[0,0]+
                              (coff_rf*RF[i]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))[0,0])


# In[16]:


fig, ax = plt.subplots(figsize=(8, 8))
from scipy.signal import savgol_filter
a = np.linspace(1,106,106)

col = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

ax.plot(a,in_situ,linestyle = '--', label = 'In-situ',color = 'black')
#color = 'tab:orange'
#ax.plot(a, GAMM, linewidth = 1,alpha = 0.2,color = col[0])
#yhat = savgol_filter(GAMM, 11, 3) # window size 51, polynomial order 3
#ax.plot(a,yhat, linestyle='--',label = 'GAMM', linewidth = 1.8,color = col[0])

color = 'darkred'
ax.plot(a, knn, linewidth = 1,alpha = 0.2, color = col[1])
yhat = savgol_filter(knn, 11, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'KNN', linewidth = 1.8,color = col[1])
color = 'magenta'
ax.plot(a, SVM, linewidth = 1,alpha = 0.2,color = col[2])
yhat = savgol_filter(SVM, 11, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'SVM', linewidth = 1.8,color = col[2])
color = 'forestgreen'
ax.plot(a, ANN, linewidth = 1,alpha = 0.2,color = col[3])
yhat = savgol_filter(ANN, 11, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'ANN', linewidth = 1.8,color = col[3])
color = 'royalblue'
ax.plot(a, RF, linewidth = 1,alpha = 0.2,color = col[4])
yhat = savgol_filter(RF, 11, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'RF', linewidth = 1.8,color = col[4])

color = 'black'
#ax.plot(a,np.array(Ensemble), label = 'In-situ',color = color)
yhat = savgol_filter(np.array(Ensemble), 11, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='-',label = 'Ensemble', linewidth = 2.3,color = color)

plt.xlabel('In-situ sampling sites', fontsize = '15')

plt.ylabel('AGB (Mg/ha)', fontsize = '15')
#ax.set_xticks(np.arange(len(plot_no)))
#ax.set_xticklabels(plot_no)
plt.xticks(fontsize = '15',rotation='vertical')

plt.yticks(fontsize = '15')
plt.legend(fontsize = '14')
ax.set_xlim(-1,107)
ax.set_xticklabels([])
#plt.plot([20,20],[0,250],'--',color = 'black', alpha = 0.5)
#plt.plot([48,48],[0,250],'--',color = 'black', alpha = 0.5)
#plt.plot([66,66],[0,250],'--',color = 'black', alpha = 0.5)
#plt.plot([92,92],[0,250],'--',color = 'black', alpha = 0.5)
#plt.text(5,30,'n=20',fontsize = 15)
#plt.text(30,30,'n=28',fontsize = 15)

#plt.text(53,30,'n=18',fontsize = 15)

#plt.text(76,30,'n=26',fontsize = 15)
#plt.text(97,30,'n=14',fontsize = 15)

#plt.savefig("Prediction_wet_all.tif", dpi = 200)


# In[17]:


coff_svm


# In[18]:


anam_KNN = knn - in_situ
anam_SVM = SVM - in_situ
anam_ANN = ANN - in_situ
anam_RF = RF - in_situ
anam_GAMM = GAMM - in_situ
anam_Ensemble = Ensemble - in_situ


# In[19]:


fig, ax = plt.subplots(figsize=(10, 10))
from scipy.signal import savgol_filter
a = np.linspace(1,106,106)
col = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


plt.plot([0,106],[0,0],'--',color = 'black', alpha = 0.7, label = 'In-situ')
color = 'darkred'
ax.plot(a, anam_KNN, linewidth = 1,alpha = 0.2, color = color)
yhat = savgol_filter(anam_KNN, 21, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'KNN', linewidth = 2,color = col[0])
color = 'magenta'
ax.plot(a, anam_SVM, linewidth = 1,alpha = 0.2, color = color)
yhat = savgol_filter(anam_SVM, 21, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'SVM', linewidth = 2,color = col[1])
color = 'forestgreen'
ax.plot(a, anam_ANN, linewidth = 1,alpha = 0.2, color = color)
yhat = savgol_filter(anam_ANN, 21, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'ANN', linewidth = 2,color = col[2])
color = 'royalblue'
ax.plot(a, anam_RF, linewidth = 1,alpha = 0.2, color = color)
yhat = savgol_filter(anam_RF, 21, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'RF', linewidth = 2,color = col[3])

color = 'black'
ax.plot(a, anam_Ensemble, linewidth = 1,alpha = 0.2, color = color)
yhat = savgol_filter(anam_Ensemble, 21, 3) # window size 51, polynomial order 3
ax.plot(a,yhat,label = 'Ensemble', linewidth = 3,color = color)
plt.plot([0,106],[0,0],'--',color = 'black', alpha = 0.5)

plt.xlabel('In-situ sampling sites', fontsize = '15')
plt.ylabel('Difference in AGB prediction (Mg/ha)', fontsize = '15')
#ax.set_xticks(np.arange(len(plot_no)))
#ax.set_xticklabels(plot_no)
plt.xticks(fontsize = '10',rotation='vertical')

plt.yticks(fontsize = '15')
plt.legend(fontsize = '15')
ax.set_xlim(-1,107)
#plt.savefig("Anamoly_Prediction_wet_all.tif", dpi = 200)


# In[20]:


np.linalg.norm(in_situ[0:19]-in_situ[0:19].mean()) / (20)

fig, ax = plt.subplots(figsize=(10, 10))

plt.scatter(np.linspace(1,11,11),np.array(r_knn), marker = '^', s= 150,label = 'KNN', alpha = 0.8)
plt.scatter(np.linspace(1,11,11),np.array(r_svm), marker = 's', s= 150, label = 'SVM', alpha = 0.8)
plt.scatter(np.linspace(1,11,11),np.array(r_ann), marker = 'o', s= 150, label = 'ANN', alpha = 0.8)
plt.scatter(np.linspace(1,11,11),np.array(r_rf), marker = 'h', s= 150, label = 'RF', alpha = 0.8)
plt.scatter(np.linspace(1,11,11),np.array(r_gamm), marker = 'X', s= 150, label = 'GAMM', alpha = 0.8)
plt.plot([0,12],[0,0],'--',color = 'black', alpha = 0.7)
plt.fill_between(np.linspace(0,12,11),np.linspace(0,0,11),np.linspace(-1,-1,11),alpha =0.2, label = 'Neglected values', color = 'grey')
plt.xlabel('Sample sets', fontsize = '15')
plt.ylabel('Correlation coefficient (r)', fontsize = '15')

plt.xticks(fontsize = '10',rotation='vertical')

plt.yticks(fontsize = '15')
plt.legend(fontsize = '15')
ax.set_xticklabels([])

plt.legend()
# In[21]:


np.array(Ensemble).shape


# In[22]:


coff_ann


# In[23]:


np.corrcoef(in_situ[0:10],ANN[0:10])


# In[24]:


fig, ax = plt.subplots(figsize=(8, 8))
data = [in_situ,GAMM,knn,SVM,ANN,RF]
plt.boxplot(data, widths=0.5)
ax.set_xticklabels(['In-situ','GAMM', 'KNN', 'SVM', 'ANN', 'RF'], fontsize = '14')
plt.ylabel('AGB (Mg/ha)', fontsize = 15)
plt.yticks(fontsize = 15)


# In[ ]:





# #### Scatter plot

# In[25]:


from scipy import stats
import numpy as np, statsmodels.api as sm


# In[26]:



print('Adjusted R2')
print(1-(1-np.corrcoef(knn,in_situ)[0,1]**2)*((106-1)/(106-(4+1))),
      1-(1-np.corrcoef(SVM,in_situ)[0,1]**2)*((106-1)/(106-(4+1))),
      1-(1-np.corrcoef(ANN,in_situ)[0,1]**2)*((106-1)/(106-(4+1))),
      1-(1-np.corrcoef(RF,in_situ)[0,1]**2)*((106-1)/(106-(4+1))),
     1-(1-np.corrcoef(in_situ,Ensemble)[0,1]**2)*((106-1)/(106-(4+1))))


# In[27]:


fig, ax = plt.subplots(figsize=(6, 6))

slope, intercept, r_value, p_value, std_err = stats.linregress(in_situ,Ensemble) ##Change
line = slope*in_situ+intercept
plt.plot(in_situ,Ensemble,'o',in_situ,line,color = 'red', alpha=0.6)  ##Change
plt.xlim(0, 250)
plt.ylim(0, 250)
plt.xticks(fontsize = '15')
plt.yticks(fontsize = '15')
plt.plot([0,250],[0,250],'--',color = 'black', alpha = 0.5)
plt.xlabel('Estimated in-situ AGB (Mg/ha)', fontsize = '15')
plt.ylabel('Ensemble estimated AGB (Mg/ha)', fontsize = '15')  ##Change
plt.text(20,220,'Adjusted $R^2=0.16$',fontsize = 15)


# In[28]:


fig, ax = plt.subplots(figsize=(8, 8))
import seaborn
#seaborn.residplot(in_situ, Ensemble)
seaborn.residplot(in_situ, SVM, label='SVM', scatter_kws={"s": 100,"alpha": 0.4} )
seaborn.residplot(in_situ, ANN, label='ANN',scatter_kws={"s": 100,"alpha": 0.4})
seaborn.residplot(in_situ, RF, label='RF', scatter_kws={"s": 100,"alpha": 0.4})
plt.xlabel('In-situ AGB (Mg/ha)', fontsize = '15')
plt.ylabel('Residuals (Mg/ha)', fontsize = '15')
plt.xticks(fontsize = '15')
plt.yticks(fontsize = '15')
plt.legend(fontsize = '13',loc = 'upper left')
plt.text(118,24,'Dry Season',fontsize = 15)
#ax.set_ylim(-45,45)


# In[29]:


p_value


# #### Statistics

# In[30]:


# RMSE
def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))


# In[31]:


# Sample standard Deviation
def var_s(lst,par):
    mean = np.mean(lst)
    sum = 0
    for i in range(len(lst)):
        sum += (lst[i]-mean)**2
    return np.sqrt(sum/((len(lst)-par)))


# In[32]:


# Bias
def bias(pred,est):
    b = []
    for i in range(len(pred)):
        x = (pred[i]-est[i])/est[i]
        b.append(x)
        #print(b)
    return np.mean(b)    


# In[33]:


# CV
def CV(y):
    x = (var_s(y,9)/np.mean(y))*100
    return x
CV(in_situ[20:48])


# In[34]:


print('Mean', np.mean(GAMM))
print('RMSE ',rmse(in_situ,GAMM) )
print('variance ',var_s(GAMM,9) )
print('Bias ', bias(GAMM,in_situ))
print('r ',np.corrcoef(in_situ,GAMM)[0,1])
print('CV ', CV(GAMM))


# In[35]:


print('RMSE ',rmse(in_situ,knn) )
print('variance ',var_s(knn,9) )
print('Bias ', bias(knn,in_situ))
print('r ',np.corrcoef(in_situ,knn)[0,1])
print('CV ', CV(knn))


# In[36]:


print('RMSE ',rmse(in_situ,SVM) )
print('variance ',var_s(SVM,9) )
print('Bias ', bias(SVM,in_situ))
print('r ',np.corrcoef(in_situ,SVM)[0,1])
print('CV ', CV(SVM))


# In[37]:


print('RMSE ',rmse(in_situ,ANN) )
print('variance ',var_s(ANN,9) )
print('Bias ', bias(ANN,in_situ))
print('r ',np.corrcoef(in_situ,ANN)[0,1])
print('CV ', CV(ANN))


# In[38]:


print('RMSE ',rmse(in_situ,RF) )
print('variance ',var_s(RF,9) )
print('Bias ', bias(RF,in_situ))
print('r ',np.corrcoef(in_situ,RF)[0,1])
print('CV ', CV(RF))


# In[39]:


print('RMSE ',rmse(in_situ,Ensemble) )
print('variance ',var_s(Ensemble,9) )
print('Bias ', bias(Ensemble,in_situ))
print('r ',np.corrcoef(in_situ,Ensemble)[0,1])
print('CV ', CV(Ensemble))


# In[ ]:





# In[40]:


GAMM_d = np.array(GAMM)
KNN_d = np.array(knn)
SVM_d = np.array(SVM)
ANN_d = np.array(ANN)
RF_d = np.array(RF)
Ensemble_d = np.array(Ensemble)


# In[41]:


fig, ax = plt.subplots(figsize=(8, 8))

data = [GAMM_d,KNN_d,SVM_d,ANN_d,RF_d, Ensemble_d]
box = ax.boxplot(data, widths=0.5,patch_artist=True)
ax.set_xticklabels(['GAMM', 'KNN', 'SVM', 'ANN', 'RF', 'Ensemble'], fontsize = '14')
plt.ylabel('AGB (Mg/ha)', fontsize = 16)
plt.yticks(fontsize = 15)

colors = ['tan','tan','tan','tan','tan','tan','tan']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

    ax.set_ylim(-10,260)


# In[42]:


import xarray as xr


# In[43]:


E_A = xr.open_dataset('Datasets/DRY/E-A.nc')


# In[44]:


E_R = xr.open_dataset('Datasets/DRY/E-R.nc')


# In[45]:


import matplotlib as mpl
mpl.rcParams.update({'font.size': 15})
hfont = {'fontname':'Arial'}
fig=plt.figure(figsize=(8,3))
ax=fig.add_subplot(111)
vals=[-30,-20,-10,0,10,20,30]
cmap = mpl.colors.ListedColormap(["red",'royalblue','white','white','royalblue',"red"])
#cmap = mpl.colors.ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
norm = mpl.colors.BoundaryNorm(vals, cmap.N)
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                spacing='uniform',
                                orientation='horizontal',
                                extend='neither',
                                ticks=vals)

cb.set_label('', **hfont)
ax.set_position((0.1, 0.45, 0.8, 0.1))


# In[46]:


fig, ax = plt.subplots(figsize=(14, 6.5))

im = E_R.Band1.plot(cmap=plt.cm.get_cmap(cmap, 6), vmax = 30, vmin = -30,cbar_kwargs=dict(orientation='horizontal',
                                                pad=0.1, aspect=30, shrink=0.6))
#plt.bgcolor('black')
ax.set_facecolor('darkgrey')


# In[47]:


fig, ax = plt.subplots(figsize=(14, 6.5))

im = E_A.Band1.plot(cmap=plt.cm.get_cmap(cmap, 6), vmax = 30, vmin = -30,cbar_kwargs=dict(orientation='horizontal',
                                                pad=0.1, aspect=30, shrink=0.6))
#plt.bgcolor('black')
ax.set_facecolor('darkgrey')


# In[48]:


from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(figsize=(14, 6.5))

im = E_R.Band1.plot(cmap=plt.cm.get_cmap('RdBu', 6), vmax = 30, vmin = -30,cbar_kwargs=dict(orientation='horizontal',
                                                pad=0.1, aspect=30, shrink=0.6))
#plt.bgcolor('black')
ax.set_facecolor('darkgrey')

#divider = make_axes_locatable(ax)
#cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
#fig.add_axes(cax)
#fig.colorbar(im, cax=cax, orientation="horizontal")

#plt.savefig("Datasets/DRY/Ensemble-RF.tif", dpi = 300)


# In[49]:


fig, ax = plt.subplots(figsize=(14, 6.5))

E_A.Band1.plot(cmap=plt.cm.get_cmap('RdBu', 6), vmax = 30, vmin = -30,cbar_kwargs=dict(orientation='horizontal',
                                                pad=0.1, aspect=30, shrink=0.6))
#plt.bgcolor('black')
ax.set_facecolor('darkgrey')


#plt.savefig("Datasets/DRY/Ensemble-ANN.tif", dpi = 300)


# In[50]:


plt.scatter(E_A.Band1.values,E_R.Band1.values)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




df_dry = pd.read_csv('Datasets/DRY/Band_1-10.csv')df_dryBaCo_10_1 = (df_dry['Band_10']-df_dry['Band_1'])/(df_dry['Band_10']+df_dry['Band_1'])
BaCo_10_2 = (df_dry['Band_10']-df_dry['Band_2'])/(df_dry['Band_10']+df_dry['Band_2'])
BaCo_10_3 = (df_dry['Band_10']-df_dry['Band_3'])/(df_dry['Band_10']+df_dry['Band_3'])
BaCo_10_4 = (df_dry['Band_10']-df_dry['Band_4'])/(df_dry['Band_10']+df_dry['Band_4'])
BaCo_10_5 = (df_dry['Band_10']-df_dry['Band_5'])/(df_dry['Band_10']+df_dry['Band_5'])
BaCo_10_6 = (df_dry['Band_10']-df_dry['Band_6'])/(df_dry['Band_10']+df_dry['Band_6'])
BaCo_10_7 = (df_dry['Band_10']-df_dry['Band_7'])/(df_dry['Band_10']+df_dry['Band_7'])
BaCo_10_8 = (df_dry['Band_10']-df_dry['Band_8'])/(df_dry['Band_10']+df_dry['Band_8'])
BaCo_10_9 = (df_dry['Band_10']-df_dry['Band_9'])/(df_dry['Band_10']+df_dry['Band_9'])

BaCo_9_1 = (df_dry['Band_9']-df_dry['Band_1'])/(df_dry['Band_9']+df_dry['Band_1'])
BaCo_9_2 = (df_dry['Band_9']-df_dry['Band_2'])/(df_dry['Band_9']+df_dry['Band_2'])
BaCo_9_3 = (df_dry['Band_9']-df_dry['Band_3'])/(df_dry['Band_9']+df_dry['Band_3'])
BaCo_9_4 = (df_dry['Band_9']-df_dry['Band_4'])/(df_dry['Band_9']+df_dry['Band_4'])
BaCo_9_5 = (df_dry['Band_9']-df_dry['Band_5'])/(df_dry['Band_9']+df_dry['Band_5'])
BaCo_9_6 = (df_dry['Band_9']-df_dry['Band_6'])/(df_dry['Band_9']+df_dry['Band_6'])
BaCo_9_7 = (df_dry['Band_9']-df_dry['Band_7'])/(df_dry['Band_9']+df_dry['Band_7'])
BaCo_9_8 = (df_dry['Band_9']-df_dry['Band_8'])/(df_dry['Band_9']+df_dry['Band_8'])

BaCo_8_1 = (df_dry['Band_8']-df_dry['Band_1'])/(df_dry['Band_8']+df_dry['Band_1'])
BaCo_8_2 = (df_dry['Band_8']-df_dry['Band_2'])/(df_dry['Band_8']+df_dry['Band_2'])
BaCo_8_3 = (df_dry['Band_8']-df_dry['Band_3'])/(df_dry['Band_8']+df_dry['Band_3'])
BaCo_8_4 = (df_dry['Band_8']-df_dry['Band_4'])/(df_dry['Band_8']+df_dry['Band_4'])
BaCo_8_5 = (df_dry['Band_8']-df_dry['Band_5'])/(df_dry['Band_8']+df_dry['Band_5'])
BaCo_8_6 = (df_dry['Band_8']-df_dry['Band_6'])/(df_dry['Band_8']+df_dry['Band_6'])
BaCo_8_7 = (df_dry['Band_8']-df_dry['Band_7'])/(df_dry['Band_8']+df_dry['Band_7'])

BaCo_7_1 = (df_dry['Band_7']-df_dry['Band_1'])/(df_dry['Band_7']+df_dry['Band_1'])
BaCo_7_2 = (df_dry['Band_7']-df_dry['Band_2'])/(df_dry['Band_7']+df_dry['Band_2'])
BaCo_7_3 = (df_dry['Band_7']-df_dry['Band_3'])/(df_dry['Band_7']+df_dry['Band_3'])
BaCo_7_4 = (df_dry['Band_7']-df_dry['Band_4'])/(df_dry['Band_7']+df_dry['Band_4'])
BaCo_7_5 = (df_dry['Band_7']-df_dry['Band_5'])/(df_dry['Band_7']+df_dry['Band_5'])
BaCo_7_6 = (df_dry['Band_7']-df_dry['Band_6'])/(df_dry['Band_7']+df_dry['Band_6'])

BaCo_6_1 = (df_dry['Band_6']-df_dry['Band_1'])/(df_dry['Band_6']+df_dry['Band_1'])
BaCo_6_2 = (df_dry['Band_6']-df_dry['Band_2'])/(df_dry['Band_6']+df_dry['Band_2'])
BaCo_6_3 = (df_dry['Band_6']-df_dry['Band_2'])/(df_dry['Band_6']+df_dry['Band_3'])
BaCo_6_4 = (df_dry['Band_6']-df_dry['Band_2'])/(df_dry['Band_6']+df_dry['Band_4'])
BaCo_6_5 = (df_dry['Band_6']-df_dry['Band_2'])/(df_dry['Band_6']+df_dry['Band_5'])

BaCo_5_1 = (df_dry['Band_5']-df_dry['Band_1'])/(df_dry['Band_5']+df_dry['Band_1'])
BaCo_5_2 = (df_dry['Band_5']-df_dry['Band_2'])/(df_dry['Band_5']+df_dry['Band_2'])
BaCo_5_3 = (df_dry['Band_5']-df_dry['Band_3'])/(df_dry['Band_5']+df_dry['Band_3'])
BaCo_5_4 = (df_dry['Band_5']-df_dry['Band_4'])/(df_dry['Band_5']+df_dry['Band_4'])

BaCo_4_1 = (df_dry['Band_4']-df_dry['Band_1'])/(df_dry['Band_4']+df_dry['Band_1'])
BaCo_4_2 = (df_dry['Band_4']-df_dry['Band_2'])/(df_dry['Band_4']+df_dry['Band_2'])
BaCo_4_3 = (df_dry['Band_4']-df_dry['Band_3'])/(df_dry['Band_4']+df_dry['Band_3'])

BaCo_3_1 = (df_dry['Band_3']-df_dry['Band_1'])/(df_dry['Band_3']+df_dry['Band_1'])
BaCo_3_2 = (df_dry['Band_3']-df_dry['Band_2'])/(df_dry['Band_3']+df_dry['Band_2'])

BaCo_2_1 = (df_dry['Band_2']-df_dry['Band_1'])/(df_dry['Band_2']+df_dry['Band_1'])df_dry = pd.concat([df_dry['Band_1'],df_dry['Band_2'],df_dry['Band_3'],df_dry['Band_4'],df_dry['Band_5'],
           df_dry['Band_6'],df_dry['Band_7'],df_dry['Band_8'],df_dry['Band_9'],df_dry['Band_10'],
           BaCo_10_1,BaCo_10_2,BaCo_10_3,BaCo_10_4,BaCo_10_5,BaCo_10_6,BaCo_10_7,BaCo_10_8,BaCo_10_9,
           BaCo_9_1,BaCo_9_2,BaCo_9_3,BaCo_9_4,BaCo_9_5,BaCo_9_6,BaCo_9_7,BaCo_9_8,
           BaCo_8_1,BaCo_8_2,BaCo_8_3,BaCo_8_4,BaCo_8_5,BaCo_8_6,BaCo_8_7,
           BaCo_7_1,BaCo_7_2,BaCo_7_3,BaCo_7_4,BaCo_7_5,BaCo_7_6,
           BaCo_6_1,BaCo_6_2,BaCo_6_3,BaCo_6_4,BaCo_6_5,
           BaCo_5_1,BaCo_5_2,BaCo_5_3,BaCo_5_4,
           BaCo_4_1,BaCo_4_2,BaCo_4_3,
           BaCo_3_1,BaCo_3_2,
           BaCo_2_1,df_dry['Chave']], axis= 1)A = pd.DataFrame(df_dry)

## save to xlsx file

filepath = 'Datasets/DRY/Dry_all_bands.xlsx'

A.to_excel(filepath, index=False)
# # Wet Season

# #### Error bar line plot (Brown 1997 and Chave et al. 2014)

# In[51]:


df = pd.read_csv('Datasets/Band_1-10_Sentinel_2_wet - Copy.csv')


# In[52]:


df


# In[53]:


statistics.stdev(df['Chave'])*.05


# In[54]:


upper_bound = df['Chave']+statistics.stdev(df['Chave'])*.1
trace = df['Chave']
lower_bound = df['Chave']-statistics.stdev(df['Chave'])*.1


# In[55]:


upper_bound1 = df['Brown']+statistics.stdev(df['Brown'])*.1
trace1 = df['Brown']
lower_bound1 = df['Brown']-statistics.stdev(df['Brown'])*.1


# In[56]:


fig, ax = plt.subplots(figsize=(15, 6))
a = np.linspace(1,106,106)
plt.plot(a,trace,'.',label = 'Chave et al. (2014)')
plt.plot(a,trace1,'.',label = 'Brown (1997)')
plt.fill_between(a, upper_bound, lower_bound,
                 alpha=0.3,label = '10% SD: Chave et al. (2014)')
plt.fill_between(a, upper_bound1, lower_bound1,
                 alpha=0.3,label = '10% SD: Brown (1997)')
plt.xlabel('In-situ sampling sites', fontsize = '15')
plt.ylabel('AGB (Mg/ha)', fontsize = '15')
plt.xticks(fontsize = '15')
plt.yticks(fontsize = '15')
plt.legend(fontsize = '15')
plt.savefig("Chave_Brown_samplecompare.tif", dpi = 200)


# In[57]:


fig, ax = plt.subplots(figsize=(6, 6))
data = [trace,trace1]
plt.boxplot(data, widths=0.5,whis=[5, 95])
ax.set_xticklabels(['Chave et al. (2014)', 'Brown (1997)'], fontsize = '14')
plt.ylabel('AGB (Mg/ha)', fontsize = 14)
plt.yticks(fontsize = 14)


# In[58]:


# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=trace, bins=15, color='#0504aa',
                            alpha=0.7, rwidth=1)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('My Very Own Histogram')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


# In[59]:


statistics.stdev(trace)/np.mean(trace)*100


# In[60]:


np.mean(trace)


# In[61]:


(123.71/(100)**(0.4298))*(0.05711)


# In[62]:


statistics.stdev(trace)


# #### AGB (wet) plot with machine learning

# In[63]:


pwd


# In[2]:


df = pd.read_csv('Datasets/Wet_all_insitu_obs.csv')


# In[3]:


df


# In[6]:


df['In-Situ'].max()


# In[66]:


in_situ = np.array(df['In-Situ'])
knn = np.array(df['KNN'])
SVM = np.array(df['SVM'])
ANN = np.array(df['ANN'])
RF = np.array(df['RF'])
GAMM = np.array(df['GAMM'])
#plot_no = np.array(df['Unnamed: 0'])


# In[67]:


r_knn = []
r_svm = []
r_ann = []
r_rf = []
r_gamm = []

Ensemble = []
for i in [0,10,20,30,40,50,60,70,80,90,100]:
    if i == 100:
        print('KNN',np.corrcoef(in_situ[i:i+10],knn[i:i+10])[0,1])
        print('SVM',np.corrcoef(in_situ[i:i+10],SVM[i:i+10])[0,1])
        print('ANN',np.corrcoef(in_situ[i:i+10],ANN[i:i+10])[0,1])
        print('RF',np.corrcoef(in_situ[i:i+10],RF[i:i+10])[0,1])
        print('GAMM',np.corrcoef(in_situ[i:i+10],GAMM[i:i+10])[0,1])
        coff_knn = (np.corrcoef(in_situ[i:i+6],knn[i:i+6])[0,1])
        coff_svm = (np.corrcoef(in_situ[i:i+6],SVM[i:i+6])[0,1])
        coff_ann = (np.corrcoef(in_situ[i:i+6],ANN[i:i+6])[0,1])
        coff_rf = (np.corrcoef(in_situ[i:i+6],RF[i:i+6])[0,1])
        coff_gamm = (np.corrcoef(in_situ[i:i+6],GAMM[i:i+6])[0,1])
        
        if coff_knn < 0.1:
            coff_knn = 0
        if coff_svm < 0.1:
            coff_svm = 0
        if coff_ann < 0.1:
            coff_ann = 0
        if coff_rf < 0.1:
            coff_rf = 0
        if coff_gamm < 0.1:
            coff_gamm = 0
        r_knn.append(coff_knn)
        r_svm.append(coff_svm)
        r_ann.append(coff_ann)
        r_rf.append(coff_rf)
        r_gamm.append(coff_gamm)
        for j in range(i,i+6):
            Ensemble.append((coff_knn*knn[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                            (coff_svm*SVM[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                             (coff_ann*ANN[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                              (coff_rf*RF[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                               (coff_gamm*GAMM[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm)))
        print("*********************************************")
        print('KNN',coff_knn)
        print('SVM',coff_svm)
        print('ANN',coff_ann)
        print('RF',coff_rf)
        print('GAMM',coff_gamm)
        print("----------------------------------------------")

    else:
        print('KNN',np.corrcoef(in_situ[i:i+10],knn[i:i+10])[0,1])
        print('SVM',np.corrcoef(in_situ[i:i+10],SVM[i:i+10])[0,1])
        print('ANN',np.corrcoef(in_situ[i:i+10],ANN[i:i+10])[0,1])
        print('RF',np.corrcoef(in_situ[i:i+10],RF[i:i+10])[0,1])
        print('GAMM',np.corrcoef(in_situ[i:i+10],GAMM[i:i+10])[0,1])
        coff_knn = (np.corrcoef(in_situ[i:i+10],knn[i:i+10])[0,1])
        coff_svm = (np.corrcoef(in_situ[i:i+10],SVM[i:i+10])[0,1])
        coff_ann = (np.corrcoef(in_situ[i:i+10],ANN[i:i+10])[0,1])
        coff_rf = (np.corrcoef(in_situ[i:i+10],RF[i:i+10])[0,1])
        coff_gamm = (np.corrcoef(in_situ[i:i+10],GAMM[i:i+10])[0,1])
        
        if coff_knn < 0.1:
            coff_knn = 0
        if coff_svm < 0.1:
            coff_svm = 0
        if coff_ann < 0.1:
            coff_ann = 0
        if coff_rf < 0.1:
            coff_rf = 0
        if coff_gamm < 0.1:
            coff_gamm = 0
        r_knn.append(coff_knn)
        r_svm.append(coff_svm)
        r_ann.append(coff_ann)
        r_rf.append(coff_rf)
        r_gamm.append(coff_gamm)
        for j in range(i,i+10):
            Ensemble.append((coff_knn*knn[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                            (coff_svm*SVM[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                             (coff_ann*ANN[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                              (coff_rf*RF[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                               (coff_gamm*GAMM[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm)))
        
        print("*********************************************")
        print('KNN',coff_knn)
        print('SVM',coff_svm)
        print('ANN',coff_ann)
        print('RF',coff_rf)
        print('GAMM',coff_gamm)
        print("----------------------------------------------")


# In[68]:


fig, ax = plt.subplots(figsize=(8, 8))
from scipy.signal import savgol_filter
a = np.linspace(1,106,106)

col = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

ax.plot(a,in_situ,linestyle = '--', label = 'In-situ',color = 'black')
color = 'tab:orange'
ax.plot(a, GAMM, linewidth = 1,alpha = 0.2,color = col[0])
yhat = savgol_filter(GAMM, 11, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'GAMM', linewidth = 1.8,color = col[0])

color = 'darkred'
ax.plot(a, knn, linewidth = 1,alpha = 0.2, color = col[1])
yhat = savgol_filter(knn, 11, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'KNN', linewidth = 1.8,color = col[1])
color = 'magenta'
ax.plot(a, SVM, linewidth = 1,alpha = 0.2,color = col[2])
yhat = savgol_filter(SVM, 11, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'SVM', linewidth = 1.8,color = col[2])
color = 'forestgreen'
ax.plot(a, ANN, linewidth = 1,alpha = 0.2,color = col[3])
yhat = savgol_filter(ANN, 11, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'ANN', linewidth = 1.8,color = col[3])
color = 'royalblue'
ax.plot(a, RF, linewidth = 1,alpha = 0.2,color = col[4])
yhat = savgol_filter(RF, 11, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'RF', linewidth = 1.8,color = col[4])

color = 'black'
#ax.plot(a,np.array(Ensemble), label = 'In-situ',color = color)
yhat = savgol_filter(np.array(Ensemble), 11, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='-',label = 'Ensemble', linewidth = 2.3,color = color)

plt.xlabel('In-situ sampling sites', fontsize = '15')

plt.ylabel('AGB (Mg/ha)', fontsize = '15')
#ax.set_xticks(np.arange(len(plot_no)))
#ax.set_xticklabels(plot_no)
plt.xticks(fontsize = '15',rotation='vertical')

plt.yticks(fontsize = '15')
plt.legend(fontsize = '15')
ax.set_xlim(-1,107)
ax.set_xticklabels([])
#plt.plot([20,20],[0,250],'--',color = 'black', alpha = 0.5)
#plt.plot([48,48],[0,250],'--',color = 'black', alpha = 0.5)
#plt.plot([66,66],[0,250],'--',color = 'black', alpha = 0.5)
#plt.plot([92,92],[0,250],'--',color = 'black', alpha = 0.5)
#plt.text(5,30,'n=20',fontsize = 15)
#plt.text(30,30,'n=28',fontsize = 15)

#plt.text(53,30,'n=18',fontsize = 15)

#plt.text(76,30,'n=26',fontsize = 15)
#plt.text(97,30,'n=14',fontsize = 15)

#plt.savefig("Prediction_wet_all.tif", dpi = 200)


# In[69]:


anam_KNN = knn - in_situ
anam_SVM = SVM - in_situ
anam_ANN = ANN - in_situ
anam_RF = RF - in_situ
anam_GAMM = GAMM - in_situ
anam_Ensemble = Ensemble - in_situ


# In[70]:


fig, ax = plt.subplots(figsize=(8, 8))
from scipy.signal import savgol_filter
a = np.linspace(1,106,106)
col = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


plt.plot([0,106],[0,0],'--',color = 'black', alpha = 0.7, label = 'In-situ')
color = 'darkred'
ax.plot(a, anam_KNN, linewidth = 1,alpha = 0.2, color = color)
yhat = savgol_filter(anam_KNN, 21, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'KNN', linewidth = 2,color = col[0])
color = 'magenta'
ax.plot(a, anam_SVM, linewidth = 1,alpha = 0.2, color = color)
yhat = savgol_filter(anam_SVM, 21, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'SVM', linewidth = 2,color = col[1])
color = 'forestgreen'
ax.plot(a, anam_ANN, linewidth = 1,alpha = 0.2, color = color)
yhat = savgol_filter(anam_ANN, 21, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'ANN', linewidth = 2,color = col[2])
color = 'royalblue'
ax.plot(a, anam_RF, linewidth = 1,alpha = 0.2, color = color)
yhat = savgol_filter(anam_RF, 21, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'RF', linewidth = 2,color = col[3])
color = 'tab:orange'
ax.plot(a, anam_GAMM, linewidth = 1,alpha = 0.2, color = color)
yhat = savgol_filter(anam_GAMM, 21, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'GAMM', linewidth = 2,color = col[4])
plt.plot([0,106],[0,0],'--',color = 'black', alpha = 0.5)
color = 'black'
ax.plot(a, anam_Ensemble, linewidth = 1,alpha = 0.2, color = color)
yhat = savgol_filter(anam_Ensemble, 21, 3) # window size 51, polynomial order 3
ax.plot(a,yhat,label = 'Ensemble', linewidth = 3,color = color)
plt.plot([0,106],[0,0],'--',color = 'black', alpha = 0.5)

plt.xlabel('In-situ sampling sites', fontsize = '15')
plt.ylabel('Difference in AGB prediction (Mg/ha)', fontsize = '15')
#ax.set_xticks(np.arange(len(plot_no)))
#ax.set_xticklabels(plot_no)
plt.xticks(fontsize = '10',rotation='vertical')

plt.yticks(fontsize = '15')
plt.legend(fontsize = '15')
ax.set_xlim(-1,107)
#plt.savefig("Anamoly_Prediction_wet_all.tif", dpi = 200)


# In[71]:


fig, ax = plt.subplots(figsize=(10, 10))

plt.scatter(np.linspace(1,11,11),np.array(r_knn), marker = '^', s= 150,label = 'KNN', alpha = 0.8)
plt.scatter(np.linspace(1,11,11),np.array(r_svm), marker = 's', s= 150, label = 'SVM', alpha = 0.8)
plt.scatter(np.linspace(1,11,11),np.array(r_ann), marker = 'o', s= 150, label = 'ANN', alpha = 0.8)
plt.scatter(np.linspace(1,11,11),np.array(r_rf), marker = 'h', s= 150, label = 'RF', alpha = 0.8)
plt.scatter(np.linspace(1,11,11),np.array(r_gamm), marker = 'X', s= 150, label = 'GAMM', alpha = 0.8)
plt.plot([0,12],[0,0],'--',color = 'black', alpha = 0.7)
plt.fill_between(np.linspace(0,12,11),np.linspace(0,0,11),np.linspace(-1,-1,11),alpha =0.2, label = 'Neglected values', color = 'grey')
plt.xlabel('Sample sets', fontsize = '15')
plt.ylabel('Correlation coefficient (r)', fontsize = '15')

plt.xticks(fontsize = '10',rotation='vertical')

plt.yticks(fontsize = '15')
plt.legend(fontsize = '15')
ax.set_xticklabels([])

plt.legend()


# In[72]:


fig, ax = plt.subplots(figsize=(10, 10))
from scipy.signal import savgol_filter
a = np.linspace(1,106,106)
ax.plot(a,in_situ, label = 'In-situ',color = 'black')
b = np.linspace(1,100,100)
ax.plot(a,np.array(Ensemble), label = 'In-situ',color = 'black')
yhat = savgol_filter(np.array(Ensemble), 11, 3) # window size 51, polynomial order 3
ax.plot(a,yhat, linestyle='--',label = 'KNN', linewidth = 2,color = color)


# In[73]:


np.array(Ensemble).shape


# In[74]:


coff_ann


# In[75]:


np.corrcoef(in_situ[0:10],ANN[0:10])


# In[76]:


fig, ax = plt.subplots(figsize=(8, 8))
data = [in_situ,GAMM,knn,SVM,ANN,RF,Ensemble]
box = ax.boxplot(data, widths=0.5,patch_artist=True)
ax.set_xticklabels(['In-situ','GAMM', 'KNN', 'SVM', 'ANN', 'RF','Ensemble'], fontsize = '14')
plt.ylabel('AGB (Mg/ha)', fontsize = 15)
plt.yticks(fontsize = 16)

colors = ['lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    
    ax.set_ylim(-10,260)


# #### Scatter plot

# In[77]:


from scipy import stats
import numpy as np, statsmodels.api as sm


# In[78]:



print('Adjusted R2')
print(1-(1-np.corrcoef(knn,in_situ)[0,1]**2)*((106-1)/(106-(10+1))),
      1-(1-np.corrcoef(SVM,in_situ)[0,1]**2)*((106-1)/(106-(10+1))),
      1-(1-np.corrcoef(ANN,in_situ)[0,1]**2)*((106-1)/(106-(10+1))),
      1-(1-np.corrcoef(RF,in_situ)[0,1]**2)*((106-1)/(106-(10+1))),
      1-(1-np.corrcoef(GAMM,in_situ)[0,1]**2)*((106-1)/(106-(10+1))),
      1-(1-np.corrcoef(Ensemble,in_situ)[0,1]**2)*((106-1)/(106-(10+1))))


# In[79]:


np.corrcoef(in_situ,Ensemble)[0,1]**2


# In[80]:


fig, ax = plt.subplots(figsize=(6, 6))

slope, intercept, r_value, p_value, std_err = stats.linregress(in_situ,GAMM) ##Change
line = slope*in_situ+intercept
plt.plot(in_situ,Ensemble,'o',in_situ,line,color = 'red', alpha=0.6)  ##Change
plt.xlim(0, 250)
plt.ylim(0, 250)
plt.xticks(fontsize = '15')
plt.yticks(fontsize = '15')
plt.plot([0,250],[0,250],'--',color = 'black', alpha = 0.5)
plt.xlabel('Estimated in-situ AGB (Mg/ha)', fontsize = '15')
plt.ylabel('Ensemble Estimated AGB (Mg/ha)', fontsize = '15')  ##Change
plt.text(20,220,'Adjusted $R^2=0.87$',fontsize = 15)


# In[81]:


fig, ax = plt.subplots(figsize=(8, 8))
import seaborn
#seaborn.residplot(in_situ, Ensemble)
seaborn.residplot(in_situ, SVM, label='SVM', scatter_kws={"s": 100,"alpha": 0.4} )
seaborn.residplot(in_situ, ANN, label='ANN',scatter_kws={"s": 100,"alpha": 0.4})
seaborn.residplot(in_situ, RF, label='RF', scatter_kws={"s": 100,"alpha": 0.4})
plt.xlabel('In-situ AGB (Mg/ha)', fontsize = '15')
plt.ylabel('Residuals (Mg/ha)', fontsize = '15')
plt.xticks(fontsize = '15')
plt.yticks(fontsize = '15')
plt.legend(fontsize = '13',loc = 'upper left')
plt.text(118,47,'Wet Season',fontsize = 15)
#ax.set_ylim(-45,45)


# In[82]:


p_value


# #### Statistics

# In[83]:


# RMSE
def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))


# In[84]:


# Sample standard Deviation
def var_s(lst,par):
    mean = np.mean(lst)
    sum = 0
    for i in range(len(lst)):
        sum += (lst[i]-mean)**2
    return np.sqrt(sum/((len(lst)-par)))


# In[85]:


# Bias
def bias(pred,est):
    b = []
    for i in range(len(pred)):
        x = (pred[i]-est[i])/est[i]
        b.append(x)
        #print(b)
    return np.mean(b)    


# In[86]:


# CV
def CV(y):
    x = (var_s(y,9)/np.mean(y))*100
    return x
#CV(in_situ[20:48])


# In[87]:


print('Mean', np.mean(GAMM))
print('RMSE ',rmse(in_situ,GAMM) )
print('variance ',var_s(GAMM,9) )
print('Bias ', bias(GAMM,in_situ))
print('r ',np.corrcoef(in_situ,GAMM)[0,1])
print('CV ', CV(GAMM))


# In[88]:


print('RMSE ',rmse(in_situ,knn) )
print('variance ',var_s(knn,9) )
print('Bias ', bias(knn,in_situ))
print('r ',np.corrcoef(in_situ,knn)[0,1])
print('CV ', CV(knn))


# In[89]:


print('RMSE ',rmse(in_situ,SVM) )
print('variance ',var_s(SVM,9) )
print('Bias ', bias(SVM,in_situ))
print('r ',np.corrcoef(in_situ,SVM)[0,1])
print('CV ', CV(SVM))


# In[90]:


print('RMSE ',rmse(in_situ,ANN) )
print('variance ',var_s(ANN,9) )
print('Bias ', bias(ANN,in_situ))
print('r ',np.corrcoef(in_situ,ANN)[0,1])
print('CV ', CV(ANN))


# In[91]:


print('RMSE ',rmse(in_situ,RF) )
print('variance ',var_s(RF,9) )
print('Bias ', bias(RF,in_situ))
print('r ',np.corrcoef(in_situ,RF)[0,1])
print('CV ', CV(RF))


# In[92]:


print('RMSE ',rmse(in_situ,Ensemble) )
print('variance ',var_s(Ensemble,9) )
print('Bias ', bias(Ensemble,in_situ))
print('r ',np.corrcoef(in_situ,Ensemble)[0,1])
print('CV ', CV(Ensemble))


# #### Bland-Altman plots
import pyCompare as pC
pC.blandAltman(in_situ, Ensemble,
            limitOfAgreement=1.96,
            confidenceInterval=95,
            confidenceIntervalMethod='approximate',
            detrend='Linear')
# ### Error bars

# In[93]:


for i in range(len(in_situ)):
    if i<20:
        var_s(in_situ[0:20],1)
    if i>=20 and i<48:
        var_s(in_situ[20:48],1)
    if i>=48 and i<66:
        var_s(in_situ[48:66],1)
    if i>=66 and i<92:
        var_s(in_situ[66:92],1)


# In[94]:


a = (GAMM-in_situ)
upper_bound1 = in_situ + a
trace1 = in_situ
lower_bound1 = in_situ - a


# In[95]:


a = (knn-in_situ)
upper_bound2 = in_situ + a
trace2 = in_situ
lower_bound2 = in_situ - a


# In[96]:


a = (SVM-in_situ)
upper_bound3 = in_situ + a
trace3 = in_situ
lower_bound3 = in_situ - a


# In[97]:


a = (ANN-in_situ)
upper_bound4 = in_situ + a
trace4 = in_situ
lower_bound4 = in_situ - a


# In[98]:


a = (RF-in_situ)
upper_bound5 = in_situ + a
trace5 = in_situ
lower_bound5 = in_situ - a


# In[99]:


a = (Ensemble-in_situ)
upper_bound6 = in_situ + a
trace6 = Ensemble
lower_bound6 = in_situ - a

a = ((GAMM-in_situ)+(knn-in_situ)+(SVM-in_situ)+(ANN-in_situ)+(RF-in_situ))/5
upper_bound = in_situ + a
trace = in_situ
lower_bound = in_situ - a
# In[100]:


fig, ax = plt.subplots(figsize=(10, 10))
a = np.linspace(1,106,106)
plt.plot(a,trace,label = 'In-situ observations', color = 'black')

col = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

plt.fill_between(a, upper_bound1, lower_bound1,
                 alpha=0.2,label = 'Error GAMM',color = col[0])
plt.fill_between(a, upper_bound2, lower_bound2,
                 alpha=0.2,label = 'Error KNN',color = col[1])
plt.fill_between(a, upper_bound3, lower_bound3,
                 alpha=0.2,label = 'Error SVM',color = col[2])
plt.fill_between(a, upper_bound4, lower_bound4,
                 alpha=0.2,label = 'Error ANN',color = col[3])
plt.fill_between(a, upper_bound5, lower_bound5,
                 alpha=0.2,label = 'Error RF',color = col[4])
#plt.plot(a,trace6,label = 'Ensemble estimation')
plt.fill_between(a, upper_bound6, lower_bound6,
                 alpha=0.3,label = 'Error Ensemble',color = 'black')
plt.plot(a, in_situ-13,'--',alpha = 1, label = 'CV = 10% of mean',color = 'y')
plt.plot(a, in_situ+13,'--',alpha = 1, color = 'y')
plt.plot(a, in_situ-26,'--',alpha = 1, label = 'CV = 20% of mean',color = 'red')
plt.plot(a, in_situ+26,'--',alpha = 1, color = 'red')

plt.xlabel('In-situ sampling sites', fontsize = '15')
plt.ylabel('AGB (Mg/ha)', fontsize = '15')
plt.xticks(fontsize = '15')
plt.yticks(fontsize = '15')
plt.legend(fontsize = '13',loc = 'upper left')
ax.set_xticklabels([])
ax.set_xlim(-1,107)
ax.set_ylim(-5,260)
#plt.savefig("Error_bar_Chave_Brown_samplecompare.tif", dpi = 200)


# #### Plots

# In[101]:


df = pd.read_csv('Datasets/KNN_Resultwet.csv')


# In[102]:


df


# In[103]:


r_knn = []
r_svm = []
r_ann = []
r_rf = []
r_gamm = []

Ensemble = []
for i in [0,10,20,30,40,50,60,70,80,90,100]:
    if i == 100:
        print('KNN',np.corrcoef(in_situ[i:i+6],knn[i:i+6])[0,1])
        print('SVM',np.corrcoef(in_situ[i:i+6],SVM[i:i+6])[0,1])
        print('ANN',np.corrcoef(in_situ[i:i+6],ANN[i:i+6])[0,1])
        print('RF',np.corrcoef(in_situ[i:i+6],RF[i:i+6])[0,1])
        print('GAMM',np.corrcoef(in_situ[i:i+6],GAMM[i:i+6])[0,1])
        coff_knn = (np.corrcoef(in_situ[i:i+6],knn[i:i+6])[0,1])
        coff_svm = (np.corrcoef(in_situ[i:i+6],SVM[i:i+6])[0,1])
        coff_ann = (np.corrcoef(in_situ[i:i+6],ANN[i:i+6])[0,1])
        coff_rf = (np.corrcoef(in_situ[i:i+6],RF[i:i+6])[0,1])
        coff_gamm = (np.corrcoef(in_situ[i:i+6],GAMM[i:i+6])[0,1])
        
        if coff_knn < 0.2:
            coff_knn = 0
        if coff_svm < 0.2:
            coff_svm = 0
        if coff_ann < 0.2:
            coff_ann = 0
        if coff_rf < 0.2:
            coff_rf = 0
        if coff_gamm < 0.2:
            coff_gamm = 0
        if coff_knn == 0 and coff_svm == 0 and coff_ann == 0 and coff_rf == 0 and coff_gamm == 0:
            coff_rf = 1
        r_knn.append(coff_knn)
        r_svm.append(coff_svm)
        r_ann.append(coff_ann)
        r_rf.append(coff_rf)
        r_gamm.append(coff_gamm)
        for j in range(i,i+6):
            Ensemble.append((coff_knn*knn[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                            (coff_svm*SVM[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                             (coff_ann*ANN[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                              (coff_rf*RF[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                               (coff_gamm*GAMM[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm)))
        print("*********************************************")
        print('KNN',coff_knn)
        print('SVM',coff_svm)
        print('ANN',coff_ann)
        print('RF',coff_rf)
        print('GAMM',coff_gamm)
        print("----------------------------------------------")

    else:
        print('KNN',np.corrcoef(in_situ[i:i+10],knn[i:i+10])[0,1])
        print('SVM',np.corrcoef(in_situ[i:i+10],SVM[i:i+10])[0,1])
        print('ANN',np.corrcoef(in_situ[i:i+10],ANN[i:i+10])[0,1])
        print('RF',np.corrcoef(in_situ[i:i+10],RF[i:i+10])[0,1])
        print('GAMM',np.corrcoef(in_situ[i:i+10],GAMM[i:i+10])[0,1])
        coff_knn = (np.corrcoef(in_situ[i:i+10],knn[i:i+10])[0,1])
        coff_svm = (np.corrcoef(in_situ[i:i+10],SVM[i:i+10])[0,1])
        coff_ann = (np.corrcoef(in_situ[i:i+10],ANN[i:i+10])[0,1])
        coff_rf = (np.corrcoef(in_situ[i:i+10],RF[i:i+10])[0,1])
        coff_gamm = (np.corrcoef(in_situ[i:i+10],GAMM[i:i+10])[0,1])
        
        if coff_knn < 0.2:
            coff_knn = 0
        if coff_svm < 0.2:
            coff_svm = 0
        if coff_ann < 0.2:
            coff_ann = 0
        if coff_rf < 0.2:
            coff_rf = 0
        if coff_gamm < 0.2:
            coff_gamm = 0
        if coff_knn == 0 and coff_svm == 0 and coff_ann == 0 and coff_rf == 0 and coff_gamm == 0:
            coff_rf = 1
        r_knn.append(coff_knn)
        r_svm.append(coff_svm)
        r_ann.append(coff_ann)
        r_rf.append(coff_rf)
        r_gamm.append(coff_gamm)
        for j in range(i,i+10):
            Ensemble.append((coff_knn*knn[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                            (coff_svm*SVM[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                             (coff_ann*ANN[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                              (coff_rf*RF[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm))+
                               (coff_gamm*GAMM[j]/(coff_ann+coff_gamm+coff_knn+coff_rf+coff_svm)))
        
        print("*********************************************")
        print('KNN',coff_knn)
        print('SVM',coff_svm)
        print('ANN',coff_ann)
        print('RF',coff_rf)
        print('GAMM',coff_gamm)
        print("----------------------------------------------")

Ensemble_test = []

for i in range(len(df['ANN'])):
    if df['ANN'][i]<47:                               #1
        r_coff_knn= r_knn[0]
        r_coff_svm= r_svm[0]
        r_coff_ann= r_ann[0]
        r_coff_rf= r_rf[0]
        r_coff_gamm= r_gamm[0]
    if df['ANN'][i]<62 and df['ANN'][i]>=47:                               #2
        r_coff_knn= r_knn[1]
        r_coff_svm= r_svm[1]
        r_coff_ann= r_ann[1]
        r_coff_rf= r_rf[1]
        r_coff_gamm= r_gamm[1]
    if df['ANN'][i]<84 and df['ANN'][i]>=62:                               #3
        r_coff_knn= r_knn[2]
        r_coff_svm= r_svm[2]
        r_coff_ann= r_ann[2]
        r_coff_rf= r_rf[2]
        r_coff_gamm= r_gamm[2]
    if df['ANN'][i]<95 and df['ANN'][i]>=84:                               #4
        r_coff_knn= r_knn[3]
        r_coff_svm= r_svm[3]
        r_coff_ann= r_ann[3]
        r_coff_rf= r_rf[3]
        r_coff_gamm= r_gamm[3]
    if df['ANN'][i]<113.5 and df['ANN'][i]>=95:                               #5
        r_coff_knn= r_knn[4]
        r_coff_svm= r_svm[4]
        r_coff_ann= r_ann[4]
        r_coff_rf= r_rf[4]
        r_coff_gamm= r_gamm[4]
    if df['ANN'][i]<131.5 and df['ANN'][i]>=113.5:                               #6
        r_coff_knn= r_knn[5]
        r_coff_svm= r_svm[5]
        r_coff_ann= r_ann[5]
        r_coff_rf= r_rf[5]
        r_coff_gamm= r_gamm[5]
    if df['ANN'][i]<154.8 and df['ANN'][i]>=131.5:                               #7
        r_coff_knn= r_knn[6]
        r_coff_svm= r_svm[6]
        r_coff_ann= r_ann[6]
        r_coff_rf= r_rf[6]
        r_coff_gamm= r_gamm[6]
    if df['ANN'][i]<162.2 and df['ANN'][i]>=154.8:                               #8
        r_coff_knn= r_knn[7]
        r_coff_svm= r_svm[7]
        r_coff_ann= r_ann[7]
        r_coff_rf= r_rf[7]
        r_coff_gamm= r_gamm[7]
    if df['ANN'][i]<182.9 and df['ANN'][i]>=162.2:                               #9
        r_coff_knn= r_knn[8]
        r_coff_svm= r_svm[8]
        r_coff_ann= r_ann[8]
        r_coff_rf= r_rf[8]
        r_coff_gamm= r_gamm[8]
    if df['ANN'][i]<225.5 and df['ANN'][i]>=182.9:                    #10
        r_coff_knn= r_knn[9]
        r_coff_svm= r_svm[9]
        r_coff_ann= r_ann[9]
        r_coff_rf= r_rf[9]
        r_coff_gamm= r_gamm[9]
    else:                               #11
        r_coff_knn= r_knn[10]
        r_coff_svm= r_svm[10]
        r_coff_ann= r_ann[10]
        r_coff_rf= r_rf[10]
        r_coff_gamm= r_gamm[10]
    
        
    Ensemble_test.append((r_coff_knn*(df['KNN'][i])/(r_coff_ann+r_coff_gamm+r_coff_knn+r_coff_rf+r_coff_svm))+
                         (r_coff_svm*(df['SVM'][i])/(r_coff_ann+r_coff_gamm+r_coff_knn+r_coff_rf+r_coff_svm))+
                         (r_coff_ann*(df['ANN'][i])/(r_coff_ann+r_coff_gamm+r_coff_knn+r_coff_rf+r_coff_svm))+
                         (r_coff_rf*(df['RF'][i])/(r_coff_ann+r_coff_gamm+r_coff_knn+r_coff_rf+r_coff_svm))+
                         (r_coff_gamm*(df['GAMM'][i])/(r_coff_ann+r_coff_gamm+r_coff_knn+r_coff_rf+r_coff_svm)))
    fig, ax = plt.subplots(figsize=(8, 8))
from collections import Counter

x = df['KNN']
y = Ensemble_test
#ax.scatter(df['KNN'],Ensemble_test, s=10, facecolors='none')
# count the occurrences of each point
c = Counter(zip(x,y))
# create a list of the sizes, here multiplied by 10 for scale
s = [10*c[(x1,y1)] for x1,y1 in zip(x,y)]

plt.scatter(x, y, s=s, alpha = 0.2)
fig, ax = plt.subplots(figsize=(7, 7))
x = np.array(df['ANN'])
y = np.array(Ensemble_test)

#histogram definition
bins = [1000, 1000] # number of bins

# histogram the data
hh, locx, locy = np.histogram2d(x, y, bins=bins)

# Sort the points by density, so that the densest points are plotted last
z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
idx = z.argsort()
x2, y2, z2 = x[idx], y[idx], z[idx]

plt.figure(1,figsize=(8,8)).clf()
s = plt.scatter(x2, y2, c=z2, cmap='jet', marker='.', s =50,vmax = 30, vmin = 0)  
plt.plot([0,250],[0,250],'--',color = 'black', alpha = 0.6)
plt.xlabel('ANN predicted AGB (Mg/ha)', fontsize = '15')
plt.ylabel('Ensemble Estimated AGB (Mg/ha)', fontsize = '15')  ##Change
plt.xticks(fontsize = '14')
plt.yticks(fontsize = '14')
plt.xlim(50,250)
plt.ylim(50,250)
plt.colorbar()fig, ax = plt.subplots(figsize=(7, 7))
x = np.array(df['RF'])
y = np.array(Ensemble_test)

#histogram definition
bins = [1000, 1000] # number of bins

# histogram the data
hh, locx, locy = np.histogram2d(x, y, bins=bins)

# Sort the points by density, so that the densest points are plotted last
z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
idx = z.argsort()
x2, y2, z2 = x[idx], y[idx], z[idx]

plt.figure(1,figsize=(8,8)).clf()
s = plt.scatter(x2, y2, c=z2, cmap='jet', marker='.', s =50,vmax = 30, vmin = 0)  
plt.plot([0,250],[0,250],'--',color = 'black', alpha = 0.6)
plt.xlabel('Estimated in-situ AGB (Mg/ha)', fontsize = '15')
plt.ylabel('Ensemble Estimated AGB (Mg/ha)', fontsize = '15')  ##Change
plt.xticks(fontsize = '14')
plt.yticks(fontsize = '14')
plt.xlim(50,250)
plt.ylim(50,250)
plt.colorbar()np.corrcoef(Ensemble_test, df['ANN'])**2np.corrcoef(Ensemble_test, df['RF'])**2
# #### AS RF classifier

# In[104]:


Ensemble_test = []

for i in range(len(df['RF'])):
    if df['RF'][i]<47:                               #1
        r_coff_knn= r_knn[0]
        r_coff_svm= r_svm[0]
        r_coff_ann= r_ann[0]
        r_coff_rf= r_rf[0]
        r_coff_gamm= r_gamm[0]
    if df['RF'][i]<62 and df['RF'][i]>=47:                               #2
        r_coff_knn= r_knn[1]
        r_coff_svm= r_svm[1]
        r_coff_ann= r_ann[1]
        r_coff_rf= r_rf[1]
        r_coff_gamm= r_gamm[1]
    if df['RF'][i]<84 and df['RF'][i]>=62:                               #3
        r_coff_knn= r_knn[2]
        r_coff_svm= r_svm[2]
        r_coff_ann= r_ann[2]
        r_coff_rf= r_rf[2]
        r_coff_gamm= r_gamm[2]
    if df['RF'][i]<95 and df['RF'][i]>=84:                               #4
        r_coff_knn= r_knn[3]
        r_coff_svm= r_svm[3]
        r_coff_ann= r_ann[3]
        r_coff_rf= r_rf[3]
        r_coff_gamm= r_gamm[3]
    if df['RF'][i]<113.5 and df['RF'][i]>=95:                               #5
        r_coff_knn= r_knn[4]
        r_coff_svm= r_svm[4]
        r_coff_ann= r_ann[4]
        r_coff_rf= r_rf[4]
        r_coff_gamm= r_gamm[4]
    if df['RF'][i]<131.5 and df['RF'][i]>=113.5:                               #6
        r_coff_knn= r_knn[5]
        r_coff_svm= r_svm[5]
        r_coff_ann= r_ann[5]
        r_coff_rf= r_rf[5]
        r_coff_gamm= r_gamm[5]
    if df['RF'][i]<154.8 and df['RF'][i]>=131.5:                               #7
        r_coff_knn= r_knn[6]
        r_coff_svm= r_svm[6]
        r_coff_ann= r_ann[6]
        r_coff_rf= r_rf[6]
        r_coff_gamm= r_gamm[6]
    if df['RF'][i]<162.2 and df['RF'][i]>=154.8:                               #8
        r_coff_knn= r_knn[7]
        r_coff_svm= r_svm[7]
        r_coff_ann= r_ann[7]
        r_coff_rf= r_rf[7]
        r_coff_gamm= r_gamm[7]
    if df['RF'][i]<182.9 and df['RF'][i]>=162.2:                               #9
        r_coff_knn= r_knn[8]
        r_coff_svm= r_svm[8]
        r_coff_ann= r_ann[8]
        r_coff_rf= r_rf[8]
        r_coff_gamm= r_gamm[8]
    if df['RF'][i]<225.5 and df['RF'][i]>=182.9:                    #10
        r_coff_knn= r_knn[9]
        r_coff_svm= r_svm[9]
        r_coff_ann= r_ann[9]
        r_coff_rf= r_rf[9]
        r_coff_gamm= r_gamm[9]
    else:                               #11
        r_coff_knn= r_knn[10]
        r_coff_svm= r_svm[10]
        r_coff_ann= r_ann[10]
        r_coff_rf= r_rf[10]
        r_coff_gamm= r_gamm[10]
    
        
    Ensemble_test.append((r_coff_knn*(df['KNN'][i])/(r_coff_ann+r_coff_gamm+r_coff_knn+r_coff_rf+r_coff_svm))+
                         (r_coff_svm*(df['SVM'][i])/(r_coff_ann+r_coff_gamm+r_coff_knn+r_coff_rf+r_coff_svm))+
                         (r_coff_ann*(df['ANN'][i])/(r_coff_ann+r_coff_gamm+r_coff_knn+r_coff_rf+r_coff_svm))+
                         (r_coff_rf*(df['RF'][i])/(r_coff_ann+r_coff_gamm+r_coff_knn+r_coff_rf+r_coff_svm))+
                         (r_coff_gamm*(df['GAMM'][i])/(r_coff_ann+r_coff_gamm+r_coff_knn+r_coff_rf+r_coff_svm)))
    

fig, ax = plt.subplots(figsize=(7, 7))
x = np.array(df['ANN'])
y = np.array(Ensemble_test)

#histogram definition
bins = [1000, 1000] # number of bins

# histogram the data
hh, locx, locy = np.histogram2d(x, y, bins=bins)

# Sort the points by density, so that the densest points are plotted last
z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
idx = z.argsort()
x2, y2, z2 = x[idx], y[idx], z[idx]

plt.figure(1,figsize=(8,8)).clf()
s = plt.scatter(x2, y2, c=z2, cmap='jet', marker='.', s =50,vmax = 30, vmin = 0)  
plt.plot([0,250],[0,250],'--',color = 'black', alpha = 0.6)
plt.xlabel('ANN predicted AGB (Mg/ha)', fontsize = '15')
plt.ylabel('Ensemble Estimated AGB (Mg/ha)', fontsize = '15')  ##Change
plt.xticks(fontsize = '14')
plt.yticks(fontsize = '14')
plt.xlim(50,250)
plt.ylim(50,250)
plt.text(60,220,'$R^2=0.91$',fontsize = 15)
plt.colorbar()fig, ax = plt.subplots(figsize=(7, 7))
x = np.array(df['RF'])
y = np.array(Ensemble_test)

#histogram definition
bins = [1000, 1000] # number of bins

# histogram the data
hh, locx, locy = np.histogram2d(x, y, bins=bins)

# Sort the points by density, so that the densest points are plotted last
z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
idx = z.argsort()
x2, y2, z2 = x[idx], y[idx], z[idx]

plt.figure(1,figsize=(8,8)).clf()
s = plt.scatter(x2, y2, c=z2, cmap='jet', marker='.', s =50,vmax = 30, vmin = 0)  
plt.plot([0,250],[0,250],'--',color = 'black', alpha = 0.6)
plt.xlabel('RF predicted AGB (Mg/ha)', fontsize = '15')
plt.ylabel('Ensemble Estimated AGB (Mg/ha)', fontsize = '15')  ##Change
plt.xticks(fontsize = '14')
plt.yticks(fontsize = '14')
plt.xlim(50,250)
plt.ylim(50,250)
plt.text(60,220,'$R^2=0.78$',fontsize = 15)
cbar =plt.colorbar()
#cbar.ax.tick_params(labelsize=30) 
# In[105]:


np.corrcoef(Ensemble_test, df['ANN'])**2


# In[106]:


np.corrcoef(Ensemble_test, df['RF'])**2

A = pd.DataFrame(Ensemble_test)

## save to xlsx file

filepath = 'Datasets/Ensemble_test_wet.xlsx'

#A.to_excel(filepath, index=False)
# #### -------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
x = np.array(Ensemble_test -df['ANN'])
y = np.array(Ensemble_test -df['RF'])

#histogram definition
bins = [1000, 1000] # number of bins

# histogram the data
hh, locx, locy = np.histogram2d(x, y, bins=bins)

# Sort the points by density, so that the densest points are plotted last
z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
idx = z.argsort()
x2, y2, z2 = x[idx], y[idx], z[idx]

plt.figure(1,figsize=(8,8)).clf()
s = plt.scatter(x2, y2, c=z2, cmap='jet', marker='.', s =50,vmax = 30, vmin = 0)  
#plt.plot([0,250],[0,250],'--',color = 'black', alpha = 0.6)
plt.xlabel('ANN predicted AGB (Mg/ha)', fontsize = '15')
plt.ylabel('RF predicted AGB (Mg/ha)', fontsize = '15')  ##Change
plt.xticks(fontsize = '14')
plt.yticks(fontsize = '14')
plt.xlim(-40,20)
plt.ylim(-60,70)
#plt.text(60,220,'$R^2=0.78$',fontsize = 15)
cbar =plt.colorbar()
#cbar.ax.tick_params(labelsize=30) fig, ax = plt.subplots(figsize=(7, 6))
y = np.array( df['ANN']-Ensemble_test )
x = np.array(Ensemble_test)

#histogram definition
bins = [1000, 1000] # number of bins

# histogram the data
hh, locx, locy = np.histogram2d(x, y, bins=bins)

# Sort the points by density, so that the densest points are plotted last
z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
idx = z.argsort()
x2, y2, z2 = x[idx], y[idx], z[idx]

plt.figure(1,figsize=(8,8)).clf()
s = plt.scatter(x2, y2, c=z2, cmap='jet', marker='.', s =50,vmax = 30, vmin = 0)  
#plt.plot([-30,30],[-30,30],'--',color = 'black', alpha = 0.6)
plt.xlabel('Ensemble AGB (Mg/ha)', fontsize = '15')
plt.ylabel('ANN-Ensemble AGB (Mg/ha)', fontsize = '15')  ##Change
plt.xticks(fontsize = '14')
plt.yticks(fontsize = '14')
#plt.xlim(50,250)
#plt.ylim(50,250)
#plt.text(60,220,'$R^2=0.78$',fontsize = 15)
cbar =plt.colorbar()
#cbar.ax.tick_params(labelsize=30) 
# In[107]:


fig, ax = plt.subplots(figsize=(7, 6))
y = np.array(df['RF']-Ensemble_test )
x = np.array(Ensemble_test)

#histogram definition
bins = [1000, 1000] # number of bins

# histogram the data
hh, locx, locy = np.histogram2d(x, y, bins=bins)

# Sort the points by density, so that the densest points are plotted last
z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
idx = z.argsort()
x2, y2, z2 = x[idx], y[idx], z[idx]

plt.figure(1,figsize=(8,8)).clf()
s = plt.scatter(x2, y2, c=z2, cmap='jet', marker='.', s =50,vmax = 30, vmin = 0)  
#plt.plot([0,250],[0,250],'--',color = 'black', alpha = 0.6)
plt.xlabel('Ensemble AGB (Mg/ha)', fontsize = '15')
plt.ylabel('RF-Ensemble AGB (Mg/ha)', fontsize = '15')  ##Change
plt.xticks(fontsize = '14')
plt.yticks(fontsize = '14')
#plt.xlim(50,250)
#plt.ylim(50,250)
#plt.text(60,220,'$R^2=0.78$',fontsize = 15)
cbar =plt.colorbar()
#cbar.ax.tick_params(labelsize=30) 


# In[108]:


a = pd.DataFrame(np.array([Ensemble_test,df['RF'],y])).transpose()


# In[109]:


high_unc_1 = a.loc[(a[2] < -20)]
high_unc_2 = a.loc[(a[2] > 20)]

mod_unc_1 = a.loc[(a[2] > -20) & (a[2] < -10)]
mod_unc_2 = a.loc[(a[2] > 10) & (a[2] < 20)]

low_unc_1 = a.loc[(a[2] > -10) & (a[2] < 0)]
low_unc_2 = a.loc[(a[2] < 10) & (a[2] > 0)]

fig, ax = plt.subplots(figsize=(10, 10))

plt.scatter(a[0], a[1], s=a[2]**2,alpha=0.4, edgecolors='w')
# In[110]:


fig, ax = plt.subplots(figsize=(7, 6))
y = np.array(df['RF']-Ensemble_test )
x = np.array(df['RF'])

#histogram definition
bins = [1000, 1000] # number of bins

# histogram the data
hh, locx, locy = np.histogram2d(x, y, bins=bins)

# Sort the points by density, so that the densest points are plotted last
z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
idx = z.argsort()
x2, y2, z2 = x[idx], y[idx], z[idx]

plt.figure(1,figsize=(8,8)).clf()
s = plt.scatter(x2, y2, c=z2, cmap='jet', marker='.', s =50,vmax = 30, vmin = 0)  
#plt.plot([0,250],[0,250],'--',color = 'black', alpha = 0.6)
plt.xlabel('RF predicted AGB (Mg/ha)', fontsize = '15')
plt.ylabel('RF-Ensemble AGB (Mg/ha)', fontsize = '15')  ##Change
plt.xticks(fontsize = '14')
plt.yticks(fontsize = '14')
#plt.xlim(50,250)
#plt.ylim(50,250)
#plt.text(60,220,'$R^2=0.78$',fontsize = 15)
cbar =plt.colorbar()
#cbar.ax.tick_params(labelsize=30) 


# In[111]:


fig, ax = plt.subplots(figsize=(7, 6))
y = np.array(df['ANN']-Ensemble_test )
x = np.array(df['ANN'])

#histogram definition
bins = [1000, 1000] # number of bins

# histogram the data
hh, locx, locy = np.histogram2d(x, y, bins=bins)

# Sort the points by density, so that the densest points are plotted last
z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
idx = z.argsort()
x2, y2, z2 = x[idx], y[idx], z[idx]

plt.figure(1,figsize=(8,8)).clf()
s = plt.scatter(x2, y2, c=z2, cmap='jet', marker='.', s =50,vmax = 30, vmin = 0)  
#plt.plot([0,250],[0,250],'--',color = 'black', alpha = 0.6)
plt.xlabel('ANN predicted AGB (Mg/ha)', fontsize = '15')
plt.ylabel('ANN-Ensemble AGB (Mg/ha)', fontsize = '15')  ##Change
plt.xticks(fontsize = '14')
plt.yticks(fontsize = '14')
#plt.xlim(50,250)
#plt.ylim(50,250)
#plt.text(60,220,'$R^2=0.78$',fontsize = 15)
cbar =plt.colorbar()
#cbar.ax.tick_params(labelsize=30) 


# In[ ]:




