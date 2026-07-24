# %% Import Modules

import sys
import os
import pickle
import numpy as np
import pandas as pd
import plotnine as p9
from pandas.api.types import CategoricalDtype





# %% Import Data

dpath = os.path.abspath(".")

#load data
data_ms = pd.read_csv(os.path.join(dpath, "SEFL17_MildStressor.csv"))
data_ms.Animal = data_ms.Animal.astype('category')
data_vf = pd.read_csv(os.path.join(dpath, 'SEFL17_vfData.csv'))
data_vf.Animal = data_vf.Animal.astype('category')
groups = pd.read_csv(os.path.join(dpath, "SEFL17_Groups.csv")).astype('category')

#add group info to data frames
data_ms = pd.merge(
    left = groups,
    right = data_ms,
    how = "right",
    on = ["Animal"])
data_vf = pd.merge(
    left = groups,
    right = data_vf,
    how = "right",
    on = ["Animal"])

#remove no stim groups and drop categories
data_ms = data_ms[data_ms.Group!='No Stim']
data_vf = data_vf[data_vf.Group!='No Stim']
data_ms.Group = data_ms.Group.astype('object').astype('category')
data_ms.Animal = data_ms.Animal.astype('object').astype('category')
data_vf.Group = data_vf.Group.astype('object').astype('category')
data_vf.Animal = data_vf.Animal.astype('object').astype('category')


# %% Organize and Subset Behavioral Data



#
# Mild Stressor data
#

#subset mild stressor based upon based upon bins looking at hour long session
#by minute (data_ms_min) and data surrounding noise in 200 ms bins (data_ms_bin)
data_ms_stl = data_ms[data_ms.phase=='noise']
data_ms_stl.t = data_ms_stl.t+.2
data_ms_min = data_ms[data_ms.phase=='min']
#further bin minute data
data_ms_min['minbin'] = 0
bin = 3
for minlow in np.arange(-30,30,bin):
    data_ms_min.minbin.loc[
        (data_ms_min.t>=minlow) & (data_ms_min.t<minlow+bin)
    ] = minlow
data_ms_min.minbin = data_ms_min.minbin + bin
#aggregate minute data for each subject
data_ms_min = data_ms_min.groupby(
    ['minbin','Animal'], as_index=False).agg({
        'Freezing' : ['mean'],
        'Motion' : ['mean']
    })
data_ms_min.columns = [''.join(ind_names) if len(ind_names[-1])==0 else '_'.join(ind_names) for ind_names in data_ms_min]
data_ms_min = pd.merge(
    left = groups,
    right = data_ms_min,
    how = "right",
    on = ["Animal"])
data_ms_min.Group = data_ms_min.Group.astype('object').astype('category')

#
# Trauma and Trauma Test data
#

trma = data_vf.loc[(data_vf['phase']=='trauma') & (data_vf['subphase']!='shock')].reset_index(drop=True)

trma_shock = data_vf.loc[(data_vf['phase']=='trauma') & (data_vf['subphase']!='postshock')].reset_index(drop=True)
trma_shock = trma_shock.groupby(
    ['Animal','subphase'], as_index=False).agg({
    'freezing': ['mean'],
    'motion': ['mean']})
trma_shock.columns = [''.join(ind_names) if len(ind_names[-1])==0 else '_'.join(ind_names) for ind_names in trma_shock]
trma_shock = pd.merge(
    left = groups,
    right = trma_shock,
    how = "right",
    on = ["Animal"])
trma_shock.Group = trma_shock.Group.astype('object').astype('category')
trma_shock.subphase = trma_shock.subphase.astype('object').astype('category')
trmatest = data_vf.loc[(data_vf['component']=='avg') & (data_vf['phase']=='traumatest')].reset_index(drop=True).copy()


# %% Compute Summary Statistics

trma_stats = trma.groupby(
    ['component','Group'], as_index=False).agg({
    'freezing': ['mean','sem'],
    'motion': ['mean','sem']})
trma_stats.component = pd.Categorical(trma_stats.component)

trma_shk_stats = trma_shock.groupby(
    ['subphase','Group'], as_index=False).agg({
    'freezing_mean': ['mean','sem'],
    'motion_mean': ['mean','sem']})

trmatest_stats = trmatest.groupby(
    ['Group'], as_index=False).agg({
    'freezing': ['mean','sem'],
    'motion': ['mean','sem']})

ms_stl_stats = data_ms_stl.groupby(
    ['t','Group'], as_index=False).agg({
        'Freezing':['mean','sem'],
        'Motion':['mean','sem']})

ms_min_stats = data_ms_min.groupby(
    ['minbin','Group'], as_index=False).agg({
        'Freezing_mean':['mean','sem'],
        'Motion_mean':['mean','sem']})


for df in [trma_stats,trma_shk_stats,trmatest_stats,ms_stl_stats,ms_min_stats]:
        df.columns = [''.join(ind_names) if len(ind_names[-1])==0 else '_'.join(ind_names) for ind_names in df]


# %% Define Overall Plot Parameters

plt = {
    'ybreaks' : [0,20,40,60,80,100],
    'ylims' : [0,80],
    'error_w' : .2,
    'error_sz' : .8,
    'bar_w' : .8,
    'bar_a' : .8,
    'bar_clr' : 'black',
    'point_a' : .5,
    'dotsize' : 3,
    'point_sz' : 5,
    'text_font' : "Arial",
    'colors' : ['dimgray','darkred','steelblue','seagreen'],
    'save' : True
}




# %% Mild Stressor Startle

plt['ylims'] = [0,3000]
plt['ybreaks'] = np.arange(0,2501,500)

plot_ms_stl = p9.ggplot(
    data=ms_stl_stats.loc[(ms_stl_stats.t>=-3) & (ms_stl_stats.t<=6)],
    mapping=p9.aes(x='t', y='Motion_mean',group='Group',fill='Group'))

plot_ms_stl = plot_ms_stl \
+ p9.geom_ribbon(
    mapping = p9.aes(
        ymin = 'Motion_mean - Motion_sem',
        ymax = 'Motion_mean + Motion_sem'),
    color = None, alpha=.75)\
+ p9.scale_x_continuous(breaks=np.arange(-3,7,3), limits=[-3,6], expand=[0,0]) \
+ p9.scale_y_continuous(breaks=plt['ybreaks'],limits=plt['ylims'],expand=[0,0])\
+ p9.geom_vline(mapping=p9.aes(xintercept=0),linetype='dashed') \
+ p9.geom_vline(mapping=p9.aes(xintercept=3),linetype='dashed') \
+ p9.scale_fill_manual(plt['colors']) \
+ p9.labels.ggtitle("Startle Response") + p9.labels.xlab('Sec') + p9.labels.ylab('Motion (au)')\
+ p9.theme_classic() \
+ p9.theme(figure_size = (3,4),
           text = p9.element_text(family = plt['text_font']),
           plot_title = p9.element_text(weight="heavy",margin={'b':10},size=18),
           axis_title = p9.element_text(weight="heavy",size=12),
           axis_text = p9.element_text(size=11,color='black'),
           legend_title = p9.element_blank(),
           legend_background = p9.element_blank(),
           legend_text = p9.element_text(weight="heavy",size=11),
           legend_position = (0.83,.8))

if plt['save'] == True:
     plot_ms_stl.save(filename='SEFL17_stl.tiff',path=dpath,dpi=300)
    
plot_ms_stl





# %% Mild Stressor Freezing Full Session by Minute

plt['ylims'] = [0,100]
plt['ybreaks'] = np.arange(0,101,20)

plot_ms_min_fz = p9.ggplot(
    data=ms_min_stats.loc[
        (ms_min_stats.minbin>=-30)&(ms_min_stats.minbin<=30)], 
    mapping=p9.aes(
        x='minbin', 
        y='Freezing_mean_mean',
        group='Group',fill='Group'))


plot_ms_min_fz = plot_ms_min_fz \
+ p9.geom_ribbon(
    mapping = p9.aes(
        ymin = 'Freezing_mean_mean - Freezing_mean_sem',
        ymax = 'Freezing_mean_mean + Freezing_mean_sem'),
    color = None, alpha=.75)\
+ p9.scale_y_continuous(breaks = plt['ybreaks'], limits=plt['ylims'], expand=[0,0]) \
+ p9.geom_vline(mapping=p9.aes(xintercept=0),linetype='dashed') \
+ p9.scale_fill_manual(plt['colors']) \
+ p9.labels.ggtitle("Mild Stressor") + p9.labels.xlab('Min') + p9.labels.ylab('% Freezing')\
+ p9.theme_classic() \
+ p9.theme(figure_size = (4,4),
           text = p9.element_text(family = plt['text_font']),
           plot_title = p9.element_text(weight="heavy",margin={'b':10},size=18),
           axis_title = p9.element_text(weight="heavy",size=12),
           axis_text = p9.element_text(size=11,color='black'),
           legend_title = p9.element_blank(),
           legend_background = p9.element_blank(),
           legend_position = (0.36,.22))

if plt['save'] == True:
     plot_ms_min.save(filename='SEFL17_ms_fz.tiff',path=dpath,dpi=300)
    
plot_ms_min_fz




# %% Mild Stressor Motion Full Session by Minute

plt['ylims'] = [0,200]
plt['ybreaks'] = np.arange(0,202,50)

plot_ms_min_mt = p9.ggplot(
    data=ms_min_stats.loc[
        (ms_min_stats.minbin>=-30)&(ms_min_stats.minbin<=30)], 
        mapping=p9.aes(
            x='minbin', 
            y='Motion_mean_mean',
            group='Group',fill='Group'))

plot_ms_min_mt = plot_ms_min_mt \
+ p9.geom_ribbon(
    mapping = p9.aes(
        ymin = 'Motion_mean_mean - Motion_mean_sem',
        ymax = 'Motion_mean_mean + Motion_mean_sem'),
    color = None, alpha=.75)\
+ p9.scale_y_continuous(breaks = plt['ybreaks'], limits=plt['ylims'], expand=[0,0]) \
+ p9.geom_vline(mapping=p9.aes(xintercept=0),linetype='dashed') \
+ p9.scale_fill_manual(plt['colors']) \
+ p9.labels.ggtitle("Mild Stressor") + p9.labels.xlab('Min') + p9.labels.ylab('Motion (au)')\
+ p9.theme_classic() \
+ p9.theme(figure_size = (4,4),
           text = p9.element_text(family = plt['text_font']),
           plot_title = p9.element_text(weight="heavy",margin={'b':10},size=18),
           axis_title = p9.element_text(weight="heavy",size=12),
           axis_text = p9.element_text(size=11,color='black'),
           legend_title = p9.element_blank(),
           legend_background = p9.element_blank(),
           legend_position = (0.26,.80))

if plt['save'] == True:
     plot_ms_min.save(filename='SEFL17_ms_mt.tiff',path=dpath,dpi=300)
    
plot_ms_min_mt



# %% Mild Stressor Freezing 3 minutes post, or whatever the bin length is

plt['ylims'] = [0,100]
plt['ybreaks'] = np.arange(0,101,20)

plot_msfreezing = p9.ggplot(data=ms_min_stats[ms_min_stats.minbin==bin], mapping=p9.aes(x='Group', y='Freezing_mean_mean',fill='Group'))

plt['binwidth']=.25
plt['dotsize'] = .03
plt['binwidth'] = 1
plt['scaledot'] = (plt['dotsize'] * max(plt['ylims'])) / plt['binwidth']

plot_msfreezing = plot_msfreezing \
+ p9.geom_bar(stat="identity", 
              position=p9.position_dodge(width=.1),
              width=plt['bar_w'], color=plt['bar_clr'],alpha=plt['bar_a'],
              show_legend=False) \
+ p9.geom_dotplot(
    data=data_ms_min.loc[data_ms_min.minbin==bin],
    stat = p9.stat_bindot(binaxis='y',binwidth=plt['binwidth']),
    mapping = p9.aes(x="Group",y='Freezing_mean'),
    stackdir = 'center', stackratio = .5,
    dotsize = plt['scaledot'],
    fill = None, alpha = plt['point_a'],
    show_legend = False) \
+ p9.geom_errorbar(mapping=p9.aes(
    ymin = 'Freezing_mean_mean - Freezing_mean_sem',
    ymax = 'Freezing_mean_mean + Freezing_mean_sem'),
    width = plt['error_w'], size = plt['error_sz']) \
+ p9.scale_fill_manual(plt['colors']) \
+ p9.scale_y_continuous(breaks = plt['ybreaks'], limits=plt['ylims'], expand=[0,0]) \
+ p9.labels.ggtitle("Post-Noise Freezing") + p9.labels.xlab('') + p9.labels.ylab('% Freezing')\
+ p9.theme_classic() \
+ p9.theme(figure_size = (2,4),
           text = p9.element_text(family = plt['text_font']),
           plot_title= p9.element_text(weight="heavy",margin={'b': 25},size=18),
           axis_title = p9.element_text(weight="heavy",size=12),
           axis_text = p9.element_text(size=11,color='black'),
           axis_text_x = p9.element_text(size=11, weight='heavy',color='black'),
           legend_position = (0.42,.75))

if plt['save'] == True:
    plot_msfreezing.save(filename='SEFL17_ms_freezing5min.tiff',path=dpath,dpi=300)
    
plot_msfreezing



# %% Trauma Shock Reactivity

plt['ylims'] = [0,1200]
plt['ybreaks'] = np.arange(0,1201,200)

plot_trma_shk = p9.ggplot(data=trma_shk_stats, 
                      mapping=p9.aes(x='subphase', y='motion_mean_mean',group='Group',fill='Group'))


plot_trma_shk = plot_trma_shk \
+ p9.geom_line() \
+ p9.geom_errorbar(
    mapping = p9.aes(
        ymin = 'motion_mean_mean - motion_mean_sem',
        ymax = 'motion_mean_mean + motion_mean_sem'),
    width = plt['error_w'], size = plt['error_sz'])\
+ p9.geom_point(size = plt['point_sz'],show_legend=True) \
+ p9.scale_fill_manual(plt['colors']) \
+ p9.scale_y_continuous(breaks = [0,200,400,600,800,1000,1200,1400], limits=[0,1400], expand=[0,0])\
+ p9.labels.ggtitle("Shock Reactivity")\
+ p9.labels.xlab('') + p9.labels.ylab('Motion (Au)')\
+ p9.theme_classic() \
+ p9.theme(figure_size = (2,4),
           text = p9.element_text(family = plt['text_font']),
           plot_title= p9.element_text(weight="heavy",margin={'b': 25},size=18),
           axis_title = p9.element_text(weight="heavy",size=12),
           axis_text = p9.element_text(size=11,color='black'),
           axis_text_x = p9.element_text(),
           legend_title = p9.element_blank(),
           legend_background = p9.element_blank(),
           legend_position = (0.42,.82))

if plt['save'] == True:
    plot_trma_shk.save(filename='SEFL17_trauma_shk_react.tiff',path=dpath,dpi=300)
    
plot_trma_shk





# %% Trauma Test

plt['ylims'] = [0,100]
plt['ybreaks'] = np.arange(0,101,20)

plot_trmatest = p9.ggplot(data=trmatest_stats, mapping=p9.aes(x='Group', y='freezing_mean',fill='Group'))

plt['binwidth']=.25
plt['dotsize'] = .03
plt['binwidth'] = 1
plt['scaledot'] = (plt['dotsize'] * max(plt['ylims'])) / plt['binwidth']

plot_trmatest = plot_trmatest \
+ p9.geom_bar(stat="identity", 
              position=p9.position_dodge(width=.1),
              width=plt['bar_w'], color=plt['bar_clr'],alpha=plt['bar_a'],
              show_legend=False) \
+ p9.geom_dotplot(data=trmatest,
                  stat = p9.stat_bindot(binaxis='y',binwidth=plt['binwidth']),
                  mapping = p9.aes(x="Group",y='freezing'),
                  stackdir = 'center', stackratio = .5,
                  dotsize = plt['scaledot'],
                  fill = None, alpha = plt['point_a'],
                  show_legend = False) \
+ p9.geom_errorbar(mapping=p9.aes(
    ymin = 'freezing_mean - freezing_sem',
    ymax = 'freezing_mean + freezing_sem'),
                   width = plt['error_w'], size = plt['error_sz']) \
+ p9.scale_fill_manual(plt['colors']) \
+ p9.scale_y_continuous(breaks = plt['ybreaks'], limits=plt['ylims'], expand=[0,0]) \
+ p9.labels.ggtitle("Trauma Test") + p9.labels.xlab('') + p9.labels.ylab('% Freezing')\
+ p9.theme_classic() \
+ p9.theme(figure_size = (2,4),
           text = p9.element_text(family = plt['text_font']),
           plot_title= p9.element_text(weight="heavy",margin={'b': 10},size=18),
           axis_title = p9.element_text(weight="heavy",size=12),
           axis_text = p9.element_text(size=11,color='black'),
           legend_position = (0.42,.75))

if plt['save'] == True:
    plot_trmatest.save(filename='SEFL10c_traumatest.tiff',path=dpath,dpi=300)
    
plot_trmatest




# %%
test = data_ms_min[data_ms_min.minbin==bin]
# %%
