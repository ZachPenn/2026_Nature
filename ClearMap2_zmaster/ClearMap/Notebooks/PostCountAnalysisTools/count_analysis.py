# %% Import modules

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from skimage import io
from sklearn.decomposition import PCA



# %% Load csv data of counts and group information

directory = '/home/zachpen87/clearmap/data/SEFL17/results'

groups = pd.read_csv(os.path.join(directory,'groups.csv'))
counts = pd.read_csv(os.path.join(directory,'countsummary.csv'))




# %% Restrict counts to gray matter regions

json_file = '/home/zachpen87/clearmap/ClearMap2/ClearMap/Resources/Atlas/ABA_annotation.json'
ABA_annotation_file = '/home/zachpen87/clearmap/AtlasDocs/Horizontal/ABA_25um_annotation.tif'

with open(json_file) as jf:
    aba = json.load(jf)
    aba = aba['msg'][0]['children'][0]['children'][0]
    
def search_labels(dictionary,key='id',labels=None,region=None):
    if region==None or dictionary[key]==region:
        labels.append(dictionary[key])
        for subdict in dictionary['children']:
            labels = search_labels(subdict,key=key,labels=labels,region=None)
    else:
        for subdict in dictionary['children']:
            labels = search_labels(subdict,key=key,labels=labels,region=region)
    return labels
graymatter_labels = search_labels(aba,labels=[],region=None)

counts['gray'] = False
for lbl in np.unique(counts['annotation_labels']):
    if lbl in graymatter_labels:
        counts['gray'].iloc[
            np.where(counts['annotation_labels']==lbl)
            ] = True      
counts = counts[counts['gray']==True]




#%% Define label parcellation

#major parcellations of aba grey matter
#cerebrum = 567
    #cortex = 688
        #cortical plate = 695
            #isocortex = 315
            #olfactory areas = 698
            #hippocampal formation = 1089
        #cortical subplate = 703   
    #nuclei = 623
        #striatum = 477
        #pallidum = 803
#brainstem = 343
    #interbrain = 1129
        #thalamus = 549
        #hypothalamus = 1097
    #midbrain = 313
    #hindbrain = 1065
        #pons = 771
        #medulla = 354
#cerebellum = 512

parcellation_keys = {
    'cortical_plate' : 695,
    'cortical_subplate' : 703,
    'cerebral_nuclei' : 623,
    'diencephalon' : 1129,
    'midbrain' : 313,
    'hindbrain' : 1065,
    'cerebellum' : 512}

parcellation_dict = {}
for key in parcellation_keys.keys():
    parcellation_dict[key] = search_labels(
        aba, 
        labels = [],
        region = parcellation_keys[key])




# %% Get list of labels and their volumes

#get labels
counts.annotation_names = pd.Categorical(counts.annotation_names)
counts.annotation_labels = pd.Categorical(counts.annotation_labels)
LabelInfo = counts.loc[:,'annotation_labels':'graph_order'].drop_duplicates()

#define parcellation groups
LabelInfo['parcellation_group'] = None
for idx in np.arange(len(LabelInfo)):
    lbl = LabelInfo['annotation_labels'].iloc[idx]
    for key in parcellation_dict.keys():
        if lbl in parcellation_dict[key]:
            LabelInfo['parcellation_group'].iloc[idx] = key 


counts['parcellation_group'] = counts['annotation_labels'].map(
    dict(zip(LabelInfo['annotation_labels'], LabelInfo['parcellation_group'])))

#get volumes
ano = io.imread(ABA_annotation_file)
volumes = {}
volumes['annotation_labels'], volumes['volumes'] = np.unique(ano, return_counts=True)
LabelInfo = pd.merge(
    left = LabelInfo,
    right = pd.DataFrame(volumes), 
    how = 'left',
    on = 'annotation_labels')
counts['volumes'] = counts['annotation_labels'].map(
    dict(zip(LabelInfo['annotation_labels'], LabelInfo['volumes'])))

LabelInfo.sort_values(by='annotation_abrv',inplace=True)




# %% Create zero counts for each subject and add group info 

#find zero counts by subject/region and populate
missing = counts.groupby(['animal','annotation_labels'], as_index = False).count()
missing = missing[missing['count'].isnull()]
for col in LabelInfo.columns:
    missing[col] = missing['annotation_labels'].map(
        dict(zip(LabelInfo['annotation_labels'], LabelInfo[col])))
missing['count'] = 0

# #add data back to summary, add group info, and calculate counts per voxel
counts = pd.concat([counts,missing]).sort_values('animal')
counts = pd.merge(left=groups, right=counts, on='animal')
counts['count_vox'] = counts['count'] / counts['volumes']




# %% Drop olfactory bulb


#drop main olfacotry bulb, accessory olfactory bulb, and undefined olfacotry regions
olfb_abrvs = ['MOB','AOB','OLF']
olfb_locs = np.array([any([abrv in item for abrv in olfb_abrvs]) for item in counts['annotation_abrv']])
counts = counts[np.invert(olfb_locs)].reset_index()
counts['annotation_labels'].cat.remove_unused_categories(inplace=True)




# %% Define grouped summary information 

#get stats
stat_list = ['mean','median','std','sem','count']
groupsummary = counts.groupby(
    ['group','annotation_labels'],as_index=False).agg({
        'count' : stat_list,
        'count_vox' : stat_list})

#drop multi-index and rename sample size variable
groupsummary.columns = [
    ''.join(ind_names) if len(ind_names[-1])==0 else '_'.join(ind_names) for ind_names in groupsummary]

#add label names
groupsummary.insert(
    loc=2, column='annotation_names', 
    value = groupsummary['annotation_labels'].map(
        dict(zip(LabelInfo['annotation_labels'], LabelInfo['annotation_names']))))




# %% Create wide data format to easily compute stats

stat_list_cols = ['_'.join(['count',item]) for item in stat_list] \
            + ['_'.join(['count_vox',item]) for item in stat_list]

#pivot long dataframes to wide
groupsummary_wide = pd.pivot(
    data = groupsummary,
    index = 'annotation_labels',
    columns = 'group',
    values = stat_list_cols).reset_index()

#drop multi-index and rename sample size variable
for df in [groupsummary_wide]:
    df.columns = [
    ''.join(ind_names) if len(ind_names[-1])==0 else '_'.join(ind_names) for ind_names in df]

#add label info
for labelcol in ['parent_labels','annotation_names']: 
    groupsummary_wide.insert(
        loc=1, column=labelcol, 
        value = groupsummary_wide['annotation_labels'].map(
            dict(zip(LabelInfo['annotation_labels'], LabelInfo[labelcol]))))




# %% Find regions with low counts and exclude

# Get labels of regions with median counts <= 1 for all 3 groups
groupsummary_wide['low_counts'] = False
groupsummary_wide['low_counts'].iloc[np.where(
        np.logical_and.reduce((
            groupsummary_wide['count_median_Control'] <= 1,
            groupsummary_wide['count_median_No Trauma'] <= 1,
            groupsummary_wide['count_median_Trauma'] <= 1,))
        )] = True
lowcount_labels = groupsummary_wide['annotation_labels'].iloc[
        np.where(groupsummary_wide['low_counts']==True)]

#drop labels from group subject/group level dataframes and resave
for lbl in lowcount_labels:
    counts = counts.iloc[
        np.where(counts['annotation_labels']!=lbl)]
    groupsummary = groupsummary.iloc[
        np.where(groupsummary['annotation_labels']!=lbl)]
    groupsummary_wide = groupsummary_wide.iloc[
        np.where(groupsummary_wide['annotation_labels']!=lbl)]




# %% Perform t tests and define which regions show a difference from controls,
# either trauma or no trauma

comparisons = [
    ['No Trauma','Control'],
    ['Trauma','Control'],
    ['Trauma','No Trauma']]

for df in [groupsummary_wide]:
    
    for g1,g2 in comparisons:
        t,p = stats.ttest_ind_from_stats(
            mean1 = df['count_mean_{g1}'.format(g1=g1)],
            mean2 = df['count_mean_{g2}'.format(g2=g2)],
            std1 = df['count_std_{g1}'.format(g1=g1)],
            std2 = df['count_std_{g2}'.format(g2=g2)],
            nobs1 = df['count_count_{g1}'.format(g1=g1)],
            nobs2 = df['count_count_{g2}'.format(g2=g2)],
            equal_var = True)
        df['{g1}vs{g2}_t'.format(g1=g1,g2=g2)] = t
        df['{g1}vs{g2}_p'.format(g1=g1,g2=g2)] = p

    df['control_crit'] = np.logical_or(
        df['No TraumavsControl_p']<.05,
        df['TraumavsControl_p']<.05)




# %% Save data frames

counts.to_csv(os.path.join(directory,'countsummary_cleaned.csv'),index=False)
groupsummary.to_csv(os.path.join(directory,'groupsummary_long.csv'),index=False)
groupsummary_wide.to_csv(os.path.join(directory,'groupsummary_wide.csv'),index=False)
LabelInfo.to_csv(os.path.join(directory,'LabelInfo.csv'),index=False)


# %%


groupsummary_wide['control_crit'].sum()
activated_regions = groupsummary_wide['an']


# %%
