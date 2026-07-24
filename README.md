# An amygdala to anterior hypothalamic circuit gates stress sensitivity
### Pennington ZT, LaBanca AR, Mahmud AN, Abdel-Raheim SD, Bacon ME, Sompolpong P, Ko B, Baggetta AM, Zaki Y, Feng Y, Dong Z, Smith ACW, Shuman T, Kenny PJ, Cai DJ
*Nash Family Department of Neuroscience, Icahn School of Medicine at Mount Sinai*
 

# Data and Analysis
All data and analysis contained in manuscript are available here for reproduction and further analysis.  


# Use Restrictions
The contained data must not be used for republication without prior consent from the authors.


## Analysis files
- All analysis were conducted using RStudio and Python. Package dependencies are listed at the beginning of each analysis script.
- Generally, each experiment has 1-2 end-stage analysis files  (r markdown, .rmd).  When the working directory is set to .rmd file location, the file can be run to load in all experimental data and reproduce analyses/plots.
- In addition to providing the data, each experiment folder has a summary file that contains information on each animal, experimental procedures, and experimental run dates.
- Please reach out if interested in additional data. 
- A brief description of each experiment is provided below.
- Scripts from github.com/ZachPenn/ClearMap2, zmaster branch, were copied here for convenience. For commit history, see original branch.
- *Analysis of calcium imaging traces was performed using github.com/DeniseCailab/Minian.*



## Nomenclature
Over the years, what certain things were called evolved, though the procedures remained the same. Nevertheless, you will likely encounter these names in these files.

### Experiments
Many experiments originated from a line of work based upon the stress-enhanced fear learning (SEFL) literature. Within this repository, experiments will sometimes be referred to with the 'SEFL" prefix (e.g., 'SEFL_17'), and sometimes only with their suffix (e.g. 17). These are nevertheless the same experiment.

### Groups
`No Stress (NS)` used to be referred to as `No Trauma (NT)`  
`Stress (S)` used to be referred to as `Trauma (T)`

#### Procedures  
`Stressor 1 (S1)` used to be referred to as `Trauma`   
`Stressor 1 (S1) Recall` used to be referred to as `Trauma Test` or `Trauma Recall`   
`Stressor 1 (S1)` used to be referred to as `Novel Stressor` or `Mild Stressor`   
`Stressor 2 (S2) Recall` used to be referred to as `SEFL test`  



## Figures
For each figure, we highlight the respective experiments included. Further down, we provide a very brief description of each experiment. 

### Figure 1
- Experiment 17 (panels a-g)

### Figure 2
- Experiment 25b (panels a-j)
- Experiment 25c_25c2_25c5 (panels k-n)

### Figure 3
- Experiment 21H2 (panels a-f)
- Experiment 21G4d (panels g-l)
- Experiment 21G3 (panels m-o)

### Figure 4
- Experiment 25d (panels a-h)
- Experiment 10N6d (panels i-n)

### Extended Data Figure 1 
- Experiment 17 (panels a-j)

### Extended Data Figure 2 
- Experiment 17 (panels a-d)

### Extended Data Figure 3
- Experiment 25b (panels a-g)

### Extended Data Figure 4
- Experiment 21H (panels a-f)
- Experiment 21H2 (panels g-i)
- Experiment 21G3 (panels j-l)

### Extended Data Figure 5
- Experiment 10N6d (panels a-j)



## Experiments

A very brief description of experiments.  See manuscript for details.

### 10N6d
Terminal inhibition of inputs from amygdala to AHN. Done in two batches (eOPN3/EGFP). Within each batch are controls that were not exposed to laser light.

### 17
iDISCO experiment examining impact of prior stress (Stressor 1) on brain-wide response to a subsequent stressor (Stressor 2).   

### 21G3
Chemogenetic activation of AHN GAD neurons during a stressor, as well as during the light-dark test.

### 21G4d
Optogenetic inhibition of GABAergic AHN neurons during Stressor 1 Recall, Stressor 2, and the Light-Dark test.

### 21H
Optogenetic inhibition of AHN neurons (pan-neuronal) during Stressor 1, and its long-term consequence on stress-associated behaviors.

### 21H2
Optogenetic inhibition of GABAergic AHN neurons during Stressor 1, and its long-term consequence on stress-associated behaviors.

### 25b
Calcium imaging of GABAergic AHN neurons during Stressor 1 and Stressor 2. Addititionally, examined sensitivity to shocks and sounds of various intensities at the end of the experiment.

### 25c_25c2_25c5
Calcium imaging of GABAergic AHN neurons during exposure to stimuli of both postive valence and negative valence.

### 25d
Calcium imaging of BLA neurons that project to the AHN.