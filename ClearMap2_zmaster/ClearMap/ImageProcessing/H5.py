"""
H5 module provides tools for applying 3d image
kernels on arrays stored in HDF5 datasets, allowing incremental loading
and distributed processing. Moreover, methods for labeleling connected objects
(presumably cells) and finding their centers of mass are provided. Lastly, 
functions for testing kernels on smaller numpy arrays are also inncluded.


Functions:
array_filter
hdf5_filter
set_filter
gen_kernel
hdf5_bgsubtract
label
drop_lbls_bysize
find centroids
check_chunksize


Note
----

"""

import os
import natsort
import numpy as np
import dask
import dask.array as da
import h5py
import scipy.ndimage as ndimage
from tqdm import tqdm
import time



#
#
#
#
#

    
    
def array_filter(array=None,filt=None,ksize=None):
    """Applies 3d filter to n-dimensional numpy array
    
    Arguments
    ---------
    array : numpy array
      Input array to be filtered.
    filt : str
      The filter to be applied:
    ksize : tuple
      Length of kernel for each dimension as tuple.
      (e.g. for 3d kernel, ksize=(5,5,5))
    
    Returns
    -------
    array: numpy array
      The filtered array.
    
    Notes
    -----
    Currently, kernel is assumed to be the same size for each
    dimension.  Moreover, although even ksizes can be specified, the
    central node of structuring element has a size of 2 in each dimension.
    
    """
    
    func = set_filter(filt, ksize)
    return func(array)
    

    
def hdf5_filter(
    hdf5_file = None,
    dset_in = None,
    dset_out = None,
    filt = None,
    ksize = None,
    chunksize = (400,400,400),
    chunkoverlap = (10,10,10),
    compression = None,
    slc = None):
    """Applies 3d filter to n-dimensional array stored within
    a dask data set, and saves the result to a separate dataset within
    the same file.
    
    Arguments
    ---------
    hdf5_file : string
      Full file path of hdf5 file from which to load and save data
    dset_in : string
      Name of existing dataset within hdf5 file to operate on
    set_out: string
      Name of dataset within hdf5 file to save data to.
    filt : str
      The filter to be applied:
          - 'median'
          - 'uniform'
          - 'min'
          - 'max'
          - 'erosion'
          - 'dilation'
          - 'closing'
          - 'opening'
          - 'local max'
      Erosion, dilation, opening, and closing all 
      utilize diamond footprint (see scipy ndimage documentation). 
      Local max returns the value of the local max in its original location.
      All other values set to 0.
    ksize : tuple
      Length of kernel for each dimension as tuple.
      (e.g. for 3d kernel, ksize=(5,5,5)). Currently, kernel is assumed to be 
      the same size for each dimension.  Moreover, although even ksizes can be 
      specified, the central node of structuring element has a size of 2 
      in each dimension.
    chunksize : tuple
      Size of each chunk in dask data array.  Must be same lenth as number of
      dimensions (e.g. for 3d dataset, chunksize=(400,400,400)).  Making chunks
      too large can stress memory load, while a chunksize that is too small will
      increase overhead and slow rate of processing.
    chunkoverlap : tuple
      Number of elements chunks will overlap across when functions applied are
      affected by borders.  Specified for each dimension, similar to chunksize.
      Overlap should be at least half the size of largest kernel used.
    compression: str
      The compression formula used. Method 'lzf' is a fast and lossless compression.
      See h5py's documentation for further details.
    slc : str or None
      Slice object to apply to dset_in before processing
    
    Returns
    -------
    
    
    Notes
    -----
    For gaussian filter, ksize sets sigma parameter.
    See dask and h5py documentation for further details.
    
    """
    
    with h5py.File(hdf5_file, 'a') as f:
    
        if ('/'+dset_out) in f:
            print('Overwriting existing dataset: {x}'.format(x=dset_out))
            del f[dset_out]
                  
        if slc!=None:
            chunksize = check_chunksize(f[dset_in][slc].shape, chunksize, chunkoverlap)
            darray = da.from_array(
                f[dset_in][slc],
                chunks = chunksize)
        else:
            chunksize = check_chunksize(f[dset_in].shape, chunksize, chunkoverlap)
            darray = da.from_array(
                f[dset_in],
                chunks = chunksize)
        
        func = set_filter(filt, ksize)
        darray.map_overlap(
            func = func, depth = chunkoverlap, boundary = 'reflect'
        ).to_hdf5(hdf5_file,('/'+dset_out), compression=compression)
        
        print('Results of {x} filter written to {y}'.format(
            x = filt, y = '/'.join([hdf5_file,dset_out])))


        
def set_filter(filt, ksize=None):
    """Returns lamda function to execute filter
    
    Arguments
    ---------
    filt : str
      The filter to be applied:
          - 'median'
          - 'uniform'
          - 'gaussian'
          - 'min'
          - 'max'
          - 'erosion'
          - 'dilation'
          - 'closing'
          - 'opening'
          - 'local max'
      Erosion, dilation, opening, and closing all 
      utilize diamond footprint (see scipy ndimage documentation). 
      Local max returns the value of the local max in its original location.
      All other values set to 0.
    ksize : tuple
      Length of kernel for each dimension as tuple.
      (e.g. for 3d kernel, ksize=(5,5,5)). Currently, kernel is assumed to be 
      the same size for each dimension.  Moreover, although even ksizes can be 
      specified, the central node of structuring element has a size of 2 
      in each dimension.
    
    Returns
    -------
    func : lambda function
      Lambda function, subsequently deployed to excecute filter with proper
      kernel
        
    Notes
    -----
    See dask and h5py documentation for further details.
    
    """
        
    if filt == 'median':
        func = lambda x: ndimage.median_filter(x, footprint = gen_kernel(ksize))
    if filt == 'uniform':
        func = lambda x: ndimage.uniform_filter(x, size = ksize)
    if filt == 'gaussian':
        func = lambda x: ndimage.gaussian_filter(x, sigma = ksize)
    if filt == 'min':
        func = lambda x: ndimage.minimum_filter(x, footprint = gen_kernel(ksize))
    if filt == 'max':
        func = lambda x: ndimage.maximum_filter(x, footprint = gen_kernel(ksize))
    if filt == 'erosion':
        func = lambda x: ndimage.grey_erosion(x, footprint = gen_kernel(ksize,uniform=False))
    if filt == 'dilation':
        func = lambda x: ndimage.grey_dilation(x, footprint = gen_kernel(ksize,uniform=False))
    if filt == 'closing':
        func = lambda x: ndimage.grey_closing(x, footprint = gen_kernel(ksize,uniform=False))
    if filt == 'opening':
        func = lambda x: ndimage.grey_opening(x, footprint = gen_kernel(ksize,uniform=False))
    if filt == 'distance_transform':
        func = lambda x: ndimage.distance_transform_cdt(x)
    if filt == 'local_max':
        func = lambda x: np.where(
            np.logical_and(
                ndimage.maximum_filter(x, footprint = gen_kernel(ksize)) == x,
                x > 0),x,0)
    return func 

    
    
def gen_kernel(ksize, uniform=True):
    """Returns kernel/footprint to use for filtering
    
    Arguments
    ---------
    ksize : tuple
      Length of kernel for each dimension as tuple.
      (e.g. for 3d kernel, ksize=(5,5,5)). Currently, kernel is assumed to be 
      the same size for each dimension.  Moreover, although even ksizes can be 
      specified, the central node of structuring element has a size of 2 
      in each dimension.
    uniform : bool
      If uniform = True, footprint/kernel used will equivalent to numpy.ones(ksize).
      Otherwise, a diamond-shaped kernel will be generated
    
    Returns
    -------
    struct : numpy array
      Kernel stucture/footprint.
        
    Notes
    -----
    See dask and h5py documentation for further details.
    
    """
    
    if uniform:
        return np.ones(ksize)
    else:
        k =  ksize[0]
        base_structure = ndimage.generate_binary_structure(3, 1)
        struct = np.zeros(ksize)
        ctr = k//2

        if k%2 == 1:
            dilations = k-ctr-1
            struct[ctr,ctr,ctr] = 1
            struct = ndimage.binary_dilation(
                struct,
                structure = base_structure,
                iterations = dilations)
        elif k%2 == 0:
            dilations = k-ctr-1
            struct[ctr-1:ctr+1,ctr-1:ctr+1,ctr-1:ctr+1] = 1
            struct = ndimage.binary_dilation(
                struct,
                structure = base_structure,
                iterations = dilations)
        return struct
    
    
    
def hdf5_bgsubtract(
    hdf5_file = None, 
    dset_img = None, 
    dset_bg = None, 
    dset_out = None, 
    chunksize = (400,400,400),
    compression = None):
    """Subtracts background from image
    """
     
    with h5py.File(hdf5_file, 'a') as f:
        
        if ('/'+dset_out) in f:      
            print('Overwriting existing {x}'.format(x=dset_out))
            del f[dset_out]
        
        image = da.from_array(
            f[dset_img],
            chunks = chunksize)
        bg = da.from_array(
            f[dset_bg],
            chunks = chunksize)
        sig = image - bg
        sig.to_hdf5(hdf5_file,('/'+dset_out))
        
        print('Results of background subtraction written to {y}'.format(
            y = '/'.join([hdf5_file,dset_out])))
        
        
        
def label(
    hdf5_file = None,
    dset_in = None,
    dset_out = 'lbls',
    min_size = None,
    max_size = None,
    chunk_dimension = 0,
    chunk_size = None,
    chunk_overlap = 100,
    srch_dist = 25):
    """Given 3d array in hdf5 dataset, creates dataset with connected regions
    from input given distinct labels
    
    Arguments
    ---------
    hdf5_file : string
      Full file path of hdf5 file from which to load and save data
    dset_in : string
      Name of existing dataset within hdf5 file to operate on
    set_out: string
      Name of dataset within hdf5 file to save data to.
    min_size: int
      minimum size of cell to include, in pixel volume
    max_size: int
      maximum size of cell to include, in pixel volume
    chunk_dimension: int (0=z,1=y or 2=x)
      Dimensionn along which to chunk data
    chunk_size: int
      Length of each chunk along chunked dimension.  Should
      be smaller than dimension length
    chunk_overlap: int
      Extent to which chunks should overlap.  Must be smaller than
      chunk_size.
    srch_dist: int
      Distance to search in chunk for duplicated 
      labels from adjacennt chunk.
    
    Returns
    -------
        
    Notes
    -----
    See h5py documentation for further details.
    
    """
    
    if chunk_overlap > chunk_size:
        raise ValueError('chunk_overlap cannot be larger than chunk_size')
    
    with h5py.File(hdf5_file, 'a') as f:
        
        # define chunking
        #######################
        
        #span of dimension to be chunked
        cdim_span = (0, f[dset_in].shape[chunk_dimension])

        #define indices of final orthogonal chunks and overlapping chunks, relative to full data set
        chunk_starts = np.arange(cdim_span[0], cdim_span[1], chunk_size)
        chunk_stops = np.array([x+chunk_size if x+chunk_size<cdim_span[1] else cdim_span[1] for x in chunk_starts])
        ovrchunk_starts = np.array([x-chunk_overlap if x!=0 else 0 for x in chunk_starts])
        ovrchunk_stops = np.array([x+chunk_overlap if x+chunk_overlap<cdim_span[1] else cdim_span[1] for x in chunk_stops])

        #define indices to trim overlapping chunks down to, leaving seam
        trmchunk_starts = chunk_starts - ovrchunk_starts
        trmchunk_stops = trmchunk_starts + chunk_size
        trmchunk_stops[0:-1] = trmchunk_stops[0:-1] + 1

        #For each successive pair of chunks (e.g. chk1/chk2, chk2/chk3, etc)
        #Define index of matching seam from trimmed chunk
        seam_indices = [(chunk_size,0)]*(len(chunk_starts)-1)
        
        if ('/'+dset_out) in f:
            print('Overwriting existing dataset: {x}'.format(x=dset_out))
            del f[dset_out]
        if ('/'+'chunks') in f:
            del f['chunks']
        
        
        # Define labels for each chunk individually
        ###########################################
        
        lblcounter = 0
        for idx in range(len(chunk_starts)):
            
            print('Defining labels for chunk: {chunk}/{tot}'.format(
                chunk = idx+1, tot = len(chunk_starts)))
            
            #get expanded chunk
            ovrchunk_slice = slice(ovrchunk_starts[idx], ovrchunk_stops[idx])
            if chunk_dimension == 0:
                ovrchunk = f[dset_in][ovrchunk_slice,:,:]
            if chunk_dimension == 1:
                ovrchunk = f[dset_in][:,ovrchunk_slice,:]
            if chunk_dimension == 2:
                ovrchunk = f[dset_in][:,:,ovrchunk_slice]

            #label connected regions
            ovrchunk = ndimage.label(
                ovrchunk, 
                structure=np.ones((3,3,3)))[0]
            
            #drop labels that are too small/large
            ovrchunk = droplbls_arr(ovrchunk,min_size=min_size,max_size=max_size)

            #trim data
            trmchunk_slice = slice(trmchunk_starts[idx], trmchunk_stops[idx])
            if chunk_dimension == 0:
                trmchunk = ovrchunk[trmchunk_slice,:,:]
            if chunk_dimension == 1:
                trmchunk = ovrchunk[:,trmchunk_slice,:]
            if chunk_dimension == 2:
                trmchunk = ovrchunk[:,:,trmchunk_slice]

            #update lbls
            chk_lab_max = trmchunk.max()
            trmchunk[trmchunk!=0] = trmchunk[trmchunk!=0] + lblcounter
            lblcounter = lblcounter + chk_lab_max

            #write processed data
            f.create_dataset('chunks/chunk{i}'.format(i=idx), data=trmchunk)
     
    
        # handle overlap in labels between chunks
        #########################################

        for idx in range(len(seam_indices)):
            
            print('Handling overlappinng labels for chunks: {chunk1} & {chunk2}'.format(
                chunk1 = idx+1, chunk2 = idx+2))

            #load seams
            if chunk_dimension == 0:
                c1_seam = f['chunks/chunk{i}'.format(i=idx)][seam_indices[idx][0],:,:]
                c2_seam = f['chunks/chunk{i}'.format(i=idx+1)][seam_indices[idx][1],:,:]
            if chunk_dimension == 1:
                c1_seam = f['chunks/chunk{i}'.format(i=idx)][:,seam_indices[idx][0],:]
                c2_seam = f['chunks/chunk{i}'.format(i=idx+1)][:,seam_indices[idx][1],:]
            if chunk_dimension == 2:
                c1_seam = f['chunks/chunk{i}'.format(i=idx)][:,:,seam_indices[idx][0]]
                c2_seam = f['chunks/chunk{i}'.format(i=idx+1)][:,:,seam_indices[idx][1]]

            #define label mapping between seams
            seam_lbl_locs = np.where(np.logical_and(c1_seam!=0, c2_seam!=0))
            lbl_dict = {}
            for (lbls, c_seam) in (['c1',c1_seam],['c2',c2_seam]):
                lbl_dict[lbls] = np.array([
                    c_seam[seam_lbl_locs[0][x], seam_lbl_locs[1][x]] 
                    for x in range(len(seam_lbl_locs[0]))])
                if len(lbl_dict[lbls])!=0:
                    _, i = np.unique(lbl_dict[lbls], return_index=True)
                    lbl_dict[lbls] = lbl_dict[lbls][np.sort(i)]
                    
            if len(lbl_dict['c1']) != len(lbl_dict['c1']):
                raise IndexError('Different numbers of unique labels detected.  Try increasing chunk overlap')
            
            #apply label mapping from chunk 1 to chunk 2
            f_location = f['chunks/chunk{i}'.format(i=idx+1)]
            c2 = f_location[:]
            if chunk_dimension == 0:
                srch_slc = (slice(None,srch_dist),slice(None,None),slice(None,None))
            if chunk_dimension == 1:
                srch_slc = (slice(None,None),slice(None,srch_dist),slice(None,None))
            if chunk_dimension == 2:
                srch_slc = (slice(None,None),slice(None,None),slice(None,srch_dist))
            for i in tqdm(range(len(lbl_dict['c2']))):
                c2[srch_slc][np.where(c2[srch_slc]==lbl_dict['c2'][i])] = lbl_dict['c1'][i]
            f_location[:] = c2

        del c2

        # create output dataset
        #######################
        
        f.create_dataset(dset_out, (f[dset_in].shape), dtype='uint32')
        chunk_slice = slice(0, chunk_size)
        for (idx, chunk) in enumerate(natsort.natsorted(f['chunks'].keys())):
            
            print('Writing chunk {chunk}/{tot} to {dset}'.format(
                chunk = idx+1, tot = len(chunk_starts),
                dset = '/'.join([hdf5_file,dset_out])))

            data_slice = slice(idx*chunk_size, (idx+1)*chunk_size)
            if chunk_dimension == 0:
                f[dset_out][data_slice,:,:] = f['/'.join(['chunks',chunk])][chunk_slice,:,:]
            if chunk_dimension == 1:
                f[dset_out][:,data_slice,:] = f['/'.join(['chunks',chunk])][:,chunk_slice,:]
            if chunk_dimension == 2:
                f[dset_out][:,:,data_slice] = f['/'.join(['chunks',chunk])][:,:,chunk_slice]
            
        del f['chunks']
        
        
        
def droplbls_arr(lbl_array, min_size=None, max_size=None):
    """Given a labelled array, remove labels with size<min_size
    and size>max and return original array.
    
    Arguments
    ---------
    lbl_array : np array
      Array of labels, such as that resulting from scipy.ndimage.label
    min_size: int or None
      Size, in pixels/voxels, below which a label should excluded.
    max_size: int or None
      Size, in pixels/voxels, above which a label should excluded.
    
    Returns
    -------
    lbl_array : np array
      Array of labels, such as that resulting from scipy.ndimage.label
      
    Notes
    -----
    See h5py documentation for further details.
    
    """
    
    #if called without specifing size, return input
    if min_size==None and max_size==None:
        return lbl_array
    
    #define labels and sizes
    all_lbls, sizes = np.unique(lbl_array,return_counts=True)
    
    #remove labels as necessary
    if max_size is not None:
        excluded_lbls =  all_lbls[np.where(sizes>max_size)]
        print('removing labels of large size')
        time.sleep(.5)
        for excluded_lbl in tqdm(excluded_lbls):
            lbl_array[np.where(lbl_array==excluded_lbl)] = 0
    if min_size is not None:
        excluded_lbls =  all_lbls[np.where(sizes<min_size)]
        print('removing labels of small size')
        time.sleep(.5)
        for excluded_lbl in tqdm(excluded_lbls):
            lbl_array[np.where(lbl_array==excluded_lbl)] = 0

    return lbl_array


        
def find_centroids(
    hdf5_file = None,
    dset_lbls = None,
    dset_wts = None,
    chunk_dimension = 0,
    chunk_size = None,
    chunk_overlap = 50):
    """Given 3d array in hdf5 dataset, creates dataset with connected regions
    from input given distinct labels
    
    Arguments
    ---------
    hdf5_file : string
      Full file path of hdf5 file from which to load and save data
    dset_lbls : string
      Name of existing dataset within hdf5 file containing labeled
      numpy array.
    dset_wts : string
      Name of existing dataset within hdf5 file that will be used
      as weights for finding center of mass for each label.  If None,
      lbl array will be used instead.
    chunk_dimension: int (0=z,1=y or 2=x)
      Dimensionn along which to chunk data
    chunk_size: int
      Length of each chunk along chunked dimension.  Should
      be smaller than dimension length
    chunk_overlap: int
      Extent to which chunks should overlap.  Must be smaller than
      chunk_size.
    
    Returns
    -------
    
    array : numpy array
    array of center of mass locations (z, y, x)
        
    Notes
    -----
    See h5py documentation for further details.
    
    """
    
    if chunk_overlap > chunk_size:
        raise ValueError('chunk_overlap cannot be larger than chunk_size')
    
    with h5py.File(hdf5_file, 'a') as f:     
        
        # define chunking
        #######################
        
        #span of dimension to be chunked
        cdim_span = (0, f[dset_lbls].shape[chunk_dimension])

        #define indices of final orthogonal chunks and overlapping chunks, relative to full data set
        chunk_starts = np.arange(cdim_span[0], cdim_span[1], chunk_size)
        chunk_stops = np.array([x+chunk_size if x+chunk_size<cdim_span[1] else cdim_span[1] for x in chunk_starts])
        ovrchunk_starts = np.array([x-chunk_overlap if x!=0 else 0 for x in chunk_starts])
        ovrchunk_stops = np.array([x+chunk_overlap if x+chunk_overlap<cdim_span[1] else cdim_span[1] for x in chunk_stops])

        #to remove overlap, define indices to trim array of centroids from overlapping chunks down
        trm_starts = chunk_starts - ovrchunk_starts
        trm_stops = trm_starts + chunk_size
        
        # Define labels for each chunk individually
        ###########################################
        
        lblcounter = 0
        for idx in range(len(chunk_starts)):
            
            print('Defining centroids for chunk: {chunk}/{tot}'.format(
                chunk = idx+1, tot = len(chunk_starts)))
            
            #get expanded chunk
            ovrchunk_slice = slice(ovrchunk_starts[idx], ovrchunk_stops[idx])
            dset_wts=dset_lbls if dset_wts==None else dset_wts
            if chunk_dimension == 0:
                ovrchunk_lbls = f[dset_lbls][ovrchunk_slice,:,:]
                ovrchunk_wts = f[dset_wts][ovrchunk_slice,:,:]
            if chunk_dimension == 1:
                ovrchunk_lbls = f[dset_lbls][:,ovrchunk_slice,:]
                ovrchunk_wts = f[dset_wts][:,ovrchunk_slice,:]
            if chunk_dimension == 2:
                ovrchunk_lbls = f[dset_lbls][:,:,ovrchunk_slice]
                ovrchunk_wts = f[dset_wts][:,:,ovrchunk_slice]
            
            #define centroids in overlappinng chunks
            lbl_ids = np.unique(ovrchunk_lbls)[np.where(np.unique(ovrchunk_lbls)!=0)]
            chunk_com = ndimage.measurements.center_of_mass(
                ovrchunk_wts, 
                labels = ovrchunk_lbls,
                index = lbl_ids)
            chunk_com = np.array([list(tup) for tup in chunk_com])

            #restrict centroids to those in non-overlapping chunks
            #correct indices, and add to overall com array
            if len(chunk_com)>0:
                indices_kept = np.where(np.logical_and(
                    chunk_com[:,chunk_dimension]>=trm_starts[idx], 
                    chunk_com[:,chunk_dimension]<trm_stops[idx]))[0]
                chunk_com = chunk_com[indices_kept,:]
                chunk_com[:,chunk_dimension] += idx*chunk_size - (idx>0)*chunk_overlap
            
                try:
                    com = np.concatenate([com,chunk_com],axis=0)
                except:
                    com = chunk_com
    return com




def check_chunksize(isize, csize, olap):
    """Given array size, chunksize, and overlap, check that smallest 
    chunk is not less than overlap, and if it is, adjust chunksize
    to avoid error
    
    Arguments
    ---------
    isize : tuple
      shape of array
    csize : tuple
      chunksize for each dimension of array
    olap : tuple
      overlap along each dimension of array
    
    Returns
    -------
    csize : tuple
      chunksize for each dimension of array
        
    Notes
    -----
    
    """
    
    isize, csize, olap = np.array([isize, csize, olap])
    changed = False
    
    idx = np.where(np.logical_and(
        (isize % csize > 0),
        (isize % csize < olap)))
    
    while len(idx[0]) > 0:
        changed = True
        csize[idx] -= 1
        idx = np.where(np.logical_and(
            (isize % csize > 0),
            (isize % csize < olap)))
        
    csize = tuple(csize)
    if changed:
        print('chunksize changed to: {csize} so overlap would be smaller than smallest chunk'.format(csize=csize))
    return csize