"""
TIF_to_HDF5 module provides reading in of image folder and conversion to hdf5 dataset.

Note
----
Currently, this module assumes image filenames provide ordered indexing of
stack (e.g img0001.tif)

"""


import fnmatch
import h5py
import os
import cv2
import numpy as np


def get_files(directory,img_ext='tif'):
    """Returns list of full path to all images of particular file type in directory
    Arguments
    ---------
    directory : str 
      The directory of the folder.
    img_ext : str
      The extension of file type (e.g. 'tif')
    
    Returns
    -------
    img_files : list
      list of full path to all images in directory
    
    Notes
    -----
    
    """
    
    img_files = fnmatch.filter(
        sorted(os.listdir(directory)), 
        '.'.join(['*',img_ext]))
    img_files = [os.path.join(directory,f) for f in img_files]
    return img_files


def get_img_info(img_files):
    """Returns dictionary object with file information
    Arguments
    ---------
    img_files : str list 
      list of full path to all images in directory
    
    Returns
    -------
    img_info : dict
      dictionary containing image information, including width ['w'], height['h'],
      the number of files in the stack ['n_files'], and the datatype as an h5py
      compatible string (e.g 'i1','i2','i4', for 8, 16, and 32 bit images)
      
    Notes
    -----
    
    """
    
    img_info = {}
    img_info['n_files'] = len(img_files)

    image = cv2.imread(
        img_files[0],
        cv2.IMREAD_ANYDEPTH)

    img_info['w'], img_info['h'] = image.shape[1], image.shape[0]

    img_info['dtype'] = image.dtype
    if img_info['dtype'] == 'uint8':
        img_info['dtype'] = 'i1'
    elif img_info['dtype'] == 'uint16':
        img_info['dtype'] = 'i2'    
    elif img_info['dtype'] == 'uint32':
        img_info['dtype'] = 'i4'
    else:
        raise TypeError('{type} not recognnized. Should be uint8, uint16, or uint32)'.format(type=img_info['dtype']))
    return img_info


def tiffolder_tohdf5(directory, img_ext='tif', output_file='output.hdf5', dsetname='signal', compression='lzf',
                     directory_out=None, buffer_size=100):
    """Creates hdf5 file with tif stack now represented as single 3d array
    Arguments
    ---------
    directory : str 
      The directory of the folder.
    img_ext : str
      The extension of file type (e.g. '.tif')
    output_file: str
      Name of hdf5 file
    dsetname: str
      The name of the dataset within the hdf5 file
    compression: str
      The compression formula used. Method 'lzf' is a fast and lossless compression.
      See h5py's documentation for further details.
    directory_out: str
      Alternative directory to save hdf5 file under. By default data hdf5 is saved in parent 
      directory of directory (e.g. if directory=/data/s1/signal, data will be saved under data/s1)
    buffer_size: int
      The number of files to write to disk at a time.  The larger this number the faster, but can
      produce significant memory demand.
    
    Returns
    -------
    
    
    Notes
    -----
    
    """

    img_files = get_files(directory,img_ext)
    img_info = get_img_info(img_files)

    if directory_out == None:
        output_file = os.path.join(
            os.path.split(directory)[0],
            output_file)
    else:
        output_file = os.path.join(directory_out,output_file)
        


    with h5py.File(output_file, 'a') as f:

        if os.path.exists(output_file):
            print('Opening hd5f file: {file}'.format(file=output_file))
        else:
            print('Creating hd5f file: {file}'.format(file=output_file))
            
        if ('/'+dsetname) in f:
            print('Overwriting existing dataset: {x}'.format(x=dsetname))
            del f[dsetname]
        
        stack = f.create_dataset(
            name = dsetname, 
            shape = (
                img_info['n_files'],
                img_info['h'],
                img_info['w']), 
            dtype = img_info['dtype'],
            compression = compression)   
        print('Creating dataset, {name}, of size={size}, type={type}'.format(
            name = dsetname,
            size = stack.shape, 
            type = img_info['dtype']))
        
        buffer = np.zeros((buffer_size,img_info['h'],img_info['w']))
        buf_cntr = 0
        for (idx,file) in enumerate(img_files):
            buffer[buf_cntr,:,:] = cv2.imread(file, cv2.IMREAD_ANYDEPTH)
            if buf_cntr == (buffer_size-1):
                stack[(idx-buf_cntr):(idx+1),:,:] = buffer
                buf_cntr = 0
                print('completed writing {x}/{y} images to disk'.format(x=idx+1,y=img_info['n_files']))
            else:
                buf_cntr += 1       
        if buf_cntr != 0:
            buffer = buffer[:buf_cntr,:,:]
            stack[(len(img_files)-buf_cntr):,:,:] = buffer
            print('completed writing {y}/{y} images to disk'.format(y=img_info['n_files']))
            

            
                
            
            
            
            
            