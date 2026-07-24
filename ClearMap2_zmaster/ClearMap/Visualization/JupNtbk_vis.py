"""
JupNtbk_vis module provides tools for interactive visualizations
in Jupyter Notebook using Holoviews package.  

Note
----

"""


import holoviews as hv
from holoviews import streams
import numpy as np
import SimpleITK as sitk
import skimage
from skimage import io
import os


def load_alignment_images(ws, channels=None):
    """Loads resampled image files in order to check
     results of alignment to Allen Brain Atlas
    
    Arguments
    ---------
    ws : ClearMap2 worksheet object
      Worksheet containing file information for 
      brain.
    channels : character list
      Either list containing postfix ids when multiple signal
      channels.  Alternatively, set to None if only single signal
      channel with no postfix.
    
    Returns
    -------
    images : dictionary
      Dictionary containing 3d array for each channel.  
      Key represents channel name. Autofluorescence channel,
      alligned with signal. Array representing Allen Brain Atlas reference brain,
      alligned to autofluorescence.
    
    Notes
    -----
    All take the form (z,y,x). This can subsequently be flipped for purposes
    of presentation.
    
    """
    
    images = {}
    signal = False
    
    images['auto'] = skimage.io.imread(ws.filename('resampled', postfix='autofluorescence'))
    
    if os.path.isfile(ws.filename('resampled')):
        images['signal'] = skimage.io.imread(ws.filename('resampled'))
        signal = True
    else:
        if channels != None: 
            signal = True
            for channel in channels:
                images[channel] = skimage.io.imread(ws.filename('resampled', postfix = channel))

    if os.path.isfile(os.path.join(ws.filename('auto_to_reference'),'result.1.mhd')):
        images['reference'] = sitk.ReadImage(os.path.join(ws.filename('auto_to_reference'),'result.1.mhd'))
        images['reference'] = sitk.GetArrayFromImage(images['reference'])
        
    if signal:
        images['auto_to_sig'] = sitk.ReadImage(
            os.path.join(
                ws.filename(
                    'resampled_to_auto', 
                    postfix = channels if channels is None else channels[0]
                ),
                'result.0.mhd'
            ))
        images['auto_to_sig'] = sitk.GetArrayFromImage(images['auto_to_sig'])
    
    return images


def plane_display(p,img,plane,lims,title,cmap,alpha,tools):
    """Given 3d volumetric data, returns holoviews image of a
    single slice/plane, along the axis of choice.
    
    Arguments
    ---------
    p : int
      The image plane/slice to display
    img : numpy array
      3-dimensional array to slice data from
    plane : 'string'
      Either 'h' (horizontal), 's' (sagittal), or 'c' (coronal).
      Note that this assumes one is working with horizontal images,
      such that dimensions (z,y,x) or an image array correspond with
      (h,c,s). 
    lims : tuple
      Tuple of the form (min, max), defining the min and max values represented
      by the image color map.
    title: string
      The title of the image, displayed atop image
    cmap: string
      Colormap name.  See holoviews documentation.
    alpha: numeric
      Alpha weighting of image opacity, where 1 = opaque.
    
    Returns
    -------
    image : holoviews.Image
      Interactive image with zoom, pan, hover options
    
    Notes
    -----
    
    """
    
    if plane == 'h':
        array = img[p,:,:]
    elif plane == 'c':
        array = img[:,p,:]
    elif plane == 's':
        array = img[:,:,p]
    
    image = hv.Image((np.arange(array.shape[1]),np.arange(array.shape[0]),array))
    image.opts(
        width = int(array.shape[1]),
        height = int(array.shape[0]),
        invert_yaxis = True,
        clim = lims,
        title = title,
        cmap=cmap,
        alpha=alpha,
        tools=tools)
    return image


def gen_hmap(img,plane,title=None,inter=25,cmap='gray',alpha=1,lims=None,tools=[]):
    """Given 3d volumetric data, returns holoviews holomap to allow
    scanning through image stack along 3 major anatomical planes.
    
    Arguments
    ---------
    img : numpy array
      3-dimensional array to slice data from
    plane : 'string'
      Either 'h' (horizontal), 's' (sagittal), or 'c' (coronal).
      Note that this assumes one is working with horizontal images,
      such that dimensions (z,y,x) or an image array correspond with
      (h,c,s). 
    title: string
      The title of the image, displayed atop image
    inter: int
      Spacing between slices, in pixel units.
    cmap: string
      Colormap name.  See holoviews documentation.
    alpha: numeric
      Alpha weighting of image opacity, where 1 = opaque.
    lims: tuple
      The upper and lower limits of values for colormap
    
    Returns
    -------
    image : holoviews.Image
      Interactive image with zoom, pan, hover options
    
    Notes
    -----
    
    """
    

    if plane == 'h':
        idx = 0
    elif plane == 'c':
        idx = 1
    elif plane == 's':
        idx = 2
    
    lims = (0,int(np.percentile(img,99.99))) if lims==None else lims
        
    img_dict = {p : plane_display(p,img,plane,lims,title,cmap,alpha,tools) for p in np.arange(0,img.shape[idx],inter)}
    return hv.HoloMap(img_dict, kdims='slice')



def crop_img(array, title='Max Projectioon', lims=None):
    """Given 2d image, returns holoviews image and cropping tool.
    
    Arguments
    ---------
    array : numpy array
      2-dimensional array
    title: string
      The title of the image, displayed atop image
    lims: tuple
      The upper and lower limits of values for colormap
    
    Returns
    -------
    image : holoviews.Image
      Interactive image with zoom, pan, hover options and cropping tool
    cropping_stream : holoviews.stream
      Holoviews stream containing x and y coordinates of drawn box
    
    Notes
    -----
    
    """
    
    image = hv.Image((np.arange(array.shape[1]),np.arange(array.shape[0]),array))
    
    lims = (0,int(np.percentile(array,99.99))) if lims==None else lims
    image.opts(
        width = int(array.shape[1]),
        height = int(array.shape[0]),
        invert_yaxis = True,
        title = title,
        clim = lims,
        cmap='gray',
        tools=['hover'])

    box = hv.Polygons([])
    box.opts(alpha=.25)
    cropping_stream = streams.BoxEdit(source=box,num_objects=1)   
    
    return image*box, cropping_stream



def get_cropcoords(img, cropping_stream):
    """Given 2d image and cropping stream from fx crop_img,
    provide coordinates to use for subsequent slicing.
    
    Arguments
    ---------
    array : numpy array
      2-dimensional array
    cropping_stream : holoviews.stream
      Holoviews stream containing x and y coordinates of drawn box
    
    Returns
    -------
    coords : numpy array
      2d array.  coords[0] holds y axis cropping and coords[1]
      holds x axis cropping
    
    Notes
    -----
    
    """
    if len(cropping_stream.data['x0']) is not 0:
        ys = sorted([cropping_stream.data['y0'][0], cropping_stream.data['y1'][0]])
        xs = sorted([cropping_stream.data['x0'][0], cropping_stream.data['x1'][0]])
    else:
        ys = [0, img.shape[0]]
        xs = [0, img.shape[1]]
    coords = np.array([ys,xs]).astype('int')
    return coords