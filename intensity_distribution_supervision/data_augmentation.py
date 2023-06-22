"""
Author : Seung Yeon Shin
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
"""

""" Functions used for 'dataset.py' """


import numpy as np
import scipy.ndimage


def rotate(img_list, order_list, rot_angle, rot_plane=(0,1)):
    """
    img_list : list of images to transform with the same transformation
    order_list : order of interpolation
    rot_angle : rotation angle in degree
    rot_plane : rotation is performed in this plane, tuple
    """
    
    num_img = len(img_list)
            
    res_img_list = []
    for i in range(num_img):
        cval = 0 if order_list[i]==0 else -1024
        res_img = scipy.ndimage.rotate(img_list[i], rot_angle, rot_plane, \
                                       reshape=False, order=order_list[i], mode="constant", cval=cval)
        res_img_list.append(res_img)

    return res_img_list


""" function for elastic deformation, which supports freely locating control points """
""" based on https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py """
def elastic_transform(img_list, ctrl_pts, order_list, alpha=15, sigma=3):
    """
    img_list : list of images to transform with the same transformation
    ctrl_pts : positions of control points, N*d
    order_list : order of interpolation
    alpha : scaling factor for the deformation
    sigma : smooting factor
    
    """
    
    num_img = len(img_list)
    shape = img_list[0].shape
    
    def make_sparse_field(shape, ctrl_pts):
        field = np.zeros(shape)
        num_ctrl_pts = ctrl_pts.shape[0]
        field_vals = np.random.randn(num_ctrl_pts)
        
        field[tuple(ctrl_pts[:,0]), tuple(ctrl_pts[:,1]), tuple(ctrl_pts[:,2])] = field_vals
        
        """for i in range(num_ctrl_pts):
            xx, yy, zz = ctrl_pts[i]
            field[xx, yy, zz] = field_vals[i]"""
        
        return field
    
    # smooth the field
    dx = scipy.ndimage.gaussian_filter(make_sparse_field(shape, ctrl_pts), sigma, mode="constant", cval=0) * alpha
    dy = scipy.ndimage.gaussian_filter(make_sparse_field(shape, ctrl_pts), sigma, mode="constant", cval=0) * alpha
    if len(shape)==2:
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = x+dx, y+dy
    
    elif len(shape)==3:
        dz = scipy.ndimage.gaussian_filter(make_sparse_field(shape, ctrl_pts), sigma, mode="constant", cval=0) * alpha
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = x+dx, y+dy, z+dz
        
    else:
        raise ValueError("can't deform because the image is not either 2D or 3D")
        
    res_img_list = []
    for i in range(num_img):
        res_img = scipy.ndimage.map_coordinates(img_list[i], indices, order=order_list[i]).reshape(shape)
        res_img_list.append(res_img)

    return res_img_list