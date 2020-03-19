import numpy as np


def myWarpPerspectiveSparse(src, H, out_size):
    """
    Perform sparse warp perspective operation

    :param      src:       The source image
    :type       src:       Image
    :param      H:         Homography Matrix
    :type       H:         Numpy Matrix
    :param      out_size:  The output image size
    :type       out_size:  Tuple (width,row)

    :returns:   Warped Image
    :rtype:     Image
    """
    output = np.zeros(out_size)
    
    # Get all indices from the src matrix
    row, col = np.indices(src.shape[:2])
    
    # Store as x,y,1
    indices = [(c, r, 1) for r, c in zip(row.ravel(), col.ravel())]
    
    for idx in indices:
        new_idx = np.matmul(H, idx)
        new_idx = new_idx / new_idx[2]
        c = int(round(new_idx[0]))
        r = int(round(new_idx[1]))
        output[r, c,0] = src[idx[1], idx[0],0]
        output[r, c,1] = src[idx[1], idx[0],1]
        output[r, c,2] = src[idx[1], idx[0],2]
            
    return np.uint8(output[:out_size[0], :out_size[1]])


def interpolatePoint(src, r, c):
    """
    Perform interpolation

    :param      src:  The source image
    :type       src:  Image
    :param      r:    Rows
    :type       r:    Int
    :param      c:    Columns
    :type       c:    Int

    :returns:   Interpolated value
    :rtype:     Int
    """
    if (r-1 >= 0 and r+1 < src.shape[0]) and (c-1 >= 0 and c+1 < src.shape[1]):
        return np.mean(src[r-1:r+2, c-1:c+2])


def myWarpPerspective(src, H, out_size, interpolate=False):
    """
    Perform warp perspective operation

    :param      src:       The source image
    :type       src:       Image
    :param      H:         Homography Matrix
    :type       H:         Numpy Matrix
    :param      out_size:  The output image size
    :type       out_size:  Tuple (width,row)

    :returns:   Warped Image
    :rtype:     Image
    """
    output = np.zeros(out_size)
    
    H_inv = np.linalg.inv(H)
    
    # get all indices from the src matrix
    row, col = np.indices(out_size)
    # storing as x,y,1
    indices = [(c, r, 1) for r, c in zip(row.ravel(), col.ravel())]
    for idx in indices:
        src_idx = np.matmul(H_inv, idx)
        src_idx = src_idx / src_idx[2]
        c = int(round(src_idx[0]))
        r = int(round(src_idx[1]))
        
        if interpolate:
            output[idx[1], idx[0]] = interpolatePoint(src, r, c)
        else:
            output[idx[1], idx[0]] = src[r, c]
    
    return output