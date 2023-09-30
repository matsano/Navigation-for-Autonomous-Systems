"""
 Simple 2D ICP implementation
 author: David Filliat
"""

import numpy as np
from scipy.spatial import KDTree
import math


THRESHOLD = 0.05
BEST_MATCHING_RATE = 0.8
MIN_RESOLUTION = 0.05


# A few helper function

def angle_wrap(a):
    """
    Keep angle between -pi and pi
    """
    return np.fmod(a + np.pi, 2*np.pi ) - np.pi


def mean_angle(angleList):
    """
    Compute the mean of a list of angles
    """

    mcos = np.mean(np.cos(angleList))
    msin = np.mean(np.sin(angleList))

    return math.atan2(msin, mcos)


def icp(model, data, maxIter, thres):
    """
    ICP (iterative closest point) algorithm
    Simple ICP implementation for teaching purpose
    - input
    model : scan taken as the reference position
    data : scan to align on the model
    maxIter : maximum number of ICP iterations
    thres : threshold to stop ICP when correction is smaller
    - output
    R : rotation matrix
    t : translation vector
    meandist : mean point distance after convergence
    """

    print('Running ICP, ', end='')

    # Various inits
    olddist = float("inf")  # residual error
    maxRange = 10  # limit on the distance of points used for ICP

    # Create array of x and y coordinates of valid readings for reference scan
    valid = model["ranges"] < maxRange
    ref = np.array([model["x"], model["y"]])
    ref = ref[:, valid]

    # Create array of x and y coordinates of valid readings for processed scan
    valid = data["ranges"] < maxRange
    dat = np.array([data["x"], data["y"]])
    dat = dat[:, valid]

    # ----------------------- TODO ------------------------
    # Filter data points too close to each other
    # Put the result in dat_filt
    last_point = dat[:, 0]
    filtered_data = [last_point]
    for iPoint in range(len(dat[0])):
        actual_point = dat[:, iPoint]
        if np.linalg.norm(actual_point - last_point) > THRESHOLD:
            filtered_data.append(actual_point)
            last_point = actual_point
    
    dat_filt = np.stack(filtered_data, axis=1)
    # dat_filt = dat

    # Initialize transformation to identity
    R = np.eye(2)
    t = np.zeros((2, 1))

    # Main ICP loop
    for iter in range(maxIter):
        ###### Nearest neighbor matching ######
        '''
        # ----- Find nearest Neighbors for each point, using kd-trees for speed
        tree = KDTree(ref.T)
        distance, index = tree.query(dat_filt.T)
        meandist = np.mean(distance)

        # ----------------------- TODO ------------------------
        # filter points matchings, keeping only the closest ones
        # you have to modify :
        # - 'dat_matched' with the points
        # - 'index' with the corresponding point index in ref
        sorted_dist = np.sort(distance)
        filtered_dist_mask = distance <= sorted_dist[int(BEST_MATCHING_RATE*(len(sorted_dist)-1))]
        dat_matched = dat_filt[:, filtered_dist_mask]
        index = index[filtered_dist_mask]
        # dat_matched = dat_filt
        '''
        
        ###### Normal shooting ######
        
        # ----- Find nearest Neighbors for each point , using kd -trees for speed
        tree = KDTree ( ref .T)
        distance , index = tree . query ( dat_filt .T)
        meandist = np . mean ( distance )
        
        # ----- Find association using normal shooting (reusing nearest neighbor for matching filtering)
        normal_idx = np . zeros ( dat_filt . shape [1] , dtype =int)
        normal_cos = np . ones ( dat_filt . shape [1])
        NS = 0
        NN = 0
        for i in range (1 , dat_filt . shape [1] - 1) :
            # computes tangent with points before and after to filter regular areas
            vect_tan_suiv = dat_filt [: , i + 1] - dat_filt [: , i]
            vect_tan_prec = dat_filt [: , i - 1] - dat_filt [: , i]
            
            vect_tan = vect_tan_suiv - vect_tan_prec # mean tangent vector
            cos_tan = abs( np . dot ( vect_tan_suiv , vect_tan_prec ) /( np . linalg . norm (vect_tan_suiv ) * np . linalg . norm ( vect_tan_prec ))) # angle between the two tangents
        
        
            if cos_tan > 0.9: # if we are in a regular area
                if distance [i] > 3 * MIN_RESOLUTION : # if we are far enough to have a good normal shooting
                    cos = np . ones ( ref . shape [1])
                    for j in range(max(0 , index [i ] -20) , min( index [i ]+20 , ref . shape [1]) ): # look for normal shooting around nearest neighbor
                        vect_norm = ref [: , j] - dat_filt [: , i]
                        if np . linalg . norm ( vect_norm ) < 1.3 * distance [i ]:
                            cos [j] = abs( np . dot ( vect_tan , vect_norm ) /( np . linalg . norm (vect_tan ) * np . linalg . norm ( vect_norm ))) # angle between tangent and normal
                            #print(cos[j])
                    normal_idx [i] = np . argmin ( cos )
                    normal_cos [i] = np .min( cos )
                    if normal_cos [i] <= 0.1:
                        NS = NS + 1
                        
                else: # if we are close to the other scan , keep nearest neighbor
                    NN = NN +1
                    normal_idx [i] = index [i]
                    normal_cos [i] = 0
            
            if distance [i] > 2 * meandist or cos_tan <= 0.9:
                # remove matchings with far nearest neighbor and in irregular areas
                normal_idx [i] = 0
                normal_cos [i] = 1.0
                
                
        valid = normal_cos <= 0.1 # keep only matchings close to normal
        dat_matched = np . array ( dat_filt [: , valid ])
        index = normal_idx [ valid ]
        print('Matched with normal shooting : ', NS )
        print('Matched with NN : ', NN )
        print('Valid matchings : ',np .sum( valid ))
        

        # ----- Compute transform

        # Compute point mean
        mdat = np.mean(dat_matched, 1)
        mref = np.mean(ref[:, index], 1)

        # Use SVD for transform computation
        C = np.transpose(dat_matched.T-mdat) @ (ref[:, index].T - mref)
        u, s, vh = np.linalg.svd(C)
        Ri = vh.T @ u.T
        Ti = mref - Ri @ mdat

        # Apply transformation to points
        dat_filt = Ri @ dat_filt
        dat_filt = np.transpose(dat_filt.T + Ti)

        # Update global transformation
        R = Ri @ R
        t = Ri @ t + Ti.reshape(2, 1)

        # Stop when no more progress
        if abs(olddist-meandist) < thres:
            break

        # store mean residual error to check progress
        olddist = meandist

    print("finished with mean point corresp. error {:f}".format(meandist))

    return R, t, meandist