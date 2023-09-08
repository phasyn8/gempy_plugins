import enum
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class NearestSurfacePointsSearcher(enum.Enum):
    KNN = 1
    RADIUS = 2


def select_nearest_surfaces_points(surface_points_xyz: np.ndarray, searchcrit: Optional[int|float] = 3,
                                      search_type: NearestSurfacePointsSearcher = NearestSurfacePointsSearcher.KNN
                                   ) -> np.ndarray:
    match search_type:
        case NearestSurfacePointsSearcher.KNN:
            Tree = NearestNeighbors(n_neighbors=searchcrit)
            Tree.fit(surface_points_xyz)
            neighbours_surfaces = Tree.kneighbors(surface_points_xyz, n_neighbors=searchcrit, return_distance=False)
        case NearestSurfacePointsSearcher.RADIUS:
            Tree = NearestNeighbors(radius=searchcrit)
            Tree.fit(surface_points_xyz)
            neighbours_surfaces = Tree.radius_neighbors(surface_points_xyz, radius=searchcrit, return_distance=False)
        case _:
            raise ValueError(f"Invalid search type: {search_type}")
        
    return neighbours_surfaces


def _select_nearest_surfaces_points(geo_model, surface_points, searchcrit):
    """
    Find the neighbour points of the same surface
    by given radius (radius-search) or fix number (knn).
    
    Parameters
    ----------
    geo_model : geo_model
        GemPy-model.
    surface_points: Pandas-dataframe
        Contains the dataframe of the (point-)data from the GemPy-model.
    searchcrit : int or float
        if is int: uses knn-search.
        if is float: uses radius-search.
    """

    # extract surface names
    surfaces = np.unique(surface_points['surface'])
    neighbours = []
    # for each surface
    if isinstance(searchcrit, int):  # in case knn-search
        searchcrit = searchcrit + 1  # because the point itself is also found
        for s in range(surfaces.size):
            # extract point-ids
            i_surfaces = surface_points['surface'] == surfaces[s]
            # extract point coordinates
            p_surfaces = surface_points[i_surfaces][['X', 'Y', 'Z']]
            # create search-tree
            Tree = NearestNeighbors(n_neighbors=searchcrit)
            # add data to tree
            Tree.fit(p_surfaces)
            # find neighbours
            neighbours_surfaces = Tree.kneighbors(p_surfaces, n_neighbors=searchcrit,
                                                  return_distance=False)
            # add neighbours with initial index to total list
            for n in neighbours_surfaces:
                neighbours.append(p_surfaces.index[n])
    else:  # in case radius-search
        for s in range(surfaces.size):
            # extract point-ids
            i_surfaces = surface_points['surface'] == surfaces[s]
            # extract point coordinates
            p_surfaces = surface_points[i_surfaces][['X', 'Y', 'Z']]
            # create search-tree
            Tree = NearestNeighbors(radius=searchcrit)
            # add data to tree
            Tree.fit(p_surfaces)
            # find neighbours (attention: relativ index!)
            neighbours_surfaces = Tree.radius_neighbors(p_surfaces,
                                                        radius=searchcrit,
                                                        return_distance=False)
            # add neighbours with initial index to total list
            for n in neighbours_surfaces:
                neighbours.append(p_surfaces.index[n])
    return neighbours


def set_orientation_from_neighbours(geo_model, neighbours):
    """
    Calculates the orientation of one point with its neighbour points
    of the same surface.
    Parameters
    ----------
    geo_model : geo_model
        GemPy-model.
    neighbours : Int64Index
        point-neighbours-id, first id is the point itself.
    """

    # compute normal vector for the point
    if neighbours.size > 2:
        # extract point coordinates
        coo = geo_model._surface_points.df.loc[neighbours][['X', 'Y', 'Z']]
        # calculates covariance matrix
        cov = np.cov(coo.T)
        # calculate normalized normal vector
        normvec = normalize(np.cross(cov[0].T, cov[1].T).reshape(1, -1))[0]
        # check orientation of normal vector (has to be oriented to sky)
        if normvec[2] < 0:
            normvec = normvec * (-1)
        # append to the GemPy-model
        geo_model.add_orientations(geo_model._surface_points.df['X'][neighbours[0]],
                                   geo_model._surface_points.df['Y'][neighbours[0]],
                                   geo_model._surface_points.df['Z'][neighbours[0]],
                                   geo_model._surface_points.df['surface'][neighbours[0]],
                                   normvec.tolist())
    # if computation is impossible set normal vector to default orientation
    else:
        print("orientation calculation of point" + str(neighbours[0]) + "is impossible")
        print("-> default vector is set [0,0,1]")
        geo_model.add_orientations(geo_model._surface_points.df['X'][neighbours[0]],
                                   geo_model._surface_points.df['Y'][neighbours[0]],
                                   geo_model._surface_points.df['Z'][neighbours[0]],
                                   geo_model._surface_points.df['surface'][neighbours[0]],
                                   orientation=[0, 0, 1])
    return geo_model._orientations


def set_orientation_from_neighbours_all(geo_model, neighbours):
    """
    Calculates the orientations for all points with given neighbours.
    Parameters
    ----------
    geo_model : geo_model
        GemPy-model.
    neighbours : list of Int64Index
        point-neighbours-IDs, the first id is the id of the point
        for which the orientation is calculated.
    """

    # compute normal vector for the points
    for n in neighbours:
        set_orientation_from_neighbours(geo_model, n)

    return geo_model._orientations
