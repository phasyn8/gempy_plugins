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