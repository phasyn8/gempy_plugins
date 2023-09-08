import gempy as gp
import numpy as np
import os

from orientations_generator import select_nearest_surfaces_points
from orientations_generator._orientations_generator import NearestSurfacePointsSearcher

input_path = os.path.dirname(__file__) + '/../../examples/data'

data_path = os.path.abspath('data')


def _model_factory():
    # Importing the data from CSV-files and setting extent and resolution
    geo_data = gp.create_geomodel(
        extent=[0, 2000, 0, 2000, 0, 2000],
        resolution=[50, 50, 50],
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=f"{data_path}/model5_orientations.csv",
            path_to_surface_points=f"{data_path}/model5_surface_points.csv",
        )
    )
    return geo_data


def test_set_orientations():
    geo_data = _model_factory()

    orientations: gp.data.OrientationsTable = gp.create_orientations_from_surface_points(geo_data.surface_points)

    gp.add_orientations(
        geo_model=geo_data,
        x=orientations.data['X'],
        y=orientations.data['Y'],
        z=orientations.data['Z'],
        pole_vector=orientations.grads,
        elements_names=geo_data.structural_frame.elements_names[0],
    )


def test_select_nearest_surface_points():

    geo_model = _model_factory()
    print(geo_model)
    
    # TODO: This is to select the surface points we want to use to calculate the orientations
    
    element: gp.data.StructuralElement = geo_model.structural_frame.get_element_by_name("fault")

    # find neighbours
    knn = select_nearest_surfaces_points(
        surface_points_xyz=element.surface_points.xyz,
        searchcrit=3
    )
    
    radius = select_nearest_surfaces_points(
        surface_points_xyz=element.surface_points.xyz,
        searchcrit=200.,
        search_type=NearestSurfacePointsSearcher.RADIUS
    )
    
    return knn


def test_set_orientation_from_neighbours():
    
    
    
    
    # find neighbours
    neighbours = gp.select_nearest_surfaces_points(geo_data, fault_poi, 5)
    # calculate one fault orientation
    gp.set_orientation_from_neighbours(geo_data, neighbours[1])
    # find the calculated orientation
    test = geo_data._orientations.df.sort_index().iloc[-1][['dip', 'azimuth']].values

    # calculate reference
    reference = [90 - np.arctan(0.5) / np.pi * 180, 90]

    assert np.array_equal(reference, test)


def test_set_orientation_from_neighbours_all():
    data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
    path_to_data = data_path + "/data/input_data/jan_models/"

    geo_data = gp.create_data_legacy('fault', extent=[0, 1000, 0, 1000, 0, 1000],
                                     resolution=[50, 50, 50],
                                     path_o=path_to_data + "model5_orientations.csv",
                                     path_i=path_to_data + "model5_surface_points.csv")

    # count orientations before orientation calculation
    length_pre = geo_data._orientations.df.shape[0]

    # find neighbours
    neighbours = gp.select_nearest_surfaces_points(geo_data, geo_data._surface_points.df, 2)
    # calculate all fault orientations
    gp.set_orientation_from_neighbours_all(geo_data, neighbours)

    # count orientations after orientation calculation
    length_after = geo_data._orientations.df.shape[0]

    assert np.array_equal(geo_data._surface_points.df.shape[0],
                          length_after - length_pre)
