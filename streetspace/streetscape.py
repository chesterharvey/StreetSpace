##############################################################################
# Module: streetscape.py
# Description: Functions to analyze streetscapes.
# License: MIT
##############################################################################

import numpy as np

from .geometry import *

def find_nearby_buildings(edge, buildings, buildings_sindex,
    primary_search_distance, secondary_search_distance, vectorized=True):
    """Identify buildings that are nearby a street centerline.

    *Nearby* buildings are identified by (1) excluding all buildings farther
    than the `primary_search_distance` from the `edge`, (2) finding the
    distance *d* to the closest remaining building, (3) then identifying all
    buildings within *d* + `secondary_search_distance` from the edge.

    Parameters
    ----------
    edge : :class:`shapely.geometry.LineString`
        Street centerline from which to search
    buildings : :class:`geopandas.GeoDataFrame`
        * Buildings to search
        * Geometry column must contain :class:`shapely.geometry.Polygon`
    buildings_sindex : :class:`rtree.index.Index`
        Spatial index for `buildings`
    primary_search_distance : :obj:`float`
        Distance from `edge` within which to search for a closest building
    secondary_search_distance : :obj:`float`
        Distance beyond the closest building within which to search for\
        additional buildings
  
    Returns
    -------
    r_search_distance : :obj:`float`
        Distance within which buildings were searched to the right of `edge`
    r_search_area : :class:`shapely.geometry.Polygon`
        Search area to the right of `edge`
    r_buildings : :class:`list`
        Row indices for `buildings` that intersect `r_search_area`
    l_search_distance : :obj:`float`
        Distance within which buildings were searched to the left of `edge`
    l_search_area : :class:`shapely.geometry.Polygon`
        Search area to the left of `edge`
    l_buildings : :class:`list`
        Row indices for `buildings` that intersect `l_search_area`
    """
    # Define search distance, search area, and buildings IDs for a side
    def search_side(side):
        if len(sides[sides == side]) > 0:
            # Get the minimum building distance on this side of the edge
            search_distance = min(building_dists[sides == side])
            # add the secondary search distance to it
            search_distance = search_distance + secondary_search_distance
            # Ensure that the maximum search distance is maintained
            if search_distance > primary_search_distance:
                search_distance = primary_search_distance
            # Make search area
            search_area = edge.buffer(search_distance, cap_style = 2)
            # Get indices for buildings in within search distance
            building_indices = ((sides == side) & 
                                (building_dists <= search_distance))
            buildings = nearby_buildings.index[()]
        else:
            search_distance = np.nan
            search_area = None
            buildings = None
        return search_distance, search_area, buildings

    # Buffer the edge at the primary search distance; flat buffer caps
    buffer = edge.buffer(primary_search_distance, cap_style = 2)
    # Find buildings intersecting the buffer
    possible_matches_index = list(buildings_sindex.intersection(buffer.bounds))
    possible_matches = buildings.iloc[possible_matches_index]
    nearby_buildings = possible_matches[possible_matches.intersects(buffer)]
    # Only further examine if there are any nearby buildings
    if len(nearby_buildings) > 0:
        # Get the distance from each building to the edge
        building_geoms = nearby_buildings['geometry']
        edge_dist = lambda x: x.distance(edge)
        building_dists = edge_dist(building_geoms)
        # Determine the side of the edge each building is on based on the
        # direction to its centroid from the edge
        centroids = nearby_buildings.centroid.tolist()
        lin_ref_dists = [edge.project(x) for x in centroids]
        lin_ref_points = [edge.interpolate(x) for x in lin_ref_dists]
        # Get azimuths between building linear references and centroids      
        centroid_azimuths = [
            azimuth(LineString([lin_ref_points[i],centroids[i]]))
            for i
            in range(len(centroids))]
        edge_azimuths = [azimuth_at_distance(edge, x) for x in lin_ref_dists]
        # Calculate which side each buildings is on
        sides = np.array([
            side_by_relative_angle(degrees_centered_at_zero(
                centroid_azimuths[i] - edge_azimuths[i]))
            for i
            in range(len(centroids))])
        # Get search distance, search area, and building IDs for each side 
        if len(sides[sides == 'R']) > 0:
            r_search_distance, r_search_area, r_buildings = search_side('R')
        if len(sides[sides == 'L']) > 0:
            l_search_distance, l_search_area, l_buildings = search_side('L')
    else:
        r_search_distance = np.nan
        r_search_area = None
        r_buildings = None
        l_search_distance = np.nan
        l_search_area = None
        l_buildings = None    
    return (r_search_distance, r_search_area, r_buildings,
            l_search_distance, l_search_area, l_buildings)




      
# vectorized version
v_find_nearby_buildings = np.vectorize(find_nearby_buildings, 
    otypes=['float64', 'object', np.ndarray, 'float64', 'object', np.ndarray], 
    excluded = [1,2,3,4])