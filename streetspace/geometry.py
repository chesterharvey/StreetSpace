""" Functions to manipulate shapely geometries."""

##############################################################################
# Module: core.py
# Package: streetspace - measure and analyze streetscapes and street networks
# License: MIT
##############################################################################

import shapely as sh
import numpy as np
from shapely.ops import linemerge
from shapely.geometry import (Point, MultiPoint, LineString, MultiLineString,
    Polygon, MultiPolygon, GeometryCollection)
from math import radians, cos, sin, asin, sqrt

def vertices_to_points(geometry):
    """
    Convert vertices from a Shapely LineString or Polygon into a list of
    points.

    Parameters
    ----------
    geometry: Shapely Linestring or Polygon

    Returns
    ----------
    list of Shapely Points

    """
    if isinstance(geometry, Polygon):
        xs, ys = geometry.exterior.coords.xy
        xs = xs[:-1] # Exclude redundant closing vertex
        ys = ys[:-1]
    elif isinstance(geometry, LineString):
        xs, ys = geometry.coords.xy
    points = [sh.geometry.Point(xy[0], xy[1]) for xy in list(zip(xs, ys))]
    return points

def extend_line(linestring, extend_dist, ends='both'):
    """
    Extend a Shapely Linestring at either end along the same aximuth as each
    endmost segment.

    Parameters
    ----------
    linestring: Shapely LineString
        line to extent

    extend_dist: float
        distance to extent in linestring units

    ends: str
        'both' = extends both ends (default)
        'start' = extends from the start of the linestring
        'end' = extends from the end of the linestring

    Returns
    ----------
    Shapely LineString
    """
    
    if ends == 'both':
        endpoints = [sh.geometry.Point(linestring.coords[0]),
                     sh.geometry.Point(linestring.coords[-1])]
        adjacent_points = [sh.geometry.Point(linestring.coords[1]),
                           sh.geometry.Point(linestring.coords[-2])]
    elif ends == 'start':
        endpoints = [sh.geometry.Point(linestring.coords[0])]
        adjacent_points = [sh.geometry.Point(linestring.coords[1])]
    elif ends == 'end':
        endpoints = [sh.geometry.Point(linestring.coords[-1])]
        adjacent_points = [sh.geometry.Point(linestring.coords[-2])]
    # Draw extensions on one or both ends:
    new_segments = []
    for endpoint, adjacent_point in zip(endpoints, adjacent_points):
        # Get the azimuth of the last segment:
        azimuth = np.arctan2(np.subtract(endpoint.x, adjacent_point.x),
                             np.subtract(endpoint.y, adjacent_point.y))
        # Construct a new endpoint along the extension of that segment:
        new_endpoint_x = np.sin(azimuth) * extend_dist + endpoint.x
        new_endpoint_y = np.cos(azimuth) * extend_dist + endpoint.y
        new_endpoint = sh.geometry.Point([new_endpoint_x,new_endpoint_y])
        # Draw a new segment that extends to this new end point:
        new_segments.append(sh.geometry.LineString([endpoint, new_endpoint]))
    # Merge new segments with existing linestring:
    return linemerge([linestring] + new_segments)

def closest_point_among_lines(
    search_pnt, lines, lines_sidx=None, search_dist=None):
    """
    Find the closest point along any of a list of Shapely LineStrings, with or
    without spatial indexing

    Parameters
    ----------
    search_pnt: Shapely Point
        point from which to search

    lines: list of Shapely LineStrings
        lines to search to

    lines_sidx: Rtree Index
        spatial index for lines (default = None)

    search_dist: float
        distance to search from the search_pnt
        (default = None; lines will be assessed no matter their distance)

    Returns
    ----------
    int
        index of the LineString along which the closest point is found
    Shapely Point
        closest point along that LineString
    
    TODO: Would it be easier for the input to this to be a geodataframe?
    That way the spatial index could be constructed inline, if necessary,
    as 'GeoDataFrame.sindex'.
    """
   
    # Get lines within the search distance based a specified spatial index:  
    if lines_sidx != None:
        if search_dist == None:
            raise ValueError('must specify search_dist if using spatial index')
        # construct search area around point
        search_area = search_pnt.buffer(search_dist)
        # get nearby IDs
        find_line_indices = [int(i) for i in
                             lines_sidx.intersection(search_area.bounds)]
        # Get nearby geometries:
        lines = [lines[i] for i in find_line_indices]
    # Get lines within a specified search distance:
    elif search_dist != None:
        # construct search area around point
        search_area = search_pnt.buffer(search_dist)
        # get lines intersecting search area
        lines, find_line_indices = zip(*[(line, i) for i, line in 
                                         enumerate(lines) if
                                         line.intersects(search_area)])
    # Otherwise, get all lines:
    find_line_indices = [i for i, _ in enumerate(lines)]
    # Calculate distances to all remaining lines
    distances = []
    for line in lines:
        distances.append(search_pnt.distance(line))
    # Only return a closest point if there is a line within search distance:
    if len(distances) > 0:
        # find the line index with the minimum distance
        _, line_idx = min((distance, i) for (i, distance) in 
                              zip(find_line_indices, distances))
        # Find the nearest point along that line
        search_line = lines[find_line_indices.index(line_idx)]
        lin_ref = search_line.project(search_pnt)
        closest_point = search_line.interpolate(lin_ref)
        return line_idx, closest_point
    else:
        return None, None

def nearest_point(search_feature, find_feature, search_dist=None,
    use_sidx=False, find_features_sidx=None, return_search_index=False,
    return_find_index=False):
    """
    Find the closest point from the search feature among the find features.

    Parameters
    ----------
    search_feature: Shapely geometry or GeoPandas GeoDataFrame
        feature(s) to search from

    find_feature: Shapely geometry or GeoPandas GeoDataFrame
        feature(s) to search toward

    search_dist: float
        maximum distance to search from the search_feature
        (default = None)

    use_sidx: bool
        whether a spatial index is used to economize potential matches
        (default = False)

    find_features_sidx: Rtree Index
        spatial index for find_feature
        (default = None)
        Note: including a pre-built spatial index object can improve
        efficiency if the same find features are used iteratively

    return_search_index: bool
        return index of search_feature that is closest to find_feature
        (default = False)

    return_find_index: bool
        return index of find_feature that is closest to search_feature
        (default = False)
    """

    

def midpoint(line):
    """
    Get the midpoint of a Shapely LineString

    Parameters
    ----------
    line: Shapely LineString

    Returns
    ----------
    Shapely Point

    """
    return line.interpolate(line.length/2)

def haversine(lon1, lat1, lon2, lat2, unit = 'km'):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)

    Parameters
    ----------
    lon1: float

    lat1: float

    lon2: float

    lat2: float

    unit: str
        'km' = kilometers (default)
        'mi' = miles

    Returns
    ----------
    float
        distance in specified units

    adapted from https://stackoverflow.com/questions/4913349

    """
    if unit == 'km':
        r = 6371 # Radius of the earth in km
    elif unit == 'mi':
        r = 3956 # Radius of the earth in mi
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    return c * r