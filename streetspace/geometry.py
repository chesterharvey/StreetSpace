##############################################################################
# Module: geometry.py
# Description: Functions to manipulate Shapely geometries.
# License: MIT
##############################################################################

import shapely as sh
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import osmnx as ox
import mplleaflet
import random
import matplotlib
import math
from shapely.ops import linemerge
from shapely.geometry import (Point, MultiPoint, LineString, MultiLineString,
    Polygon, MultiPolygon, GeometryCollection)
from math import radians, cos, sin, asin, sqrt, ceil
from geopandas import GeoDataFrame
from rtree import index
from itertools import cycle, groupby
from pprint import pprint
from time import time
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree  
from warnings import warn

from .utils import *

def vertices_to_points(geometry):
    """Convert vertices of a Shapely LineString or Polygon into points.

    Parameters
    ----------
    geometry : :class:`shapely.geometry.LineString`
        LineString whose vertices will be converted to Points.
   
    Returns
    -------
    :obj:`list`
        List of :class:`shapely.geometry.Point`.
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
    """Extend a LineString at either end.

    Extensions will follow the same azimuth as the endmost segment(s).

    Parameters
    ----------
    linestring : :class:`shapely.geometry.LineString`
        LineString to extend
    extend_dist : :obj:`float`
        Distance to extend
    ends : :obj:`str`, optional, default = ``'both'``
        * ``'both'`` : Extend from both ends
        * ``'start'`` : Extend from start only
        * ``'end'`` : Extend from end only
    
    Returns
    -------
    :class:`shapely.geometry.LineString`
        Extended LineString
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


def shorten_line(linestring, shorten_dist, ends = 'both'):
    """Shorten a LineString at either end.

    Parameters
    ----------
    linestring : :class:`shapely.geometry.LineString`
        LineString to extend
    shorten_dist : :obj:`float`
        Distance to shorten
    ends : :obj:`str`, optional, default = ``'both'``
        * ``'both'`` : Shorten from both ends
        * ``'start'`` : Shorten from start only
        * ``'end'`` : Shorten from end only
    
    Returns
    -------
    :class:`shapely.geometry.LineString`
        Shortened LineString
    """
    if ends == 'both':
        start = linestring.interpolate(shorten_dist)
        end = linestring.interpolate(linestring.length - shorten_dist)
    elif ends == 'start':
        start = linestring.interpolate(shorten_dist)
        end = endpoints(linestring)[1]
    elif ends == 'end':
        start = endpoints(linestring)[0]
        end = linestring.interpolate(linestring.length - shorten_dist)
    return segment(linestring, start, end)


def split_line_at_points(linestring, points):
    """Split a LineString into segments defined by Points along it.

    Adapted from: https://stackoverflow.com/questions/34754777/shapely-split
    -linestrings-at-intersections-with-other-linestrings

    Parameters
    ----------
    linestring : :class:`shapely.geometry.LineString`
        LineString to split

    points : :obj:`list`
        Must contain :class:`shapely.geometry.Point`

    Returns
    ----------
    :obj:`list`
        Segments as :class:`shapely.geometry.LineString`
    """

    # get original coordinates of line
    coords = list(linestring.coords)
    # break off last coordinate in case the first/last are the same (loop)
    last_coord = coords[-1]
    coords = coords[0:-1]
    # make a list identifying which coordinates will be segment endpoints
    cuts = [0] * len(coords)
    cuts[0] = 1     
    # add the coords from the cut points
    coords += [list(p.coords)[0] for p in points]    
    cuts += [1] * len(points)
    # calculate the distance along the linestring for each coordinate
    dists = [linestring.project(Point(p)) for p in coords]
    # sort the coords/cuts axd on the distances
    coords = [p for (d, p) in sorted(zip(dists, coords))]
    cuts = [p for (d, p) in sorted(zip(dists, cuts))]
    # add back last coordinate
    coords = coords + [last_coord]
    cuts = cuts + [1]
    # generate the Lines      
    linestrings = []
    for i in range(len(coords)-1):           
        if cuts[i] == 1:    
            # find next element in cuts == 1 starting from index i + 1   
            j = cuts.index(1, i + 1)    
            linestrings.append(LineString(coords[i:j+1]))
    return linestrings

def split_line_at_intersection(linestring, split_linestring):
    """Split one LineString at its points of intersection with another LineString.

    Parameters
    ----------
    linestring : :class:`shapely.geometry.LineString`
        LineString to split

    split_linestring : :class:`shapely.geometry.LineString`
        LineString to split by

    Returns
    ----------
    :obj:`list`
        Segments as :class:`shapely.geometry.LineString`
    """
    points = linestring.intersection(split_linestring)
    if isinstance(points, Point):
        points = [points]
    else:
        points = [x for x in points]
    return split_line_at_points(linestring, points)


def split_line_at_dists(linestring, dists):
    """Split a LineString into segments defined by distances along it.

    Parameters
    ----------
    linestring : :class:`shapely.geometry.LineString`
        LineString to split

    dists : :obj:`list`
        Must contain distances as :obj:`float`

    Returns
    ----------
    :obj:`list`
        Segments as :class:`shapely.geometry.LineString`
    """
    points = [linestring.interpolate(x) for x in dists]
    return split_line_at_points(linestring, points)


def segment(linestring, u, v):
    """Extract a LineString segment defined by two Points along it.

    The order of u and v specifies the directionality of the returned
    LineString. Directionality is not inhereted from the original LineString.

    Parameters
    ----------
    linestring : :class:`shapely.geometry.LineString`
        LineString from which to extract segment
    u : :class:`shapely.geometry.Point`
        Segment start point
    v : :class:`shapely.geometry.Point`
        Segment end point

    Returns
    ----------
    :class:`shapely.geometry.LineString`
        Segment of `linestring`
    """
    segment = split_line_at_points(linestring, [u, v])[1]
    # See if the beginning of the segment aligns with u
    if endpoints(segment)[0].equals(u):
        return segment
    # Otherwise, flip the line direction so it matches the order of u -> v
    else:
        return LineString(np.flip(np.array(segment), 0))
    return LineString(np.flip(np.array(segment), 0)) 


def closest_point_along_lines(search_point, lines, search_distance=None, 
    sindex=None):
    """Find the closest position along multiple lines.

    Parameters
    ----------
    search_point : :class:`shapely.geometry.Point`
        Point from which to search
    lines : :obj:`list` 
        Lines to search. Must be either a list of linestrings or a list of\
        (index, linestring) tuples, if linestrings have predifined indices
    search_distance : :obj:`float`, optional, default = ``None``
        Maximum distance to search from the `search_point`
    sindex : :class:`rtree.index.Index`, optional, default = ``None``
        Spatial index for list of lines (best created with ``list_sindex``)
    
    Returns
    -------
    match_point : :class:`shapely.geometry.Point`
        Location of closest point
    index : :obj:`tuple`
        Index for the closest line
    distance : :obj:`float`
        Distance from the search_point to the closest line
    """
    # Check whether lines are formatted as a list of tuples, with indices
    # in the first positions and geometries in the second positions
    
    tuples = True
    for line in lines:
        if not isinstance(line, tuple):
            tuples = False

    # If not indexed tuples, create them
    if tuples is False:
        line_tuples = [(i, x) for i, x in enumerate(lines)]
    else:
        line_tuples = lines
        
    # Pare down lines, if spatial index provided
    if sindex:
        if not search_distance:
            raise ValueError('Must specify search_distance if using spatial index')
        # Construct search bounds around the search point
        search_bounds = search_point.buffer(search_distance).bounds
        # Get indices for lines within search bounds
        line_indices = [i for i in sindex.intersection(search_bounds, 
                                                       objects='raw')]
        # Get pared lines
        line_tuples = [line_tuples[i] for i in line_indices]
    
    # Pare down lines, if only search distance provided
    elif search_distance:
        # Construct search bounds around the search point
        search_area = search_point.buffer(search_distance)
        # Get pared IDs
        line_tuples = [line_tuple for line_tuple in line_tuples if 
                       line_tuple[-1].intersects(search_area)]
   
    # Calculate the distance between the search point and each line   
    distances = []   

    for _, line in line_tuples:
        distances.append(search_point.distance(line))    
    
    if len(distances) > 0:

        if len(distances) == 1:

            i, line = line_tuples[0]
            distance = distances[0]
        
        elif len(distances) > 1:
            # Find closest line
            line_tuples = np.asarray(line_tuples, dtype='object')
            distances = np.asarray(distances, dtype='float')
            distance_array = np.column_stack((distances, line_tuples))
            distance_array = distance_array[distance_array[:,0].argsort()]
            distance = distance_array[0,0]
            i = distance_array[0,1]
            line = distance_array[0,2]

        # Find the nearest point along that line
        match_point = line.interpolate(line.project(search_point))
        return match_point, i, distance
    
    # If no lines within search distance, return nothing
    else:
        return None, None, None

def closest_point_along_line_vectorized(point, line_start, line_end, constrain_to_segment=True):
    '''Find coordinates for the point along a line segment that is closest to the input point
    
    adapted from: https://stackoverflow.com/questions/28931007/how-to-find-the-closest-point-on-a-line-segment-to-an-arbitrary-point
    
    All required inputs are 2D coordinate tuples (x,y); x and y may be floats or arrays.
    
    If constrain_to_segment is True, the closest point is constrained between the defined start and end of the line.
    If False, the line is treated as an infitinitely long vector extending beyond the defined ends and the closest point may be outside these bounds.
    '''
    x1, y1 = line_start
    x2, y2 = line_end
    x3, y3 = point
    dx = x2 - x1
    dy = y2 - y1
    d2 = dx*dx + dy*dy
    nx = ((x3-x1)*dx + (y3-y1)*dy) / d2
    if constrain_to_segment:
        if isinstance(nx, float):
            nx = min(1, max(0, nx))          
        else:
            nx = np.clip(nx, 0, 1)
    return (dx*nx + x1, dy*nx + y1)

def list_sindex(geometries):
    """Create a spatial index for a list of geometries.

    Parameters
    ----------
    geometries : :obj:`list`
        List of :class:`shapely.geometry.Point`,\
        :class:`shapely.geometry.MultiPoint`,\
        :class:`shapely.geometry.LineString`,\
        :class:`shapely.geometry.MultiLineString`,\
        :class:`shapely.geometry.Polygon`,\
        :class:`shapely.geometry.MultiPolygon` or\
        :class:`shapely.geometry.collection.GeometryCollection`

    Returns
    ----------
    :class:`rtree.index.Index`
        Spatial index
    """
    idx = index.Index()
    for i, geom in enumerate(geometries):
        idx.insert(i, geom.bounds, obj=i)
    return idx


def spaced_points_along_line(linestring, spacing, centered=False, return_lin_refs=False):
    """Create equally spaced points along a Shapely LineString.

    If a list of LineStrings is entered, the function will construct points
    along each LineString but will return all points together in the same
    list.

    Parameters
    ----------
    linestring : :class:`shapely.geometry.LineString` or :obj:`list`
        If list, must contain only :class:`shapely.geometry.LineString` objects.
    spacing : :obj:`float`
        Spacing for points along the `linestring`.
    centered : :obj:`bool` or :obj:`str`, optional, default = ``False``
        * ``False``: Points/Spaces aligned with the start of the `linestring`.
        * ``'point' or True``: Points aligned with the midpoint of the `linestring`.
        * ``'space'``: Spaces aligned with the midpoint of the `linestring`.
    lin_refs : :obj:`bool`, optional, default = ``False``
        * ``False``: Linear references of points will not be returned
        * ``True``: Linear references of points will be returned

    Returns
    ----------
    if lin_refs=False:
        :obj:`list`
            List of points
    if lin_refs=True:
        :obj:`list`
            List of points
        :obj:`list`
            List of linear references for points (floats)
    """
    if isinstance(linestring, LineString):
        linestring = [linestring] # If only one LineString, make into list
    all_lin_refs = []
    all_points = []
    for l, line in enumerate(linestring):
        lin_refs = []
        points = []
        length = line.length
        for p in range(int(ceil(length/spacing))):
            if centered == False:
                starting_point = 0
            elif centered in ['point', True]:
                half_length = length / 2
                starting_point = (
                    half_length - ((half_length // spacing) * spacing))
            elif centered == 'space':
                # Space the starting point from the end so the points are
                # centered on the edge
                starting_point = (length - (length // spacing) * spacing) / 2
            lin_ref = starting_point + (p * spacing)
            x, y = line.interpolate(lin_ref).xy
            point = sh.geometry.Point(x[0], y[0])
            # Store point in list
            lin_refs.append(lin_ref)
            points.append(point)
        all_lin_refs.extend(lin_refs)
        all_points.extend(points)
    if not return_lin_refs:
        return all_points
    else:
        return all_points, all_lin_refs


def azimuth(linestring, degrees=True, warning=True):
    """Calculate azimuth between endpoints of a LineString.

    
    ###### WARNING: This function was returning reversed azimuths in an earlier version.
    ###### Code depending on it should be reviewed for logic errors. 


    Parameters
    ----------
    linestring : :class:`shapely.geometry.LineString`
        Azimuth will be calculated between ``linestring`` endpoints.

    degrees : :obj:`bool`, optional, default = ``True``
        * ``True`` for azimuth in degrees.
        * ``False`` for azimuth in radians.

    Returns
    ----------
    :obj:`float`
        Azimuth between the endpoints of the ``linestring``.
    """ 
    if warning:
        warn('A previous version of streetspace.geometry.azimuth returned reverse azimuths 180 degree off')
    u, v = endpoints(linestring)
    azimuth = np.arctan2(v.y - u.y, v.x - u.x)
    if degrees:
        return np.degrees(azimuth)
    else:
        return azimuth


def split_line_at_vertices(linestring):
    """Split a LineString into segments at each of its vertices.

    Parameters
    ----------
    linestring : :class:`shapely.geometry.LineString`
        LineString to split into segments

    Returns
    ----------
    :obj:`list`
        Contains a :class:`shapely.geometry.LineString` for each segment
    """
    coords = list(linestring.coords)
    n_lines = len(coords) - 1
    return [LineString([coords[i],coords[i + 1]]) for i in range(n_lines)]


def endpoints(linestring):
    """Get endpoints of a LineString.

    Parameters
    ----------
    linestring : :class:`shapely.geometry.LineString`
        LineString from which to extract endpoints

    Returns
    ----------
    u : :class:`shapely.geometry.Point`
        Start point
    v : :class:`shapely.geometry.Point`
        End point
    """
    u = Point(linestring.xy[0][0], linestring.xy[1][0])
    v = Point(linestring.xy[0][-1], linestring.xy[1][-1])
    return u, v 


def azimuth_at_distance(linestring, distance, degrees=True):
    """Get the azimuth of a LineString at a certain distance along it.

    Parameters
    ----------
    linestring : :class:`shapely.geometry.LineString`
        LineString along which an azimuth will be calculated.
    distance : :obj:`float`
        Distance along `linestring` at which to calculate azimuth
    degrees: :obj:`bool`, optional, default = ``False``
        * ``True`` : Azimuth calculated in degrees
        * ``False`` : Azimuth calcualted in radians

    Returns
    -------
    :obj:`float`
        Azimuth of `linestring` at specified `distance`
    """
    segments = split_line_at_vertices(linestring)
    segment_lengths = [edge.length for edge in segments]
    cumulative_lengths = []
    for i, length in enumerate(segment_lengths):
        if i == 0:
            cumulative_lengths.append(length)
        else:
            cumulative_lengths.append(length + cumulative_lengths[i-1])
    # Get index of split edge that includes the specified distance by
    # searching the list in reverse order
    for i, length in reversed(list(enumerate(cumulative_lengths))):
        if length >= distance:
            segment_ID = i
    return azimuth(segments[segment_ID], degrees=degrees)          


def line_by_azimuth(start_point, length, azimuth, degrees=True):
    """Construct a LineString axd on a start point, length, and azimuth.

    Parameters
    ----------
    start_point : :class:`shapely.geometry.Point`
        Line start point
    length : :obj:`float`
        Line length
    azimuth : :obj:`float`
        Line aximuth
    degrees : :obj:`bool`, optional, default = ``True``
        * ``True`` : Azimuth specified in degrees
        * ``False`` : Azimuth specified in radians

    Returns
    -------
    :class:`shapely.geometry.LineString`
        Constructed LineString
    """
    if degrees:
        azimuth = np.radians(azimuth)
    vx = start_point.x + np.cos(azimuth) * length
    vy = start_point.y + np.sin(azimuth) * length
    u = Point([start_point.x, start_point.y])
    v = Point([vx, vy])
    return LineString([u, v])


def midpoint(linestring):
    """Get the midpoint of a LineString.

    Parameters
    ----------
    linestring : :class:`shapely.geometry.LineString`
        LineString along which to identify midpoint

    Returns
    -------
    :class:`shapely.geometry.Point`
        Midpoint of `linestring`
    """
    return linestring.interpolate(linestring.length / 2)


def gdf_spaced_points_along_lines(gdf, spacing, centered=False, return_lin_refs=False):
    """Create equally-spaced Points along LineStrings in a GeoDataFrame.
    Attributes in accompanying columns are copied to all children of each
    parent record.
    Parameters
    ----------
    gdf : :class:`geopandas.GeoDataFrame`
        Geometry type must be :class:`shapely.geometry.LineString`
    spacing : :obj:`float`
        Spacing for points along the `linestring`.
    centered : :obj:`bool` or :obj:`str`, optional, default = ``False``
        * ``False``: Points/Spaces aligned with the start of the `linestring`.
        * ``'point' or True``: Points aligned with the midpoint of the `linestring`.
        * ``'space'``: Spaces aligned with the midpoint of the `linestring`.
    lin_refs : :obj:`bool`, optional, default = ``False``
        * ``False``: Linear references of points will not be returned a column
        * ``True``: Linear references of points will be returned
    Returns
    -------
    :class:`geopandas.GeoDataFrame`
    """
    # initiate new dataframe to hold points
    points_gdf = gpd.GeoDataFrame(data=None, columns=gdf.columns, 
                                geometry = 'geometry', crs=gdf.crs)
    for i, line in gdf.iterrows():
        if return_lin_refs:
            points, lin_refs = spaced_points_along_line(
                line['geometry'], 
                spacing, 
                centered = centered,
                return_lin_refs=True)
        else:
            points = spaced_points_along_line(
                line['geometry'], 
                spacing, 
                centered = centered)
        # copy columns from the original geodataframe  
        point_records = gpd.GeoDataFrame(
            data=[line]*len(points), columns=gdf.columns, 
            geometry = 'geometry', crs=gdf.crs)
        # replace the geometry for these copied records with the segment geometry
        point_records['geometry'] = points
        # add lin_ref column if applicable
        if return_lin_refs:
            point_records['lin_ref'] = lin_refs
        # add new points to full list
        points_gdf = points_gdf.append(point_records, ignore_index=True)
    return points_gdf


def gdf_split_lines(gdf, segment_length, centered = False, min_length = 0, return_lin_refs=False):
    """Split LineStrings in a GeoDataFrame into equal-length segments.

    Attributes in accompanying columns are copied to all children of each
    parent record.

    Parameters
    ----------
    gdf : :class:`geopandas.GeoDataFrame`
        Geometry type must be :class:`shapely.geometry.LineString`
    segment_length: :obj:`float`
        Length of segments to create.
    centered : :obj:`bool` or :obj:`str`, optional, default = ``False``
        * ``False`` : Not centered; points are spaced evenly from the start of each LineString 
        * ``'end'`` : A segment end is centered on each LineString
        * ``'segment'`` : A segment is centered on each LinesString

    Returns
    -------
    :class:`geopandas.GeoDataFrame`
    """
    
    if centered == 'end':
        centered = 'point'
    elif centered == 'segment':
        centered = 'space'

    # initiate new dataframe to hold segments
    # segments = gpd.GeoDataFrame(data=None, columns=gdf.columns, 
    #                             geometry = 'geometry', crs=gdf.crs)
    segments = []

    for i, segment in gdf.iterrows():
        points, lin_refs = spaced_points_along_line(segment['geometry'], 
                                          segment_length, 
                                          centered = centered,
                                          return_lin_refs=True)
        points = points[1:] # exclude the starting point
        # cut the segment at each point
        segment_geometries = split_line_at_points(segment['geometry'], points)
        if len(segment_geometries) > 1:
            # merge the end segments less than minimum length
            if segment_geometries[0].length < min_length:
                print(len(segment_geometries))
                segment_geometries[1] = linemerge(MultiLineString(
                    [segment_geometries[0], segment_geometries[1]]))
                segment_geometries = segment_geometries[1:]
            if segment_geometries[-1].length < min_length:
                segment_geometries[-2] = linemerge(MultiLineString(
                    [segment_geometries[-2], segment_geometries[-1]]))
                segment_geometries = segment_geometries[:-1]
        # copy the segment records
        segment_records = gpd.GeoDataFrame(
            data=[segment]*len(segment_geometries), columns=gdf.columns, 
            geometry = 'geometry', crs=gdf.crs)
        if return_lin_refs:
            segment_records['lin_ref'] = lin_refs
        # replace the geometry for these copied records with the segment geometry
        segment_records['geometry'] = segment_geometries
        # add new segments to full list
        # segments = segments.append(segment_records, ignore_index=True)
        segments.append(segment_records)
    
    # return segments
    return pd.concat(segments, axis=0, ignore_index=True)


def gdf_bbox(gdf):
    """Make a bounding box around all geometries in a GeoDataFrame.

    Parameters
    ----------
    gdf : :class:`geopandas.GeoDataFrame`
        GeoDataFrame with geometries around which to define bounding box

    Returns
    -------
    :class:`geopandas.Polygon`
        Bounding box
    """
    bounds = gdf.total_bounds
    return Polygon([(bounds[0], bounds[1]),
                    (bounds[2], bounds[1]),
                    (bounds[2], bounds[3]),
                    (bounds[0], bounds[3])])


def gdf_centroid(gdf):
    """Replace GeoDataFrame geometries with centroids.

    Parameters
    ----------
    gdf : :class:`geopandas.GeoDataFrame`
        GeoDataFrame with LineString or Polygon geometries

    Returns
    -------
    :class:`geopandas.GeoDataFrame`
        GeoDataFrame with original geometies replaced by their centroids
    """
    gdf = gdf.copy()
    centroids = gdf.centroid
    gdf['geometry'] = centroids
    return gdf 


def haversine(lon1, lat1, lon2, lat2, unit = 'km'):
    """Calculate the great circle distance between two lat/lons.

    Adapted from https://stackoverflow.com/questions/4913349

    Parameters
    ----------
    lon1 : :obj:`float` or vector of :obj:`float`
        Longitude of 1st point
    lat1 : :obj:`float` or vector of :obj:`float`
        Latitute of 1st point
    lon2 : :obj:`float` or vector of :obj:`float`
        Longitude of 2nd point
    lat2 : :obj:`float` or vector of :obj:`float`
        Latitude of 2nd point
    unit : :obj:`str`, optional, default = ``'km'``
        * ``'km'`` : Kilometers
        * ``'mi'`` : Miles

    Returns
    -------
    :obj:`float`
        Distance in specified unit
    """
    if unit == 'km':
        r = 6371 # Radius of the earth in km
    elif unit == 'mi':
        r = 3956 # Radius of the earth in mi
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * r


def degrees_centered_at_zero(degrees):
    """Rescale degrees so they are centered at 0.

    Ouputs will range from -180 to 180.
    
    Parameters
    ----------
    degrees : :obj:`float`
        Degrees centered at 180 (e.g., ranging from 0 to 360)

    Returns
    -------
    :obj:`float`
        Degrees centered at 0
    """
    if degrees > 180:
        degrees = degrees - 360
    elif degrees < -180:
        degrees = degrees + 360
    elif degrees == -180:
        degrees = 180
    return degrees



def side_by_relative_angle(angle):
    """Assign side axd on relative angle centered on 0 degrees.

    Negative angles are left. Positive angles are right.
    
    Parameters
    ----------
    degrees : :obj:`float`
        Degrees centered at 180 (e.g., ranging from 0 to 360)

    Returns
    -------
    :obj:`str`
        * ``'L'`` : Left
        * ``'R'`` : Right
        * ``'C'`` : Centered
    """
    if angle < 0:
        return 'R'
    elif angle > 0:
        return 'L'
    else:
        return 'C'

 
def float_overlap(min_a, max_a, min_b, max_b):
    """Get the overlap between two floating point ranges.

    Adapted from https://stackoverflow.com/questions/2953967/built-in-function-for-computing-overlap-in-python

    Parameters
    ----------
    min_a : :obj:`float`
        First range's minimum
    max_a : :obj:`float`
        First range's maximum
    min_b : :obj:`float`
        Second range's minimum
    max_b : :obj:`float`
        Second range's maximum

    Returns
    -------
    :obj:`float`
        Length of overlap between ranges
    """
    return max(0, min(max_a, max_b) - max(min_a, min_b))


def clip_line_by_polygon(line, polygon):
    """Clip a polyline to the portion within a polygon boundary.
    
    Parameters
    ----------
    line : :class:`shapely.geometry.LineString`
        Line to clip
    polygon : :class:`shapely.geometry.Polygon`
        Polygon to clip by

    Returns
    -------
    :class:`shapely.geometry.LineString` or :class:`shapely.geometry.MultiLineString`
        Line segment(s) within the polygon boundary
    """
    if line.intersects(polygon.boundary):
        split_lines = split_line_at_intersection(line, polygon.boundary)
        within_lines = []
        for line in split_lines:
            if shorten_line(line, 1e-6).within(polygon):
                within_lines.append(line)
        if len(within_lines) == 1:
            return within_lines[0]
        else:
            return MultiLineString(within_lines)
    elif shorten_line(line, 1e-6).within(polygon):
        return line
    else:
        return None


def gdf_clip_line_by_polygon(line_gdf, polygon_gdf):
    """Clip a polyline to the portion within a polygon boundary.
    
    Parameters
    ----------
    line_gdf : :class:`geopandas.GeoDataFrame`
        Lines to clip. Geometry type must be :class:`shapely.geometry.LineString`
    polygon_gdf : :class:`geopandas.GeoDataFrame`
        Polygons to clip by. Geometry type must be :class:`shapely.geometry.Polygon`
    Returns
    -------
    :class:`geopandas.GeoDataFrame`
        Line segments within the polygons
    """
    line_gdf = line_gdf.copy()
    line_gdf['line_index'] = line_gdf.index
    line_columns = list(line_gdf.columns)
    line_columns.remove('geometry')
    polygon_gdf = polygon_gdf.copy()
    polygon_gdf['polygon_index'] = polygon_gdf.index
    polygon_columns = list(polygon_gdf.columns)
    polygon_columns.remove('geometry')
    output_columns = line_columns + polygon_columns + ['geometry']
    clip_gdf = gpd.GeoDataFrame(columns=output_columns, geometry='geometry', crs=line_gdf.crs)
    for polygon in polygon_gdf.itertuples():
        for line in line_gdf.itertuples():
            clipped = clip_line_by_polygon(line.geometry, polygon.geometry)
            if clipped is not None:
                polygon_dict = polygon._asdict()
                for x in ['geometry', 'Index']:
                    polygon_dict.pop(x, None)
                line_dict = line._asdict()
                for x in ['geometry', 'Index']:
                    line_dict.pop(x, None)
                new_dict = {**line_dict, **polygon_dict}
                if isinstance(clipped, LineString):
                    new_dict['geometry'] = clipped
                    new_gdf_row = gpd.GeoDataFrame([new_dict], geometry='geometry', crs=line_gdf.crs)
                    clip_gdf = pd.concat([clip_gdf, new_gdf_row])
                elif isinstance(clipped, MultiLineString):
                    for line in clipped:
                        new_dict['geometry'] = line
                        new_gdf_row = gpd.GeoDataFrame([new_dict], geometry='geometry', crs=line_gdf.crs)
                        clip_gdf = pd.concat([clip_gdf, new_gdf_row])
    clip_gdf = df_first_column(clip_gdf, 'line_index')
    clip_gdf = df_first_column(clip_gdf, 'polygon_index')
    clip_gdf = df_last_column(clip_gdf, 'geometry')
    return clip_gdf


def lines_polygons_intersection(lines, polygons, polygons_sindex=None, singlepart=False):
    """Finds intersection of all lines with all polygons.
    
    Parameters
    ----------
    lines : LineString or MultiLineString GeoDataFrame
    polygons : Polygon GeoDataFram
    polygons_sindex : Spatial index for polygons

    Returns
    -------
    LineString or MultiLineString GeoDataFrame
    """
    # Create a spatial index for polygons if not supplied
    if polygons_sindex is None:
        polygons_sindex = polygons.sindex
    # Initiate a new geodataframe to store results
    results = GeoDataFrame(columns = ['polygon_id'] + list(lines.columns))
    # Iterate through lines
    for line in lines.itertuples():
        # Convert line record to a mutable dictionary
        possible_matches = polygons.iloc[list(
            polygons_sindex.intersection(line.geometry.bounds))]
        for polygon in possible_matches.itertuples():
            # print(polygon)
            intersection = line.geometry.intersection(polygon.geometry)
            if isinstance(intersection, (LineString, MultiLineString)):
                # If input was MultiLineString, keep as MultiLineString
                if (type(lines.geometry[0]) == MultiLineString) and not singlepart:
                    # And intersection is only LineString
                    if isinstance(intersection, LineString):
                        # Convert to MultiLineString
                        instersection = MultiLineString([intersection])
                # If explicitly singlepart, convert MultiLineStrings to LineStrings
                if singlepart:
                    if isinstance(intersection, MultiLineString):
                        intersection = [x for x in intersection]
                # Make intersection into a list if not already
                intersection = listify(intersection)
                # Add row to results for each element in intersection
                for x in intersection:
                    # Make a copy of the row
                    result_line = line._asdict()
                    # Replace its geometry with the intersection
                    result_line['geometry'] = x
                    # Add a field for the polygon id
                    result_line['polygon_id'] = polygon.Index
                    # Rename line id field
                    result_line['line_id'] = line.Index
                    result_line.pop('Index')
                    # Append it to the results
                    results = results.append(result_line, ignore_index=True)

    # Ensure that indices are stored as integers
    results['line_id'] = pd.Series(results['line_id'], dtype='int32')
    results['polygon_id'] = pd.Series(results['polygon_id'], dtype='int32')
    # Rearrange columns
    results = df_first_column(results, 'polygon_id')
    results = df_first_column(results, 'line_id')
    results = df_last_column(results, 'geometry')
    return results


def lines_polygons_difference(lines, polygons, polygons_sindex=None):
    """Finds intersection of all lines with all polygons.
    
    Parameters
    ----------
    lines : LineString or MultiLineString GeoDataFrame
    polygons : Polygon GeoDataFram
    polygons_sindex : Spatial index for polygons

    Returns
    -------
    LineString or MultiLineString GeoDataFrame
    """
    # Create a spatial index for polygons if not supplied
    if polygons_sindex is None:
        polygons_sindex = polygons.sindex
    # Initiate a new geodataframe to store results
    results = GeoDataFrame(columns = ['polygon_id'] + list(lines.columns))
    # Iterate through lines
    for line in lines.itertuples():
        # Convert line record to a mutable dictionary
        possible_matches = polygons.iloc[list(
            polygons_sindex.intersection(line.geometry.bounds))]
        for polygon in possible_matches.itertuples():
            # print(polygon)
            intersection = line.geometry.difference(polygon.geometry)
            if isinstance(intersection, (LineString, MultiLineString)):
                # If input was MultiLineString
                if type(lines.geometry[0]) == MultiLineString:
                    # And intersection is only LineString
                    if isinstance(intersection, LineString):
                        # Convert to MultiLineString
                        instersection = MultiLineString([intersection])
                # Make a copy of the row
                result_line = line._asdict()
                # Replace its geometry with the intersection
                result_line['geometry'] = intersection
                # Add a field for the polygon id
                result_line['polygon_id'] = polygon.t
                # Rename line id field
                result_line['line_id'] = line.Index
                result_line.pop('Index')
                # Append it to the results
                results = results.append(result_line, ignore_index=True)
    # Ensure that indices are stored as integers
    results['line_id'] = pd.Series(results['line_id'], dtype='int32')
    results['polygon_id'] = pd.Series(results['polygon_id'], dtype='int32')
    # Rearrange columns
    results = df_first_column(results, 'polygon_id')
    results = df_first_column(results, 'line_id')
    results = df_last_column(results, 'geometry')
    return results
    

def shape_to_gdf(shape, crs=None):
    """Convert one or more shapes to a geodataframe.

    Parameters
    ----------
    shape : list or Shapely geometry
        List of geometries to be converted into a geodataframe. If a single\
        geometry, the returned geodataframe will have one row.

    Returns
    -------
    :class:`geopandas.GeoDataFrame`
        Geodataframe with geometries in the 'geometry' column.
    """
    # if just one shape, put it in a list
    if isinstance(shape, (Point, MultiPoint, LineString, MultiLineString, 
                          Polygon, MultiPolygon)):
        shape = [shape]
    return gpd.GeoDataFrame(geometry=shape, crs=crs)

def label_features(axis, gdf, label_column, offset, **kwargs):
    """Label features plotted from a geodataframe.

    """
    
    # shift midpoint randomly +/- 25% of edge length
    def get_label_point(feature, jitter = None):
        """Calculate point for labeling a feature

        ``jitter`` is the proportion of a feature's extent within which to randomly
        jitter label position relative to the feature's centroid of midpoint.

        """
        if isinstance(feature, LineString):
            mid_dist = feature.length / 2
            if jitter:
                jitter_range = feature.length * jitter
                mid_dist = mid_dist + random.uniform(-jitter_range, jitter_range)
            return feature.interpolate(mid_dist)

        else:
            label_point = feature.centroid
            if jitter:
                minx, miny, maxx, maxy = feature.bounds
                jitter_range = max([(maxx-minx),(maxy-miny)]) * jitter
                x = label_point.x + random.uniform(-jitter_range, jitter_range)
                y = label_point.y + random.uniform(-jitter_range, jitter_range)
            else:
                x = label_point.x
                y = label_point.y
            return Point(x,y)

    gdf.apply(lambda edge: axis.annotate(s=edge[label_column], 
        xy=(get_label_point(edge.geometry, jitter = None).x + offset[0], 
            get_label_point(edge.geometry, jitter = None).y + offset[1]), 
        **kwargs), axis=1)

def plot_shapes(shapes, ax=None, axis_off=True, size=8, extent=None, 
    legend=None, base_shapes=None, leaflet=False):
    """Plot multiple shapes.

    Parameters
    ----------
    shapes : single Shapely geometry, list of geometries, list of lists of geometries, GeoDataFrame, or list of GeoDataFrames
        * a single geometry will be plotted by itself
        * each geometry or list of geometries within a list will be plotted in a seperate color
        * all records in a GeoDataFrame will be plotted in the same color
        * tuples like (geom, {'color':'color'}) may be passed to specify colors\
        and other attributes
        * default color order is: brgcmyk
        * colors will be repeated as necessary

    ax : predifined axis object, optional, default = ``None``
        If specified, shapes will be plotted on this axis

    axis_off : :obj:`bool`, optional, default = ``False``
        * ``True`` : plot will omit axis markings
        * ``False`` : plot will include axis markings

    size : :obj:`int`, optional, default = ``8``
        Square size of returned plot (used for both length and width)

    extent : :obj:`tuple`, optional, default = ``None``
        * (minx, miny, maxx, maxy)
        * If specified, restricts axis to specific extent.

    legend : :obj:`list`, optional, default = ``None``
        List with the same length and order as ``shapes``

    base_shapes : GeoDataFrame
        If specified, extents of `base_shapes` will be clipped to the maximum extent of `shapes` 
        and plotted as the bottom layer in grey 

    leaflet : :obj:`bool`, optional, default = ``False`` (deprecated)
        * ``True`` : will plot in leaflet in a new browser window
        * ``False`` : will plot normally in-line
    """

    # Make sure shapes are in a list
    shapes = listify(shapes.copy())

    # Turn all individual shapes and lists of shapes into geodataframes
    attribute_dicts = [None] * len(shapes)
    for i, shape in enumerate(shapes):
        # If shapes are specified as tuples with attributes, break out attributes
        if isinstance(shape, tuple):
            shape, attribute_dicts[i] = shape
        # If individual shape, make into sublist
        if isinstance(shape, (Point, MultiPoint, LineString, 
                              MultiLineString, Polygon, MultiPolygon)):
            shape = [shape]
        # If list, make into geodataframe
        if isinstance(shape, list):
            shape = shape_to_gdf(shape)
        # Save manipulated shape back to shapes list
        shapes[i] = shape
   
    # Collect each attribute
    colors = [None] * len(shapes)
    alphas = [1] * len(shapes)
    labels = [None] * len(shapes)
    legend_entries = [None] * len(shapes)

    if any(attribute_dicts):
        for i, attribute_dict in enumerate(attribute_dicts):
            if attribute_dict:
                if 'color' in attribute_dict:
                    colors[i] = attribute_dict['color']
                if 'alpha' in attribute_dict:
                    alphas[i] = attribute_dict['alpha']   
                if 'label' in attribute_dict:
                    labels[i] = attribute_dict['label']
                if 'legend' in attribute_dict:
                    legend_entries[i] = attribute_dict['legend']
   
    # Combine default and custom colors
    default_colors = cycle(list('brgcmyk'))
    colors = [color if color else next(default_colors) for 
              color in colors]
   
    # Reverse the lists so the first one draws last
    shapes = list(reversed(shapes))
    colors = list(reversed(colors))
    alphas = list(reversed(alphas))
    labels = list(reversed(labels))
       
    # If labeling, compute maximum extent to enable label placement
    if any(labels):
        bboxes = []
        for i, shape in enumerate(shapes):
            bboxes.append(sh.geometry.box(*tuple(shape.total_bounds)))
        minx, miny, maxx, maxy = shape_to_gdf(bboxes).total_bounds
        max_extent = max([(maxx-minx),(maxy-miny)])
        offset = max_extent / 120

    # Set up axis
    if not ax:
        if not isinstance(size, tuple):
            size = (size, size)
        fig, ax = plt.subplots(1, figsize=size)

    # Plot base shapes if specified
    if base_shapes is not None:
        if isinstance(base_shapes, list):
            base_shapes = gpd.GeoDataFrame(geometry=base_shapes)
        # Calculate bounding box of primary shapes
        bbox = gdf_bbox(gpd.GeoDataFrame(geometry=[geom for shape in shapes for geom in shape.geometry]))
        # Get intersection of bbox with base shapes
        base_shapes = gpd.overlay(base_shapes, gpd.GeoDataFrame(geometry=[bbox], crs=base_shapes.crs))
        # Plot the base shapes
        base_shapes.plot(ax=ax, color='#ECECEC')

    # Plot shapes
    for i, shape in enumerate(shapes):
        shape.plot(ax=ax, color=colors[i], alpha=alphas[i])
        if labels[i]:
            if not labels[i] in shape.columns:
                shape[labels[i]] = labels[i]
            label_features(
                ax, shape, labels[i], (offset,offset), color=colors[i], ha='left')

    # 
    # first_shape = shapes[0]
    # first_shape.plot(ax=ax, color=colors[0], alpha=alphas[0])
    # if labels[0]:
    #     if not labels[0] in first_shape.columns:
    #         first_shape[labels[0]] = labels[0]
    #     label_features(
    #         ax, first_shape, labels[0], (offset,offset), color=colors[0], ha='left')

    # # plot remaining shapes
    # if len(shapes) > 0:
    #     remaining_shapes = shapes[1:]
    #     for i, shape in enumerate(remaining_shapes):
    #         shape.plot(ax=ax, color=colors[i+1], alpha=alphas[i+1])
    #         if labels[i+1]:
    #             if not labels[i+1] in shape.columns:
    #                 shape[labels[i+1]] = labels[i+1]
    #             label_features(
    #                 ax, shape, labels[i+1], (offset,offset), color=colors[i+1], ha='left')
    
    if leaflet:
        mplleaflet.show(fig=fig, crs=shapes[0].crs, tiles='cartodb_positron')       

        if extent:
            ax.axis('equal')
            minx, miny, maxx, maxy = extent
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            if axis_off:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
        
        else:
            ax.axis('equal')
            if axis_off:
                ax.axis('off')

        if any(legend_entries):
            # Get list of layers
            layers = ax.collections
            # Reverse list
            layers = layers[::-1]

            # Keep only layers with non-None legend entries
            layers = [layer for layer, entry 
                      in zip(layers, legend_entries) if entry]

            # Keep only non-None legend entries
            legend_entries = [entry for entry in legend_entries if entry]
            
            # legend_handles, _ = ax.get_legend_handles_labels()
            ax.legend(layers, legend_entries)

    try:
        return fig, ax
    except:
        pass

def intersect_shapes(shapes_a, shapes_b, shapes_b_sindex=None):
    """Find intersections between shapes in two lists

    Parameters
    ----------
    shapes_a : list of Shapely geometries
        List of geometries to be intersected with those in shapes_b
    shapes_b : list of Shapely geometries
        List of geometries to be intersected with those in shapes_a
    shapes_b_sindex : :class:`rtree.index.Index`, optional, default = ``None``
        Spatial index for shapes_b (best created with ``list_sindex``)

    Returns
    -------
    :obj:`list`
        List of tuples for each intersection with structure:\
        (a_index, b_index, intersection_geometry)
    """
    intersections = []
    for i, shape_a in enumerate(shapes_a):
        indices_b = list(range(len(shapes_b)))
        if shapes_b_sindex:
            b_for_analysis = [(indices_b[i], shapes_b[i]) for i in 
                              shapes_b_sindex.intersection(shape_a.bounds)]#, objects='raw')]
        else:
            b_for_analysis = zip(indices_b, shapes_b)       
        for j, shape_b in b_for_analysis:
            if shape_a.intersects(shape_b):
                intersection = shape_a.intersection(shape_b)
                intersections.append((i, j, intersection))
    return intersections


def normalize_azimuth(azimuth, zero_center=False):
    """Normalize an azimuth in degrees so it falls between 0 and 360.
    
    If ``zero_center=True``, azimuth will be normalized
    between -180 and 180.
    """
    if (azimuth > 360 or azimuth < 0):
        azimuth %= 360
    if zero_center:
        if azimuth > 180:
            azimuth -= 360
    return azimuth


def normalize_azimuth_array(azimuths, zero_center=False):
    """Normalize an array of azimuths in degrees so they falls between 0 and 360
    
    ``azimuths`` should be a NumPy array or eqivalent (Pandas Series or DataFrame)
    
    If ``zero_center=True``, azimuths will be normalized
    between -180 and 180.
    """
    azimuths = azimuths.copy()
    azimuths = np.where((azimuths > 360) | (azimuths < 0) , azimuths % 360, azimuths) 
    if zero_center:
        azimuths = np.where(azimuths > 180, azimuths - 360, azimuths)
    return azimuths      


def azimuth_difference(azimuth_a, azimuth_b, directional=True):
    """Find the difference between two azimuths specifed in degrees.
       
    If ``directional=True`` (default), will produce a difference 
    between 0 and 180 degrees that ignores sign but accounts for
    inverted differences in orientation.
    
    If ``directional=False`` or ``directional='inverse'``, will ingore 
    inverted differences in rotation by also calculating the difference 
    if one azimuth is rotated 180 degrees and returning the smaller of 
    the two differences.
    
    If ``directional='polar'``, will produce a difference between
    0 and 360 degrees, accounting for differences past 180 degrees.
    
    If ``directional='signed'``, will produce a difference between -180
    and 180, accounting for the sign of the difference.
    """
    
    def unsigned_difference(a, b):
        difference = a-b
        if difference > 180:
            difference -= 360
        if difference < -180:
            difference += 360
        return abs(difference)
    
    if directional is True:  
        azimuth_a = normalize_azimuth(azimuth_a, zero_center=True)
        azimuth_b = normalize_azimuth(azimuth_b, zero_center=True)
        return unsigned_difference(azimuth_a, azimuth_b)
    
    elif (directional is False) or (directional == 'inverse'):
        azimuth_a = normalize_azimuth(azimuth_a, zero_center=True)
        azimuth_b = normalize_azimuth(azimuth_b, zero_center=True)
        return min(
            [unsigned_difference(azimuth_a, azimuth_b), 
             unsigned_difference(azimuth_a + 180, azimuth_b)])
    
    elif directional == 'polar':        
        return normalize_azimuth((azimuth_b - azimuth_a))
    
    elif directional == 'signed':        
        azimuth_a = normalize_azimuth(azimuth_a)
        azimuth_b = normalize_azimuth(azimuth_b)
        return azimuth_b - azimuth_a


def closest_point_along_line(point, line, return_linear_reference=False):
    """Return the point along a line that is closest to another point.

    ``point`` must be a Shapely Point.

    ``line`` must be a Shapely LineString.
    """
    lin_ref = line.project(point)
    point = line.interpolate(lin_ref)
    if return_linear_reference:
        return point, lin_ref
    else:
        return point


def vertices_to_points(shape):
    """Return vertices of a shape as a list of points.

    ``shape`` must be a Shapely geometry. 
    """
    return [Point(coords) for coords in np.array(shape)]


def directed_hausdorff(a, b):
    """Calculate the directed Hausdorff distance from shape a to shape b.

    ``a`` and ``b`` must be Shapely geometries
    """
    a_nodes = vertices_to_points(a)
    b_match_points = [closest_point_along_line(node, b) for node in a_nodes]
    dists = [a_node.distance(b_point) for a_node, b_point in zip(a_nodes, b_match_points)]
    return max(dists)


def gdf_intersecting_polygon(gdf, polygon, gdf_sindex=None, quadrat_size=None):
    """

    """
    if not gdf_sindex:
        gdf_sindex = gdf.sindex
    if not quadrat_size:
        minx, miny, maxx, maxy = polygon.bounds
        max_dimension = max([maxx-minx, maxy-miny])
        quadrat_size = max_dimension / 10
    gdf['unique_identifier'] = gdf.index
    polygon_cut = ox.quadrat_cut_geometry(polygon, quadrat_width=2500)
    # Find the points that intersect with each subpolygon and add them to points_within_geometry
    selection = pd.DataFrame()
    for poly in polygon_cut:
        # Buffer by the <1 micron dist to account for any space lost in the quadrat cutting
        # otherwise may miss point(s) that lay directly on quadrat line
        poly = poly.buffer(1e-14).buffer(0)
        # Find approximate matches with r-tree, then precise matches from those approximate ones
        possible_matches_index = list(gdf_sindex.intersection(poly.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(poly)]
        selection = selection.append(precise_matches)
    if len(selection) > 0:
        # Drop duplicate rows intersecting with multiple polygons
        selection = selection.drop_duplicates(subset=['unique_identifier'])
        selection = selection.drop(columns=['unique_identifier'])
        return selection


def major_axis_azimuth(polygon):
    """Calculate the azimuth of a LineString's or Polygon's major axis

    """
    rectangle = polygon.minimum_rotated_rectangle
    if isinstance(rectangle, LineString):
        longest_sides = [rectangle]
    # elif isinstance(rectangle, Point):
    #     return 0
    else:
        try:
            sides = split_line_at_vertices(rectangle.boundary)
            lengths = [x.length for x in sides]
            longest_sides = [side for side, length 
                in zip(sides, lengths) if length == max(lengths)]
        except:
            print(type(polygon), polygon, type(rectangle), rectangle)
    azimuths = [azimuth(x) for x in longest_sides]    
    return max(azimuths)


def major_minor_axes(shape, azimuths=False):

    # Get minimum bounding rectangle
    min_rectangle = shape.minimum_rotated_rectangle
    
    # Synthesize sides if minimum polygon is a linestring
    if isinstance(min_rectangle, LineString):
        # Synthesize longest sides
        longest_sides = [min_rectangle, min_rectangle]
        # Synthesize shortest sides
        start, end = endpoints(test)
        shortest_sides = [LingString(start, start), LineString(end, end)]
        
    else:
        # Split rectangle into polyline sides
        sides = split_line_at_vertices(min_rectangle.boundary)

        # If minimum polygon is a square, major and minor axes are equal
        if all([sides[0].length == sides[x].length for x in range(1,4)]):
            # Arbitrarily select shorter and longer pairs or sides
            longest_sides = [sides[0], sides[2]]
            shortest_sides = [sides[1], sides[3]]

        # Otherwise, figure out which sides are shortest and longest
        else:
            lengths = [x.length for x in sides]
            # longest_sides = [side for side, length 
            #     in zip(sides, lengths) if length == max(lengths)]
            # shortest_sides = [side for side, length 
            #     in zip(sides, lengths) if length == min(lengths)]
            sides_by_length = [side for _, side in sorted(zip(lengths, sides), key=lambda x: x[0])]
            shortest_sides = sides_by_length[:2]
            longest_sides = sides_by_length[-2:]
            
    # Get azimuths for longest and shortest sides (if applicable)
    if azimuths:
        major_azimuth = max([azimuth(x) for x in longest_sides])
        if max([x.length for x in shortest_sides]) > 0:        
            minor_azimuth = max([azimuth(x) for x in shortest_sides])
        else:
            minor_azimuth = None
        return major_azimuth, minor_azimuth
    
    # Find major axis
    shortest_side_midpoints = [midpoint(x) for x in shortest_sides]
    
    try:
        major_axis = LineString(shortest_side_midpoints)
    except:
        print(shortest_sides)
    
    # Find minor axis
    longest_side_midpoints = [midpoint(x) for x in longest_sides]
    minor_axis = LineString(longest_side_midpoints)
    
    return major_axis, minor_axis


def remove_invalid_geometries(gdf):
    """Remove GeoDataFrame rows with non-standard geometries.

    """
    gdf=gdf.copy()
    geom_types = (Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon)
    for row in gdf.itertuples():
        if not isinstance(row.geometry, geom_types):
            gdf.drop([row.Index], inplace=True)
    return gdf


def singlepart(gdf):
    """Convert GeoDataFrame geometries to singlepart.

    """
    gdf = gdf.copy()
    # Make a static version of the index
    static_index = 'index'
    while static_index in gdf.columns:
        static_index = static_index + '_'
    gdf[static_index] = gdf.index
    # Look through each row to see if geometry is multipart
    for row in gdf.itertuples():
        if isinstance(row.geometry, (MultiPoint, MultiLineString, MultiPolygon)):
            # Divide into individual shapes
            shapes = [x for x in row.geometry]
            # Make a new row for each individual shape
            for shape in shapes:
                new_row = row._asdict()
                new_row['geometry'] = shape
                new_row.pop('Index', None)
                gdf = gdf.append(new_row, ignore_index=True)
            # Get current index of row based on the static index
            current_index = gdf.loc[gdf[static_index] == row._asdict()[static_index]].index[0]
            # Drop original row
            gdf.drop(current_index, inplace=True)
    return gdf


def nearest_neighbor(shape, gdf, hausdorff_distance=False):
    """Identify the nearest neighbor to a shape among features in a geodataframe
    
    """
    distances = []
    if hausdorff_distance:
        for feature in gdf.itertuples():
            # Minimum of directed hausdorff distances in both directions
            dist_a = directed_hausdorff(feature.geometry, shape)
            dist_b = directed_hausdorff(shape, feature.geometry)
            distances.append(min([dist_a, dist_b]))
    else:
        for feature in gdf.itertuples():
            distances.append(shape.distance(feature.geometry))
    return gdf.iloc[[np.argmin(distances)]]


def merge_ordered_lines(lines):
    """Merge lines together in a specified order, filling gaps between line ends
    
    Always creates a continuous LineString.
    
    To merge without filling gaps (e.g., pruducing a MultiLineString), use shapely.ops.linemerge
        
    """
    # Collapse line coordinates into a single list
    coords = [pair for pairs in lines for pair in zip(*pairs.coords.xy)]
    
    # Remove sequentially-redundant coordinates
    coords = [k for k, g in groupby(coords)]
    
    # Construct a line from these coordinates
    merged_line = LineString(coords)
    
    return merged_line


def generate_points_within_polygon(polygon, n):
    """Generate n random points within a polygon.    
    """
    points = []
    
    # Get maximum bounds of polygon
    x_min, y_min, x_max, y_max = polygon.bounds

    while len(points) < n:

        # Draw random coordinate values within these extents
        x = np.random.uniform(x_min, x_max, n * 2)
        y = np.random.uniform(y_min, y_max, n * 2)

        # Construct points from these values
        new_points = [sh.geometry.Point(x,y) for x, y in zip(x, y)]
        new_points = gpd.GeoDataFrame(geometry=new_points)

        # Only keep points within the polygon
        new_points = gdf_intersecting_polygon(new_points, polygon, quadrat_size=500)

        points.extend(new_points['geometry'].tolist())

    if len(points) > n:
        points = points[:n]
        
    return points


def construct_hexagons(startx, starty, endx, endy, radius):
        """ 
        Calculate a grid of hexagon coordinates of the given radius
        given lower-left and upper-right coordinates 
        Returns a list of lists containing 6 tuples of x, y point coordinates
        These can be used to construct valid regular hexagonal polygons

        You will probably want to use projected coordinates for this

        from: https://gist.github.com/urschrei/17cf0be92ca90a244a91
        """
        # calculate side length given radius   
        sl = (2 * radius) * math.tan(math.pi / 6)
        # calculate radius for a given side-length
        # (a * (math.cos(math.pi / 6) / math.sin(math.pi / 6)) / 2)
        # see http://www.calculatorsoup.com/calculators/geometry-plane/polygon.php

        # calculate coordinates of the hexagon points
        # sin(30)
        p = sl * 0.5
        b = sl * math.cos(math.radians(30))
        w = b * 2
        h = 2 * sl

        # offset start and end coordinates by hex widths and heights to guarantee coverage     
        startx = startx - w
        starty = starty - h
        endx = endx + w
        endy = endy + h

        origx = startx
        origy = starty


        # offsets for moving along and up rows
        xoffset = b
        yoffset = 3 * p

        polygons = []
        row = 1
        counter = 0

        while starty < endy:
            if row % 2 == 0:
                startx = origx + xoffset
            else:
                startx = origx
            while startx < endx:
                p1x = startx
                p1y = starty + p
                p2x = startx
                p2y = starty + (3 * p)
                p3x = startx + b
                p3y = starty + h
                p4x = startx + w
                p4y = starty + (3 * p)
                p5x = startx + w
                p5y = starty + p
                p6x = startx + b
                p6y = starty
                poly = [
                    (p1x, p1y),
                    (p2x, p2y),
                    (p3x, p3y),
                    (p4x, p4y),
                    (p5x, p5y),
                    (p6x, p6y),
                    (p1x, p1y)]
                polygons.append(poly)
                counter += 1
                startx += w
            starty += yoffset
            row += 1
        return polygons


def hexagon_grid(gdf, radius):
    """Create mesh of hexagons with a specific radius across the same extent as a gdf

    """
    # get bounds of supplied geodataframe
    minx, miny, maxx, maxy = gdf.total_bounds

    # calculate coordinates for hexagons
    hex_coords = construct_hexagons(minx, miny, maxx, maxy, radius)

    # construct hexagon polygons
    hexagons = [sh.geometry.Polygon(coords) for coords in hex_coords]

    # convert to geodataframe
    hexagons = gpd.GeoDataFrame(geometry=hexagons, crs=gdf.crs) 

    return hexagons


def merge_multilinestring(multilinestring, tolerance):
    """Function to merge all linestrings making up a multilinestring.
    
    Connects linestrings with endpoints that are within the `tolerance` distance of one another.
    
    Automatically flips the direction of connecting linestrings so they are consistent.
    
    If all linestrings are connectable within the tolerance, returns a single linestring
    
    If not all linestrings are connectable, returns MultiLineString made up of both connected
        and unconnected linestrings.
    """
    
    def _find_similar_endpoints(i_edge, j_edge, tolerance):
        # Iterate through combinations of endpoints           
        i_endpoints = zip(('u','v'), endpoints(i_edge))
        j_endpoints = zip(('u','v'), endpoints(j_edge))
        for i_end, i_point in i_endpoints:
            for j_end, j_point in j_endpoints:
                # See if the points are the same
                if i_point.distance(j_point) <= tolerance:
                    # If the lines are headed the same way into their shared vertex, flip one of them
                    if i_end == 'v' and j_end == 'u':
                        merged_edge = merge_ordered_lines([i_edge, j_edge])
                    elif j_end == 'v' and i_end == 'u':
                        merged_edge = merge_ordered_lines([j_edge, i_edge])
                    elif i_end == 'v' and j_end == 'v':
                        # Flip j
                        j_edge = reverse_linestring(j_edge)
                        merged_edge = merge_ordered_lines([i_edge, j_edge])
                    elif i_end == 'u' and j_end == 'u':
                        # Flip i
                        i_edge = reverse_linestring(i_edge)
                        merged_edge = merge_ordered_lines([i_edge, j_edge])
                    # Return the merged line
                    return merged_edge
        # Otherwise, return nothing
        return None

    def _merge_with_connecting_edge(remaining_edges, tolerance):
        first_edge = remaining_edges[0]
        # If there are a lot of remaining edges, use a spatial index
        if len(remaining_edges) > 4:
            # Initiate an rtree index to find nearby edges
            idx = index.Index()
            # Make a spatial index of remaining edges
            idx = index.Index()
            for i, edge in enumerate(remaining_edges):
                idx.insert(i, edge.bounds)
            # Identify edges that are nearby the first edge
            nearby_edges = set(idx.intersection(first_edge.buffer(tolerance * 1.1).bounds))
        # Otherwise, just list all the edge indices
        else:
            nearby_edges = set(range(len(remaining_edges)))
        # Remove the first edge from the list (don't want to connect it to itself)
        nearby_edges.discard(0)
        # Iterate through the nearby edges until if finds something
        for i in nearby_edges: 
            # Try to find similar endpoints with the first edge
            merged_edge = _find_similar_endpoints(
                first_edge, remaining_edges[i], tolerance)
            if merged_edge:
                return merged_edge, i
        # If nothing connects, return nothing
        return None, None
    
    # Explode multilinestring into individual linestring edges
    remaining_edges = [edge for edge in multilinestring]   
    standalone_edges = []
    # Iterate through edges while there is still more than one edge
    while len(remaining_edges) > 0:           
        # Merge the first remaining edge with a connecting edge
        merged_edge, i = _merge_with_connecting_edge(remaining_edges, tolerance) 
        if merged_edge:
            # Replace the first edge with the new merged edge
            remaining_edges[0] = merged_edge
            # Remove the other original edge
            remaining_edges.pop(i)
        else:
            # If nothing merged, move original edge to the list of standalone edges
            standalone_edges.append(remaining_edges[0])
            remaining_edges.pop(0)
    # Return either a multilinestring or a single linestring
    if len(standalone_edges) > 1:
        return MultiLineString(standalone_edges)
    else:
        return standalone_edges[0]


def reverse_linestring(linestring):
    """Reverses the direction of shapely linestring
    """
    return LineString(linestring.coords[::-1])


def standardize_geometry_column(gdf, current_geom_column=None):
    """Converts the name of a gdf's geometry column to the standard 'geometry'
    """
    if not current_geom_column:
        current_geom_column = gdf.geometry.name
    geometry = gdf[current_geom_column].tolist()
    crs = gdf.crs
    gdf = gdf.drop(columns=[current_geom_column])
    gdf = gpd.GeoDataFrame(gdf, geometry=geometry, crs=crs)
    return gdf


def quadrat_cut_gdf(gdf, width):
    """Split polygon geometries in a gdf into `width`-sized quadrats
    """
    # Initiate a new dataframe for storing splits
    split_rows = []
    
    # Iterate through the original geometries to split them
    for row in gdf.itertuples():
        # Convert row to a dictionary, so it's mutable
        row = row._asdict()
        # Split the geometry
        split_geometry = ox.quadrat_cut_geometry(row['geometry'], quadrat_width=width)
        # Convert to a list
        split_geometry = [x for x in split_geometry]
        for geometry in split_geometry:
            _row = row.copy()
            _row['geometry'] = geometry
            split_rows.append(_row) 
    crs = gdf.crs
    gdf = gpd.GeoDataFrame(split_rows, geometry='geometry', crs=crs)
    # Change name of index column so it doesn't interfere with another pass through itertuples
    gdf = gdf.rename(columns={'Index':'orig_index'})
    return gdf


def identify_nearest_points(gdf_a, gdf_b, b_column=None, dist_as_int=True, merge_original=False):
    """Identify the nearest point in `gdf_b` for each point in `gdf_a`.

    Value in `b_column` is reported for each row in `gdf_a`.

    Adapted from https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe
    """   
    nA = np.array(list(zip(gdf_a.geometry.x, gdf_a.geometry.y)) )
    nB = np.array(list(zip(gdf_b.geometry.x, gdf_b.geometry.y)) )
    btree = cKDTree(nB)
    dist, idx = btree.query(nA,k=1)
    if dist_as_int:
        dist = dist.astype(int)
    cols = {'distance': dist}
    if b_column:
        cols[b_column] = gdf_b.loc[idx, b_column].values

    df = pd.DataFrame.from_dict(cols)
    if merge_original:
        df = gdf_a.merge(df, left_index=True, right_index=True)
    # Return as a series if there's only one column
    if len(df.columns) == 1:
        return df.distance
    else:
        return df


def aerial_count_interpolation(source_gdf, count_field, dest_gdf):
    """Interpolate counts from `source_gdf` to `dest_gdf` polygons
    
    `source_gdf` : geodataframe with polygon geometries and a numeric field with counts
    `count_field` : field in `source_gdf` with counts
    `dest_gdf` : geodataframe with polygon geometries into which counts will be interpolated

    The function interpolates counts based on the proportion of overlap between `source_gdf` and
    `dest_gdf` polygons.

    """
    # Specify IDs for each input dataframe
    source_gdf['ai_source_index'] = range(0, len(source_gdf))
    dest_gdf['ai_dest_index'] = range(0, len(dest_gdf))
    # Calculate areas within the source dataframe
    source_gdf['ai_source_area'] = source_gdf.geometry.area
    # Union the shapes
    union_df = gpd.overlay(source_gdf, dest_gdf, how='union')
    # Calculate areas within the unioned parts
    union_df['ai_union_area'] = union_df.geometry.area
    # Estimate count for each unioned part
    union_df['ai_union_count'] = union_df[count_field] / union_df['ai_source_area'] * union_df['ai_union_area']
    # Sum up parts for destination ids
    result_df = union_df.groupby('ai_dest_index').agg({'ai_union_count':sum})
    # Add results back onto destination dataframe
    dest_gdf = gpd.GeoDataFrame(dest_gdf.merge(result_df, on='ai_dest_index'), geometry='geometry', crs=dest_gdf.crs)
    # Rename and drop columns
    dest_gdf[count_field] = dest_gdf['ai_union_count']
    dest_gdf = dest_gdf.drop(columns=['ai_dest_index','ai_union_count'])
    return dest_gdf


def gdf_3d_to_2d(gdf):
    '''Convert a geodataframe with 3D polygons or linestrings to 2D geometries

    Based on https://gist.github.com/rmania/8c88377a5c902dfbc134795a7af538d8

    TO-DO: Add point and multipoint conversion
    '''
    gdf = gdf.copy()
    geometry = gdf.geometry
    new_geo = []
    for p in geometry:
        if p.has_z:
            if p.geom_type == 'Polygon':
                lines = [xy[:2] for xy in list(p.exterior.coords)]
                new = Polygon(lines)
                new_geo.append(new)
            elif p.geom_type == 'MultiPolygon':
                new_multi_p = []
                for ap in p:
                    lines = [xy[:2] for xy in list(ap.exterior.coords)]
                    new = Polygon(lines)
                    new_multi.append(new)
                new_geo.append(MultiPolygon(new_multi_p))
            elif p.geom_type == 'LineString':
                points = [xy[:2] for xy in list(p.coords)]
                new = LineString(points)
                new_geo.append(new)
            elif p.geom_type == 'MultiLineString':
                new_multi_p = []
                for ap in p:
                    points = [xy[:2] for xy in list(ap.coords)]
                    new = LineString(points)
                    new_multi.append(new)
                new_geo.append(MultiLineString(new_multi))
            # elif p.geom_type == 'Point': ####### TO-DO
            # elif p.geom_type == 'MultiPoint': ####### TO-DO
    gdf.geometry = new_geo
    return gdf

def gdf_cast_singlpart_geometry_to_multipart(gdf, geometry_column='geometry'):
    '''Convert any singlepart geometries in a mixed-type geodataframe to multipart.

    Based on https://stackoverflow.com/questions/7571635/fastest-way-to-check-if-a-value-exists-in-a-list
    '''
    gdf = gdf.copy()
    if type(gdf.iloc[0][geometry_column]) in [Point, MultiPoint]:
        gdf[geometry_column] = [MultiPoint([feature]) if type(feature) == Point else feature for feature in gdf[geometry_column]]
    elif type(gdf.iloc[0][geometry_column]) in [LineString, MultiLineString]:
        gdf[geometry_column] = [MultiLineString([feature]) if type(feature) == LineString else feature for feature in gdf[geometry_column]]
    elif type(gdf.iloc[0][geometry_column]) in [Polygon, MultiPolygon]:
        gdf[geometry_column] = [MultiPolygon([feature]) if type(feature) == Polygon else feature for feature in gdf[geometry_column]]
    return gdf


def intersection_of_lines_vectorized(line_a_start, line_a_end, line_b_start, line_b_end, constrain_on_a=True, constrain_on_b=True):
    def line_intersect(a1, a2, b1, b2):
        """
        From https://www.py4u.net/discuss/15536
        """
        T = np.array([[0, -1], [1, 0]])
        da = np.atleast_2d(a2 - a1)
        db = np.atleast_2d(b2 - b1)
        dp = np.atleast_2d(a1 - b1)
        dap = np.dot(da, T)
        denom = np.sum(dap * db, axis=1)
        num = np.sum(dap * dp, axis=1)
        # Ignore dividing by 0 and multiplying by nan
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.atleast_2d(num / denom).T * db + b1

    intersections = line_intersect(line_a_start, line_a_end, line_b_start, line_b_end)
    intersections = pd.DataFrame(intersections).rename(columns={0:'x',1:'y'})

    # Remove cases where the intersection isn't on line a or b
    if constrain_on_a or constrain_on_b:
        # Add line start and end coordinates to the dataframe
        intersections = pd.concat([
            intersections,
            pd.DataFrame(line_a_start, columns=['line_a_start_x', 'line_a_start_y']),
            pd.DataFrame(line_a_end, columns=['line_a_end_x', 'line_a_end_y']),
            pd.DataFrame(line_b_start, columns=['line_b_start_x', 'line_b_start_y']),
            pd.DataFrame(line_b_end, columns=['line_b_end_x', 'line_b_end_y']),
        ], axis=1)

        if constrain_on_a:
            intersections = intersections[
                (intersections.x >= intersections[['line_a_start_x','line_a_end_x']].min(axis=1)) & 
                (intersections.x <= intersections[['line_a_start_x','line_a_end_x']].max(axis=1)) & 
                (intersections.y >= intersections[['line_a_start_y','line_a_end_y']].min(axis=1)) & 
                (intersections.y <= intersections[['line_a_start_y','line_a_end_y']].max(axis=1))].copy()

        if constrain_on_b:
            intersections = intersections[
                (intersections.x >= intersections[['line_b_start_x','line_b_end_x']].min(axis=1)) & 
                (intersections.x <= intersections[['line_b_start_x','line_b_end_x']].max(axis=1)) & 
                (intersections.y >= intersections[['line_b_start_y','line_b_end_y']].min(axis=1)) & 
                (intersections.y <= intersections[['line_b_start_y','line_b_end_y']].max(axis=1))].copy()

    return intersections[['x','y']]
            

def get_nearest(src_points, candidates, k_neighbors):
    """
    Find nearest neighbors for all source points from a set of candidate points
    Adapted from https://stackoverflow.com/questions/62198199/k-nearest-points-from-two-dataframes-with-geopandas
    """

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15)

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)
   
    return (indices, distances)


def nearest_neighbor(left_gdf, right_gdf, k_neighbors=1, return_left_columns=True, return_right_columns=True):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.
    Adapted from https://stackoverflow.com/questions/62198199/k-nearest-points-from-two-dataframes-with-geopandas
    """
    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)
    
    left_coords = np.array(list(zip(left_gdf[left_gdf.geometry.name].x, left_gdf[left_gdf.geometry.name].y)))
    right_coords = np.array(list(zip(right[right_gdf.geometry.name].x, right[right_gdf.geometry.name].y)))

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_nearest(src_points=left_coords, candidates=right_coords, k_neighbors=k_neighbors)
   
    closest = pd.DataFrame({'right_index':[list(x) for x in closest]}).explode('right_index')
    dist = pd.DataFrame({'right_dist':[list(x) for x in dist]}).explode('right_dist')
    closest = pd.concat([closest, dist], axis=1).reset_index().rename(columns={'index':'left_index'})
    
    if return_left_columns:
        closest = left_gdf.merge(closest, left_index=True, right_on='left_index', how='left').reset_index(drop=True)
    if return_right_columns:
        closest = closest.merge(right.drop(columns=[right_gdf.geometry.name]), left_on='right_index', right_index=True).reset_index(drop=True)
        
    closest = closest.sort_values('left_index').reset_index()
    
    return closest
