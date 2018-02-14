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
    """Convert vertices of a Shapely LineString or Polygon into points.

    Parameters
    ----------
    geometry : Shapely LineString or Polygon) :class:`shapely.geometry.linestring.LineString`
   
    Returns:
        list: List of Shapely Points
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
    """Extend a Shapely Linestring at either end.

    Extensions will follow the same azimuth as the endmost segment(s).

    Args:
        linestring (Shapely LineString): line to extend
        extend_dist (float): distance to extend
        ends (str): specifies which end(s) to extend from
            'both' (default) = extends both ends 
            'start' = extends from the start of the LineString
            'end' = extends from the end of the LineString
    
    Returns:
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


def shorten_line(linestring, shorten_dist, ends = 'both'):
    """
    Shorten a Shapely Linestring at either end.

    Parameters
    ----------
    linestring : Shapely LineString
        LineString to extent

    shorten_dist : float
        distance to shorten in LineString units

    ends : str
        'both' = extends both ends (default)
        'start' = extends from the start of the LineString
        'end' = extends from the end of the LineString

    Returns
    ----------
    Shapely LineString
    """
    if ends == 'both':
        start = linestring.interpolate(shorten_dist)
        end = linestring.interpolate(linestring.length - shorten_dist)
    elif ends == 'start':
        start = linestring.interpolate(shorten_dist)
        end = endPoints(linestring)[1]
    elif ends == 'end':
        start = endPoints(linestring)[0]
        end = linestring.interpolate(linestring.length - shorten_dist)
    return segment(linestring, start, end)


def split_line_at_points(linestring, points):
    """
    Split a Shapely LineString into multiple segments defined by Shapely
    Points along it.

    Adapted from: "https://stackoverflow.com/questions/34754777/shapely-split
    -linestrings-at-intersections-with-other-linestrings

    Parameters
    ----------
    linestring : Shapely LineString
        LineString to split

    points : list
        list of Shapely Points

    Returns
    ----------
    list of Shapely LineStrings
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
    # sort the coords/cuts based on the distances
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


def split_line_at_dists(linestring, dists):
    """
    Split a Shapely LineString into multiple segments defined by linear
    distances along it.

    Parameters
    ----------
    linestring : Shapely LineString
        LineString to split

    dists : list
        list of distances along LineString

    Returns
    ----------
    list of Shapely LineStrings
    """

    points = [linestring.interpolate(x) for x in dists]
    return split_line_at_points(linestring, points)


def segment(linestring, u, v):
    """
    Retrieve a segment from a Shapely LineString based on two Shapely Points
    along it.

    The order of u and v specifies the directionality of the returned
    LineString. Directionality is not inhereted from the original LineString.

    Parameters
    ----------
    linestring : Shapely LineString
        LineString from which to extract segment

    u : Shapely Point
        segment start

    v : Shapely Point
        segment end

    Returns
    ----------
    Shapely LineString
    
    """
    segment = split_line_at_points(linestring, [u, v])[1]
    # See if the beginning of the segment aligns with u
    if endPoints(segment)[0].equals(u):
        return segment
    # Otherwise, flip the line direction so it matches the order of u -> v
    else:
        return LineString(np.flip(np.array(segment), 0))
    return LineString(np.flip(np.array(segment), 0)) 


def closest_point_among_lines(search_point, lines, lines_sindex=None, 
    search_distance=None):
    """
    TODO: Would it be easier for the input to this to be a geodataframe?
    That way the spatial index could be constructed inline, if necessary,
    as 'GeoDataFrame.sindex'.

    Find the closest point along any of a list of Shapely LineStrings, with or
    without spatial indexing

    Parameters
    ----------
    search_point : Shapely Point
        point from which to search

    lines : list of Shapely LineStrings
        lines to search to

    lines_sindex : Rtree Index
        spatial index for lines (default = None)

    search_distance : float
        distance to search from the search_point
        (default = None; lines will be assessed no matter their distance)

    Returns
    ----------
    int
        index of the LineString along which the closest point is found
    Shapely Point
        closest point along that LineString
    
    """
   
    # Get lines within the search distance based a specified spatial index:  
    if lines_sidx != None:
        if search_dist == None:
            raise ValueError('must specify search_dist if using spatial index')
        # construct search area around point
        search_area = search_point.buffer(search_dist)
        # get nearby IDs
        find_line_indices = [int(i) for i in
                             lines_sidx.intersection(search_area.bounds)]
        # Get nearby geometries:
        lines = [lines[i] for i in find_line_indices]
    # Get lines within a specified search distance:
    elif search_dist != None:
        # construct search area around point
        search_area = search_point.buffer(search_dist)
        # get lines intersecting search area
        lines, find_line_indices = zip(*[(line, i) for i, line in 
                                         enumerate(lines) if
                                         line.intersects(search_area)])
    # Otherwise, get all lines:
    find_line_indices = [i for i, _ in enumerate(lines)]
    # Calculate distances to all remaining lines
    distances = []
    for line in lines:
        distances.append(search_point.distance(line))
    # Only return a closest point if there is a line within search distance:
    if len(distances) > 0:
        # find the line index with the minimum distance
        _, line_idx = min((distance, i) for (i, distance) in 
                              zip(find_line_indices, distances))
        # Find the nearest point along that line
        search_line = lines[find_line_indices.index(line_idx)]
        lin_ref = search_line.project(search_point)
        closest_point = search_line.interpolate(lin_ref)
        return line_idx, closest_point
    else:
        return None, None


def list_sindex(geom_list):
    """
    Create an rtree spatial index for a list of Shapely geometries

    Parameters
    ----------
    geom_list : list
        list of Shapely geometries

    Returns
    ----------
    rtree Index
    """

    idx = index.Index()
    for i, geom in enumerate(geom_list):
        idx.insert(i, geom.bounds)
    return idx


def points_along_lines(linestrings, spacing, centered = False):
    """Create equally spaced points along Shapely LineStrings.

    Args:
        linestrings (list or Shapely LineString): if list, must include only
            Shapely LineStrings
        spacing (float): spacing for points along the LineString(s)
        centered : bool or str
            False (default) = not centered; points are spaced evenly from the
                start of the LineString 
            'Point' = a point is located at the LineString midpoint
            'Space' = a gap between points is centered on the LinesString
                midpoint

    Returns:
        list: List of Shapely Points
    """
    if isinstance(linestrings, LineString):
        linestrings = [linestrings] # If only one LineString, make into list
    all_points = []
    for l, line in enumerate(linestrings):
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
            x, y = line.interpolate(starting_point + (p * spacing)).xy
            point = sh.geometry.Point(x[0], y[0])
            # Store point in list
            points.append(point)
        all_points.extend(points)
    return all_points


def azimuth(linestring, degrees=True):
    """
    Calculate azimuth between endpoints of a Shapely LineString.

    Parameters
    ----------
    linestring : Shapely LineString

    degrees : bool
        True (default) = azimuth calculated in degrees
        False = azimuth calcualted in radians

    Returns
    ----------
    float

    """ 
    u = endpoints(linestring)[0]
    v = endpoints(linestring)[1]
    azimuth = np.arctan2(u.y - v.y, u.x - v.x)
    if degrees:
        return np.degrees(azimuth)
    else:
        return azimuth


def split_line_at_vertices(linestring):
    """
    Split a Shapely LineString into segments at each of its vertices.

    Parameters
    ----------
    linestring : Shapely LineString

    Returns
    ----------
    list of Shapely LineStrings

    """
    coords = list(linestring.coords)
    n_lines = len(coords) - 1
    return [LineString([coords[i],coords[i + 1]]) for i in range(n_lines)]


def endpoints(linestring):
    """
    Get endpoints of a Shapely LineString

    Parameters
    ----------
    linestring : Shapely LineString

    Returns
    ----------
    Shapely Point, Shapely Point
        LineString start, LineString end

    """
    u = Point(linestring.xy[0][0], linestring.xy[1][0])
    v = Point(linestring.xy[0][-1], linestring.xy[1][-1])
    return u, v 


def azimuth_along_line(linestring, distance, degrees=True):
    """
    Get the aximuth of a Shapely LineString at a certain distance along it.

    Parameters
    ----------
    linestring : Shapely LineString

    degrees: bool
        True (default) = azimuth calculated in degrees
        False = azimuth calcualted in radians

    Returns
    ----------
    float

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


def line_by_azimuth(start_point, distance, azimuth, degrees=True):
    """
    Construct a Shapely LineString based on a starting point, distance, and 
    azimuth.

    Parameters
    ----------
    start_point : Shapely Point

    distance : float

    azimuth : float

    degrees : bool
        True (default) = azimuth specified in degrees
        False = azimuth specified in radians

    Returns
    ----------
    Shapely LineString

    """
    if degrees:
        azimuth = np.radians(azimuth)
    vx = start_point.x + np.cos(azimuth) * distance
    vy = start_point.y + np.sin(azimuth) * distance
    u = Point([start_point.x, start_point.y])
    v = Point([vx, vy])
    return LineString([u, v])


def midpoint(linestring):
    """
    Get the midpoint of a Shapely LineString.

    Parameters
    ----------
    line : Shapely LineString

    Returns
    ----------
    Shapely Point

    """
    return linestring.interpolate(linestring.length / 2)


def closest_network_point(G, search_point, search_distance,
    geometry='geometry', edges_sindex=None):
    """
    Find the closest point along the edges of a NetworkX graph with Shapely 
    LineString geometry attributes in the same coordinate system.

    Parameters
    ----------
    G : NetworkX graph

    search_point : Shapely Point
        point from which to search

    search_distance : float
        maximum distance to search from the search_point

    geometry : str
        (default = 'geometry')
        attribute of edges containing Shapely LineString representations of
        edge paths

    edges_sindex : rtree Index 

    Returns
    ----------
    u, v, key, point : tuple 

    """
    # extract edge indices and geometries from the graph
    edge_IDs = [i for i in G.edges]
    edge_geometries = [data[geometry] for _, _, data in G.edges(data=True)]
    if edges_sindex is None:
        line_index = [data[geometry] for _, _, data in G.edges(data=True)]

    # find the closest point for connection along the network 
    edge_ID, point = closestPointAmongLines(search_point, edge_geometries, 
        lines_sindex=edges_sindex, search_distance=search_distance)

    if edge_ID is not None:
    
        try: # will not return key if the network is DiGraph
            # get the node_IDs for the edge with the closest point
            u, v  = edge_IDs[edge_ID]
            return u, v, point
        except:
            pass
        
        try: # will return key if network is MultiDiGraph
            # get the node_IDs for the edge with the closest point
            u, v, key  = edge_IDs[edge_ID]
            return u, v, key, point
        except:
            return None, None, None, None
    else:
        return None, None, None, None


def insert_node(G, u, v, node_point, node_name, key = None):
    """
    Insert a node along a NetworkX graph edge and split the edge's Shapely
    LineString geometry where the node is inserted

    Parameters
    ----------
    G : NetworkX graph

    u : int
        first node ID for edge along which node is being inserted

    v : int
        second node ID for edge along which node is being inserted

    node_point : Shapely Point
        geometric location for new node

    node_name : str
        name for new node

    key : int
        (default = None)
        key for edge along which node is being inserted

    Returns
    ----------
    G : NetworkX Graph 

    """
    # get attributes from the existing nodes
    u_attrs = G.node[u]
    v_attrs = G.node[v]
    # assemble attributes for the new node
    new_node_attrs = {'geometry': node_point, 
                      'x': node_point.x,
                      'y': node_point.y}
    if key is None:
        if G.has_edge(u, v): # examine the edge from u to v
            # get attributes from existing edge
            attrs = G.get_edge_data(u, v)
            original_geom = attrs['geometry']
            # delete existing edge
            G.remove_edge(u, v)
            # specify nodes for the new edges
            G.add_node(u, **u_attrs)
            G.add_node(v, **v_attrs)
            G.add_node(node_name, **new_node_attrs)
            # construct attributes for first new edge
            attrs['geometry'] = segment(original_geom, 
                                        endPoints(original_geom)[0], 
                                        node_point)
            attrs['length'] = attrs['geometry'].length
            G.add_edge(u, node_name, **attrs)
            # construct attributes for second new edge
            attrs['geometry'] = segment(original_geom, 
                                        node_point, 
                                        endPoints(original_geom)[1])
            attrs['length'] = attrs['geometry'].length
            G.add_edge(node_name, v, **attrs)
        if G.has_edge(v, u): # examine the edge from v to u
            # get attributes from existing edge
            attrs = G.get_edge_data(v, u)
            original_geom = attrs['geometry']
            # delete existing edge
            G.remove_edge(v, u)
            # specify nodes for the new edges
            G.add_node(u, **u_attrs)
            G.add_node(v, **v_attrs)
            G.add_node(node_name, **new_node_attrs)
            # construct attributes for first new edge
            attrs['geometry'] = segment(original_geom, 
                                        endPoints(original_geom)[0], 
                                        node_point)
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge
            G.add_edge(v, node_name, **attrs)
            # construct attributes for second new edge
            attrs['geometry'] = segment(original_geom, 
                                        node_point, 
                                        endPoints(original_geom)[1])
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge   
            G.add_edge(node_name, u, **attrs)
        return G
    else:
        if G.has_edge(u, v, key):
            # get attributes from existing edge
            attrs = G.get_edge_data(u, v, key)
            original_geom = attrs['geometry']
            # delete existing edge
            G.remove_edge(u, v, key)
            # specify nodes for the new edges
            G.add_node(u, **u_attrs)
            G.add_node(v, **v_attrs)
            G.add_node(node_name, **new_node_attrs)        
            # construct attributes for first new edge            
            attrs['geometry'] = segment(original_geom, 
                                        endPoints(original_geom)[0], 
                                        node_point)
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge
            G.add_edge(u = u, v = node_name, key = 0, **attrs)

            # construct attributes for second new edge
            attrs['geometry'] = segment(original_geom,
                                        node_point, 
                                        endPoints(original_geom)[1])
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge   
            G.add_edge(u = node_name, v = v, key = 0, **attrs)    
        if G.has_edge(v, u, key):
            # get attributes from existing edge
            attrs = G.get_edge_data(v, u, key)
            original_geom = attrs['geometry']
            # delete existing edge
            G.remove_edge(v, u, key)
            # specify nodes for the new edges
            G.add_node(u, **u_attrs)
            G.add_node(v, **v_attrs)
            G.add_node(node_name, **new_node_attrs)
            # construct attributes for first new edge
            attrs['geometry'] = segment(original_geom, 
                                        endPoints(original_geom)[0], 
                                        node_point)
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge
            G.add_edge(u = v, v = node_name, key = 0, **attrs)
            # construct attributes for second new edge
            attrs['geometry'] = segment(original_geom, 
                                        node_point, 
                                        endPoints(original_geom)[1])
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge   
            G.add_edge(u = node_name, v = u, key = 0, **attrs)
        return G


def split_gdf_Lines(gdf, segment_length, centered = False, min_length = 0):
    """Split linestrings in a geodataframe into equal-length peices.

    Attributes in accompanying columns are copied to all children of each
    parent record.

    Args:
        gdf (GeoPandas GeoDataFrame): geodataframe with LineString geometry
        segment_length (float): length of segments to create
        centered : bool or str
            False (default) = not centered; points are spaced evenly from the
                start of the LineString 
            'Point' = a point is located at the LineString midpoint
            'Space' = a gap between points is centered on the LinesString

    Returns:
        GeoPandas GeoDataFrame
    """
    # initiate new dataframe to hold segments
    segments = gpd.GeoDataFrame(data=None, columns=gdf.columns, 
                                geometry = 'geometry', crs=gdf.crs)
    for i, segment in gdf.iterrows():
        points = points_along_lines(segment['geometry'], 
                                    segment_length, 
                                    centered = centered)
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
        # replace the geometry for these copied records with the segment geometry
        segment_records['geometry'] = segment_geometries
        # add new segments to full list
        segments = segments.append(segment_records, ignore_index=True)
    return segments


#### Function to make a rectangular bounding box around all elements in a geodataframe
def boundsBox(gdf):
    bounds = gdf.total_bounds
    return Polygon([(bounds[0], bounds[1]),
                    (bounds[2], bounds[1]),
                    (bounds[2], bounds[3]),
                    (bounds[0], bounds[3])])


#### Function to replace gdf geometry with centroids
def centroidGDF(gdf):
    gdf = gdf.copy()
    centroids = gdf.centroid
    gdf['geometry'] = centroids
    return gdf 


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

