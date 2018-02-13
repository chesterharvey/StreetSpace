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


def shortenLine(linestring, shorten_dist, ends = 'both'):
    """
    Shorten a Shapely Linestring at either end.

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
        start = linestring.interpolate(shorten_dist)
        end = linestring.interpolate(linestring.length - shorten_dist)
    elif ends == 'start':
        start = linestring.interpolate(shorten_dist)
        end = endPoints(linestring)[1]
    elif ends == 'end':
        start = endPoints(linestring)[0]
        end = linestring.interpolate(linestring.length - shorten_dist)
    return LineString([start,end])


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

# def nearest_point(search_feature, find_feature, search_dist=None,
#     use_sidx=False, find_features_sidx=None, return_search_index=False,
#     return_find_index=False):
#     """
#     Find the closest point from the search feature among the find features.

#     Parameters
#     ----------
#     search_feature: Shapely geometry or GeoPandas GeoDataFrame
#         feature(s) to search from

#     find_feature: Shapely geometry or GeoPandas GeoDataFrame
#         feature(s) to search toward

#     search_dist: float
#         maximum distance to search from the search_feature
#         (default = None)

#     use_sidx: bool
#         whether a spatial index is used to economize potential matches
#         (default = False)

#     find_features_sidx: Rtree Index
#         spatial index for find_feature
#         (default = None)
#         Note: including a pre-built spatial index object can improve
#         efficiency if the same find features are used iteratively

#     return_search_index: bool
#         return index of search_feature that is closest to find_feature
#         (default = False)

#     return_find_index: bool
#         return index of find_feature that is closest to search_feature
#         (default = False)
#     """

def closestPoint(search_point, find_points, attribute_list = None, points_index = None, search_dist = None):
    # only lines within a search distance based on spatial index, if specified
    if points_index != None:
        if search_dist == None:
            raise ValueError('must specify search_dist if using spatial index')
        # construct search area around point
        search_area = search_point.buffer(search_dist)
        # get nearby IDs
        nearby_IDs = [int(i) for i in points_index.intersection(search_area.bounds)]
        # get nearby geometries
        find_points = [find_points[i] for i in nearby_IDs]
    # get only those lines within search distance, if specified
    elif search_dist != None:
        # construct search area around point
        search_area = search_point.buffer(search_dist)
        # get lines intersecting search area
        find_points = [point for point in find_points if point.intersect(search_area)]
    # find closest point along find geometry
    _, find_point = sh.ops.nearest_points(search_point, MultiPoint(find_points))
    # get attribute 
    if attribute_list != None:
        # throw exception if attribute list isn't appropriate length
        if len(attribute_list) != len(find_points):
            raise ValueError('attribute list must have same length as find_points list')
        # find index for point identified above
        point_index = [i for i, point in enumerate(find_points) if point.equals(find_point)][0]
        # get attribute based on that index
        attribute = int(attribute_list[point_index])
        # return both geometry and attribute
        return(find_point, attribute)
    # if no attribute list provided, just return geometry
    return find_point


#### Create rtree spatial index
def spatialIndex(geometries):
    idx = index.Index()
    for i, geom in enumerate(geometries):
        idx.insert(i, geom.bounds)
    return idx


#### Function to create equally spaced points along a line (centered can equal 'Point','Space', or False)
def pointsAlongLines(lines, spacing, centered = False):
    # check whether input is a single linestring
    if isinstance(lines, LineString):
        lines = [lines]
    # Iterate through line geometries
    all_points = []
    for l, line in enumerate(lines):
        points = []
        for p in range(int(ceil(line.length/spacing))):
            if centered == False:
                starting_point = 0
            elif centered in ['point', True]:
                starting_point = (line.length / 2) - (((line.length / 2) // spacing) * spacing)
            elif centered == 'space':
                # Space the starting point from the end so the points are centered on the edge
                starting_point = (line.length - (line.length // spacing) * spacing) / 2
            x, y = line.interpolate(starting_point + (p * spacing)).xy
            point = sh.geometry.Point(x[0], y[0])
            # Store point in list
            points.append(point)
        all_points.extend(points)
    return all_points


#### Function to split a line at points along it
# Adapted from: https://stackoverflow.com/questions/34754777/shapely-split-linestrings-at-intersections-with-other-linestrings
def splitLineByPoint(line, points_list):
    # get original coordinates of line
    coords = list(line.coords)
    # break off last coordinate (in case line is a loop first/last are the same)
    last_coord = coords[-1]
    coords = coords[0:-1]
    # keep list of coords for cut points (new segment endpoints)
    cuts = [0] * len(coords)
    cuts[0] = 1     
    # add the coords from the cut points
    coords += [list(p.coords)[0] for p in points_list]    
    cuts += [1] * len(points_list)
    # calculate the distance along the line for each coordinate
    dists = [line.project(Point(p)) for p in coords]
    # sort the coords/cuts based on the distances
    coords = [p for (d, p) in sorted(zip(dists, coords))]
    cuts = [p for (d, p) in sorted(zip(dists, cuts))]
    # add back last coordinate
    coords = coords + [last_coord]
    cuts = cuts + [1]
    # generate the Lines      
    lines = []
    for i in range(len(coords)-1):           
        if cuts[i] == 1:    
            # find next element in cuts == 1 starting from index i + 1   
            j = cuts.index(1, i + 1)    
            lines.append(LineString(coords[i:j+1]))
    return lines

#### Get segment of a linestring
def getLineSegment(line, pointA, pointB):
    segment = splitLineByPoint(line, [pointA, pointB])[1]
    if endPoints(segment)[0].equals(pointA):
        return segment
    else:
        return LineString(np.flip(np.array(segment), 0))

#### Function to get azimuth between line endpoints
def lineAzimuth(line):
    a = endPoints(line)[0]
    b = endPoints(line)[1]
    return np.degrees(np.arctan2(a.y - b.y, a.x - b.x))


#### Function to split a line at vertices
def splitLineAtVertices(line):
    # Get coordinates of linestring
    coords = list(line.coords)
    # generate individual lines   
    lines = [sh.geometry.LineString([coords[i],coords[i + 1]]) for i in range(len(coords) - 1)]
    return lines


#### Function to get start and endpoint of line
def endPoints(line):
    return Point(line.xy[0][0], line.xy[1][0]), Point(line.xy[0][-1], line.xy[1][-1])


#### Function to get angle of a linestring at a certain distance along it
def azimuthAtDistance(line, distance):
    line_segments = splitLineAtVertices(line)
    segment_lengths = [edge.length for edge in line_segments]
    cumulative_lengths = []
    for i, length in enumerate(segment_lengths):
        if i == 0:
            cumulative_lengths.append(length)
        else:
            cumulative_lengths.append(length + cumulative_lengths[i-1])
    # get index of split edge that includes the specified distance by searching the list in reverse order
    for i, length in reversed(list(enumerate(cumulative_lengths))):
        if length >= distance:
            line_segment_ID = i
    return lineAzimuth(line_segments[line_segment_ID])



#### Function to draw a line based on starting point, distance, and angle
def drawLineAtAngle(start_point, distance, angle):
    end_x = start_point.x + np.cos(np.radians(angle)) * distance
    end_y = start_point.y + np.sin(np.radians(angle)) * distance
    return LineString([Point([start_point.x, start_point.y]), Point([end_x, end_y])])


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


#### Function to identify the position of the closest point along a network
def closestPointAlongNetwork(G, point, search_distance, edges_sindex = None):
    # extract edge indices and geometries from the graph
    edge_IDs = [i for i in G.edges]
    edge_geometries = [data['geometry'] for _, _, data in G.edges(data=True)]
    if edges_sindex is None:
        line_index = [data['geometry'] for _, _, data in G.edges(data=True)]

    # find the closest point for connection along the network 
    edge_ID, point = closestPointAmongLines(point, edge_geometries, lines_index = edges_sindex, search_dist = search_distance)

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

#### Function to insert a node along a network edge
def insertNode(G, u, v, node_point, node_name, key = None):
   
    # get attributes from the existing nodes
    u_attrs = G.node[u]
    v_attrs = G.node[v]   

    # assemble attributes for the new node
    new_node_attrs = {'geometry': node_point, 'x': node_point.x, 'y': node_point.y}

    # add new edges connecting the node into the network
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
            attrs['geometry'] = getLineSegment(original_geom, endPoints(original_geom)[0], node_point)
            # attrs['geometry'] = LineString([G.node[u]['geometry'], node_point]) ########
            attrs['length'] = attrs['geometry'].length
            # specify new edge
            G.add_edge(u, node_name, **attrs)

            # construct attributes for second new edge
            attrs['geometry'] = getLineSegment(original_geom, node_point, endPoints(original_geom)[1])
            # attrs['geometry'] = LineString([node_point, G.node[v]['geometry']]) ########
            attrs['length'] = attrs['geometry'].length
            # specify new edge   
            G.add_edge(node_name, v, **attrs)
            
        if G.has_edge(v, u): # examine the edge from v to u
            # get attributes from existing edge
            attrs = G.get_edge_data(v, u)

            # delete existing edge
            G.remove_edge(v, u)

            # specify nodes for the new edges
            G.add_node(u, **u_attrs)
            G.add_node(v, **v_attrs)
            G.add_node(node_name, **new_node_attrs)

            # construct attributes for first new edge
            attrs['geometry'] = getLineSegment(original_geom, endPoints(original_geom)[0], node_point)
            # attrs['geometry'] = LineString([G.node[v]['geometry'], node_point])
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge
            G.add_edge(v, node_name, **attrs)

            # construct attributes for second new edge
            attrs['geometry'] = getLineSegment(original_geom, node_point, endPoints(original_geom)[1])
            # attrs['geometry'] = LineString([node_point, G.node[u]['geometry']])
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge   
            G.add_edge(node_name, u, **attrs)
        return G
    
    else:
        # insert new node and connecting edges into the network
        if G.has_edge(u, v, key): # examine the edge from u to v
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
            attrs['geometry'] = getLineSegment(original_geom, endPoints(original_geom)[0], node_point)
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge
            G.add_edge(u = u, v = node_name, key = 0, **attrs)

            # construct attributes for second new edge
            attrs['geometry'] = getLineSegment(original_geom, node_point, endPoints(original_geom)[1])
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge   
            G.add_edge(u = node_name, v = v, key = 0, **attrs)    

        if G.has_edge(v, u, key): # examine the edge from v to u
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
            attrs['geometry'] = getLineSegment(original_geom, endPoints(original_geom)[0], node_point)
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge
            G.add_edge(u = v, v = node_name, key = 0, **attrs)

            # construct attributes for second new edge
            attrs['geometry'] = getLineSegment(original_geom, node_point, endPoints(original_geom)[1])
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge   
            G.add_edge(u = node_name, v = u, key = 0, **attrs)
        return G


#### Function to split line records in a geodataframe into pieces, maintaining original attributes
def splitGDFLines(gdf, spacing, centered = False, min_length = 0):

    # initiate new dataframe to hold segments
    segments = gpd.GeoDataFrame(data=None, columns=gdf.columns, geometry = 'geometry', crs=gdf.crs)
    
    for i, segment in gdf.iterrows():
        
        # make equally spaced points along the segment, excluding the start point
        points = pointsAlongLines(segment['geometry'], spacing, centered = centered)[1:]
        
        # cut the segment at each point
        segment_geometries = splitLineByPoint(segment['geometry'], points)

        if len(segment_geometries) > 1:
        
            # merge the end segments less than minimum length
            if segment_geometries[0].length < min_length:
                print(len(segment_geometries))
                segment_geometries[1] = sh.ops.linemerge(MultiLineString([segment_geometries[0], segment_geometries[1]]))
                segment_geometries = segment_geometries[1:]

            if segment_geometries[-1].length < min_length:
                segment_geometries[-2] = sh.ops.linemerge(MultiLineString([segment_geometries[-2], segment_geometries[-1]]))
                segment_geometries = segment_geometries[:-1]

        # copy the segment records
        segment_records = gpd.GeoDataFrame(data=[segment]*len(segment_geometries), columns=gdf.columns, geometry = 'geometry', crs=gdf.crs)

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