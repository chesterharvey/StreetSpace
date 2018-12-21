################################################################################
# Module: network.py
# Description: Functions to manipulate and analyze NetworkX graphs.
# License: MIT
################################################################################

import networkx as nx
import numpy as np
import os
import shutil
import pathlib
from rtree import index
from networkx import Graph, DiGraph, MultiGraph, MultiDiGraph
from time import time
from pandas import DataFrame
from geopandas import GeoDataFrame
from collections import OrderedDict

from .geometry import *
from .utils import *


def closest_point_along_network(search_point, G, search_distance=None, 
    sindex=None, accessibility_functions=None, verbose = False):
    """
    Find the closest point along the edges of a NetworkX graph with Shapely 
    LineString geometry attributes in the same coordinate system.

    Example restriction function:

    def low_stress_highway(data):
        low_stress = ['cycleway','residential','unclassified','tertiary',
                      'secondary','primary']
        # print(data.keys())
        if (data is not None and 'highway' in data):
            return any(h in low_stress for h in listify(data['highway']))
        else:
            return False

    Parameters
    ----------
    search_point : :class:`shapely.geometry.Point`
        Point from which to search
    G : :class:`networkx.Graph`, :class:`networkx.DiGraph`, \
    :class:`networkx.MultiGraph` or :class:`networkx.MultiDiGraph`
        Graph along which closest point will be found. Each edge of `G` must\
        have a :class:`shapely.geometry.LineString` geometry attribute.
    search_distance : :obj:`float`, optional, default = ``None``
        Maximum distance to search from the `search_point`
    sindex : :class:`rtree.index.Index`, optional, default = ``None``
        Spatial index for edges of `G`
    restrictions: :obj:`list`
        Function used to restrict edges on which closest point may\
        be located. Function must have a single argument that accepts the\
        attribute dictionary for each edge (produced by  ``G.get_edge_data``)\
        and must return ``True`` or ``False``. See above for example function.

    Returns
    -------
    closest_point : :class:`shapely.geometry.Point`
        Location of closest point
    edge : :obj:`tuple`
        Index for the closest edge (u, v, key) or (u, v)
    distance : :obj:`float`
        Distance from the search_point to the closest edge    
    """
    if sindex:
        if not search_distance:
            raise ValueError('Must specify search_distance if using spatial index')
        # Construct search bounds around the search point
        search_bounds = search_point.buffer(search_distance).bounds
        # Get indices for edges that intersect the search bounds
        edges = [edge for edge in sindex.intersection(
            search_bounds, objects='raw')]
        # Restrict to edges that are in the current graph
        edges = [edge for edge in edges if (G.has_edge(*edge))]
        # Collect geometries for each edge
        geometries = [G.get_edge_data(*edge)['geometry'] for edge in edges]
        # Remove geometries and associated edges that are not line shapes
        for i, geometry in enumerate(geometries):
            if not isinstance(geometry, (LineString, MultiLineString)):
                del geometries[i]
                del edges[i]
        edge_tuples = list(zip(edges, geometries))
    elif search_distance:
        # Construct search bounds around the search point
        search_area = search_point.buffer(search_distance)
        edges = G.edges(keys=True, data='geometry')
        # Collect edges that intersect the search area as (index, geometry) tuples
        edge_tuples = [seperate_edge_index_and_geom(edge) for edge
                       in edges if edge[-1].intersects(search_area)]
    else:
        # Collect all edges as (index, geometry) tuples
        edges = G.edges(keys=True, data='geometry')
        edges_tuples = [seperate_edge_index_and_geom(edge) for edge in edges]
    # Pare down edges based on further criteria
    if accessibility_functions:
        edge_tuples = [edge_tuple for edge_tuple in edge_tuples
                       if _flag_accessible(
                           G.get_edge_data(*edge_tuple[0]), 
                           accessibility_functions)]                         
    # Feed edges to general function for finding closest point among lines
    closest_point, edge, distance = closest_point_along_lines(
        search_point, edge_tuples)
    return closest_point, edge, distance


def reverse_edge(edge):
    """Switches u and v in an edge tuple.

    """
    reverse = list(edge)
    reverse[0] = edge[1]
    reverse[1] = edge[0]
    return tuple(reverse)


def add_new_edge(G, edge, geometry, attrs=None, sindex=None, next_sindex_id=None):
    """Add a new edge within a graph, and update 

    """   
    # Assemble attributes for first new edge            
    if attrs is None:
        attrs = {}
    attrs['geometry'] = geometry
    attrs['length'] = attrs['geometry'].length
    G.add_edge(*edge, **attrs)
    if sindex:
        if next_sindex_id:
            sindex.insert(next_sindex_id, G.get_edge_data(*edge)['geometry'].bounds, edge)
            return next_sindex_id + 1
        else:
            sindex.insert(0, G.get_edge_data(*edge)['geometry'].bounds, edge)       


def remove_edge(G, edge, location, sindex=None):
    if sindex:
        edge_geom = G.get_edge_data(*edge)['geometry']
        location = edge_geom.buffer(1).bounds
        # location = location.buffer(500).bounds
        nearby_edges = search_sindex_items(sindex, location, bbox=True)
        sindex_id = [x[0] for x in nearby_edges if x[1] == edge]
        sindex_id = sindex_id[0]
        bbox = [x[2] for x in nearby_edges if x[1] == edge]
        bbox = bbox[0]
        sindex.delete(sindex_id, bbox)
    G.remove_edge(*edge)


def lookup_sindex_id(object, items=None, sindex=None, search_bounds=None):
    if items is None:
        items = search_sindex_items(sindex, search_bounds=search_bounds)
    return [sindex_id for (sindex_id, edge_id) in 
            items if edge_id == object][0]


def insert_node_along_edge(G, edge, node_point, node_name, node_points=None,
    both_ways=False, sindex=None, verbose=False):
    """Insert a node along an edge with a geometry attribute.

    ``edge`` must have a LineString geometry attribute
    which will be split at the location of the new node.

    If `G` is :class:`networkx.DiGraph` or :class:`networkx.MultiDiGraph`,
    the new node will split edges in both directions.

    Parameters
    ----------
    G : :class:`networkx.Graph`, :class:`networkx.DiGraph`,\
    :class:`networkx.MultiGraph` or :class:`networkx.MultiDiGraph`
        Graph into which new node will be inserted. Each edge of `G` must\
        have a :class:`shapely.geometry.LineString` geometry attribute.
    edge : :obj:`tuple`
        * if G is Graph or DiGraph: (u, v)
        * if G is MultiGraph or MultiDiGraph: (u, v, key)
    node_point : :class:`shapely.geometry.Point`
        Geometric location of new node
    node_name : :obj:`str`
        Name for new node
    both_ways : :obj:`bool`, optional, default = ``False``
        Specifies whether a node will also be inserted into an edge in the\
        opposite direction 
    sindex : rtree index
        Spatial index for G (best created with graph_sindex). If specified,\
        this index will be updated appropriatly when edges are added or\
        removed from ``G``.
    """   
    # Add new node
    new_node_attrs = {'geometry': node_point, 
                      'x': node_point.x,
                      'y': node_point.y,
                      'osmid': 000000000}
    G.add_node(node_name, **new_node_attrs)
    # Get attributes from existing edge
    attrs = G.get_edge_data(*edge)
    original_geom = attrs['geometry']
    # Delete existing edge
    if verbose:
        print('removing edge {} from graph'.format(edge))
    remove_edge(G, edge, node_point)       
    if verbose:
        print('adding first forward edge {} to graph and index'.format((edge[0], node_name, 0)))
    # Add new edges
    add_new_edge(G, (edge[0], node_name, 0), 
                segment(original_geom, endpoints(original_geom)[0], node_point),
                attrs, sindex=sindex)
    if verbose:
        print('adding second forward edge {} to graph and index'.format((node_name, edge[1], 0)))
    add_new_edge(G, (node_name, edge[1], 0), 
                segment(original_geom, node_point, endpoints(original_geom)[1]),
                attrs, sindex=sindex)
    if both_ways:
        # Flip the start and end node
        reverse = tuple(reverse_edge(edge))
        # Check whether reverse edge in graph
        if G.has_edge(*reverse):           
            # See if the two edges are similar:
            reverse_geometry = G.get_edge_data(*reverse)['geometry']
            reverse_midpoint = midpoint(reverse_geometry)
            reverse_length = reverse_geometry.length
            edge_midpoint = midpoint(original_geom)
            edge_length = original_geom.length
            if (edge_midpoint.almost_equals(edge_midpoint, 0) and 
                (0.95 < (edge_length/reverse_length) < 1.05)):

            # edge_buffer = original_geom.buffer(1)
            # if reverse_geometry.within(edge_buffer):

                # print('reverse passed equivalency test')

                if verbose:
                    print('forward edge and reverse edge passed similarity test')
                # Get attributes for the reverse edge
                attrs = G.get_edge_data(*reverse)
                if verbose:    
                    print('removing reverse edge {} from graph'.format(reverse))   
                remove_edge(G, reverse, node_point)
                # Add new edges
                if verbose:
                    print('adding first reverse edge {} to graph and index'.format((reverse[0], node_name, 0)))
                add_new_edge(G, (reverse[0], node_name, 0), 
                    segment(original_geom, endpoints(reverse_geometry)[0], node_point),
                    attrs, sindex=sindex)
                if verbose:
                    print('adding second reverse edge {} to graph and index'.format((node_name, reverse[1], 0)))
                add_new_edge(G, (node_name, reverse[1], 0), 
                    segment(original_geom, node_point, endpoints(reverse_geometry)[1]),
                    attrs, sindex=sindex)
            else:
                if verbose:
                    print('Edge {} and its reverse ({}) are not alinged.'.format(edge, reverse))


def search_sindex_items(sindex, search_bounds=None, bbox=False):
    if search_bounds is None:
        search_bounds = sindex.bounds
    edges = [x for x in sindex.intersection(search_bounds, objects=True)]
    indices = [x.id for x in edges]
    objects = [x.object for x in edges]
    if bbox:
        bboxes = [x.bbox for x in edges]
        return list(zip(indices, objects, bboxes))
    else:
        return list(zip(indices, objects))


def _flag_accessible(data, accessibility_functions):
    # Default flag position is true (edge is accessible)
    flag = True
    # Iterate through accessibility functions
    for key, function in accessibility_functions.items():
        if key in data:
            # If function response false
            if function(data[key]) is False:
                # Switch flag to false
                flag = False
                break
        else:
            flag = False
    return flag


def connect_points_to_closest_edges(G, points, search_distance=None, 
    sindex=None, points_to_nodes=True, accessibility_functions=None, verbose=False):
    """Connect points to a graph by inserting a node along their closest edge.

    G : :class:`networkx.Graph`, :class:`networkx.DiGraph`,\
    :class:`networkx.MultiGraph` or :class:`networkx.MultiDiGraph`
        Graph into which new node will be inserted. Each edge of `G` must\
        have a :class:`shapely.geometry.LineString` geometry attribute.
    points : :obj:`list`
        List of tuples with structure (point_name, point_geometry). 'u' and\
        'v' suffixes will be appended to point names when they are applied\
        to nodes in order to distinguish between original points (u) and\
        those connecting to the graph (v).
    search_distance : :obj:`float`, optional, default = ``None``
        Maximum distance to search for an edge from each point
    sindex : :class:`rtree.index.Index`, optional, default = ``None``
        Spatial index for `G`
    return_unplaced : :obj:`bool`, optional, default = ``False``
        If ``True``, will return points that are outside the search distance\
        or have otherwise not been connected to the graph
    points_to_nodes : :obj:`bool`, optional, default = ``True``
        If ``True``, will add ``points`` as new nodes and edges connecting\
        them to the inserted nodes. If `G` is directed, connecting edges will\
        be added in both directions.

    Returns
    ----------
    :obj:`list`
        Points not connected to the graph (if ``return_unplaced`` is ``True``)
    """
    # Make list to record unplaced points
    unplaced_points =[]
    # Make a dictionary for mapping alias node names
    node_aliases = {}
    
    if verbose:
        node_check_time = 0
        draw_connector_time = 0
        insert_node_time = 0
    # u_point refers to the off-the-graph point which is being connected
    # v_point refers to the on-graph point where the connection is made   
    for name, u_point in points:
        if points_to_nodes:
            u_name = name
            v_name = '{}_link'.format(name)
        else:
            v_name = name
        # Find the closest point suitable for connection along the network
        v_point, edge, _ = closest_point_along_network(
            u_point, G, search_distance=search_distance, 
            sindex=sindex, accessibility_functions=accessibility_functions, verbose=verbose)
        if v_point:
            if verbose:
                node_check_start = time()
            # See if v_point is the same as an existing graph node
            node_already_exists = False
            # Get nearby nodes from edges sindex
            nearby_edges = [x.object for x in 
                sindex.intersection(v_point.buffer(10).bounds, objects=True)]
            nearby_nodes = list(set([x for edge in nearby_edges
                for x in edge[1:2]]))
            nearby_points = [Point(G.node[node]['x'], G.node[node]['y'])
                for node in nearby_nodes]
            distances = [point.distance(v_point) for point in nearby_points]
            min_index = distances.index(min(distances))
            min_distance = min(distances)
            min_id = nearby_nodes[min_index]
            if min_distance < 1:               
                node_already_exists = True
                node_aliases[v_name] = min_id
                v_name = min_id
            if verbose:
                node_check_time += (time()-node_check_start)
                insert_node_start = time()
            if not node_already_exists:
                # If proposed node is farther, insert it into the graph 
                if not G.has_edge(*edge):
                    unplaced_points.append((name, u_point))
                else:
                    insert_node_along_edge(
                        G, edge, v_point, v_name, both_ways=True, sindex=sindex)
            if verbose:
                insert_node_time += (time()-insert_node_start)
                draw_connector_start = time()
            # Connect u_point to the graph
            if points_to_nodes:
                # Add a node at the location of u_point and edges to link it
                attrs = {'geometry': u_point,
                         'x': u_point.x,
                         'y': u_point.y}
                G.add_node(u_name, **attrs)
                # Add an edge connecting it to the previously inserted point
                add_new_edge(G, (u_name, v_name, 0), 
                            LineString([u_point, v_point]),
                            sindex=sindex)
                if nx.is_directed(G):
                    add_new_edge(G, (v_name, u_name, 0), 
                        LineString([v_point, u_point]),
                        sindex=sindex)
            if verbose:
                draw_connector_time += (time()-draw_connector_start)
        else:
            unplaced_points.append((name, u_point))
    if verbose:
        print('node check time: {}'.format(node_check_time))
        print('insert node time: {}'.format(insert_node_time))
        print('draw connector time: {}'.format(draw_connector_time))
    nodes_sindex = None
    return unplaced_points, node_aliases


def seperate_edge_index_and_geom(edge):
    """Seperate an edge's index tuple (e.g., (u, v, key)) from its geometry.

    Designed to handle the output of G.edges(data='geometry').

    Parameters
    ----------
    edge : :obj:`tuple`
        Geometry object be the last element. For example:\
        (u, v, key, LineString)

    Returns
    ----------
    :obj:`tuple`
        Edge idex elements seperated into their own tuple,\
        e.g.: ((u, v, key), LineString) 
    """
    index = edge[:len(edge)-1]
    geometry = edge[-1]
    return (index, geometry)


def _make_index(G, path, open_copy):
    def generator(edges):
        for i, edge in enumerate(edges):
            edge_tuple, geometry = seperate_edge_index_and_geom(edge)
            yield (i+1, geometry.bounds, edge_tuple)
    # Remove any old versions that exist
    # (since `index` method appends rather than overwriting)
    try:
        os.remove(path + '.idx')
    except OSError:
        pass
    try:
        os.remove(path + '.dat')
    except OSError:
        pass
    # Construct index into the path
    edges = G.edges(keys=True, data='geometry')
    idx = index.Index(path, generator(edges))#, properties = p)
    # Close the file to complete writing process
    idx.close()
    # Reopen the serialized index, either original or temp copy
    return load_index(path, open_copy=open_copy) 


def make_graph_sindex(G, path=None, load_existing=False, open_copy=False):
    """Create a spatial index from a graph with geometry attributes.

    Index IDs begin at 1 so ID 0 is reserved for inserting temporary items\
    (e.g., new edges for a particular analysis) into the index.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        Must include :class:`shapely.geometry.LineString` geometry attributes

    path : :obj:`str`
        Path at which to store serialized index or from which to load\
        existing serialization.
    load_existing : :obj:`bool`
        * ``True`` : Will load existing serialized index if available
        * ``False`` : Will overwrite existing serialized index if it exists
    open_copy : :obj:`bool`
        Open a copy of the serialized index (storing at `temp\index`),\
        leaving the original version archived.

    Returns
    ----------
    :class:`rtree.index.Index`
        Spatial index
    """
    
    def make_index(G, path=None, open_copy=False):
        def generator(edges):
            for i, edge in enumerate(edges):
                edge_tuple, geometry = seperate_edge_index_and_geom(edge)
                yield (i+1, geometry.bounds, edge_tuple)
        # Remove any old versions that exist
        edges = G.edges(keys=True, data='geometry')
        if path:
            # (since `index` method appends rather than overwriting)
            try:
                os.remove(path + '.idx')
            except OSError:
                pass
            try:
                os.remove(path + '.dat')
            except OSError:
                pass
            # Construct index into the path
            idx = index.Index(path, generator(edges))#, properties = p)
            # Close the file to complete writing process
            idx.close()
            # Reopen the serialized index, either original or temp copy
            return load_index(path, open_copy=open_copy)
        else:
            return index.Index(generator(edges))

    if path:
        if open_copy:
            if (os.path.isfile(path + '.idx') & os.path.isfile(path + '.dat')):
                return load_index(path, open_copy=open_copy)
            else:
                return make_index(G, path, open_copy)
        else:
            return make_index(G, path)
    else:
        return make_index(G)
        # return _make_index(G, path, open_copy)  

def load_index(path, open_copy=False, close_first=None):
    """Load a serialized rtree index.
    
    ``open_copy`` is useful because rtree indices load by default in\
    'append mode', where any changes to the index object will modify the\
    serialized version saved on disk. ``open_copy`` copies the index to a\
    temporary location (``temp\index``) and opens this version so that\
    the original remains unmodified.
    
    Parameters
    ----------
    path : :obj:`str`
        Path to the serialized sindex that will be copied to temp.* in the\
        current directory

    open_copy : :obj:`bool`
        Make a copy of the index files at ``temp\`` and open that copy,\
        leaving the original version archived.

    Returns
    ----------
    :class:`rtree.index.Index`
        Spatial index
    """
    # Make sure file is closed
    if close_first:
        path.close()
    if open_copy:
        # Ensure that temp directory exists
        pathlib.Path('temp').mkdir(parents=True, exist_ok=True)
        # Copy the indes files to a temp folder
        try:
            os.remove('index.idx')
        except OSError:
            pass
        try:
            os.remove('index.dat')
        except OSError:
            pass
        shutil.copy(path + '.idx', 'temp/index.idx')
        shutil.copy(path + '.dat', 'temp/index.dat')
        return index.Index('temp/index')
    else:
        return index.Index(path)


def route_node_pairs(node_pairs, G, weight='length', both_ways=False, verbose=False):
    """Route shortest paths between pairs of nodes in a graph.

    Parameters
    ----------
    nodes_pairs : :obj:`list`
        List of tuples of the form (origin_node, destination_node).
    G : :class:`networkx.Graph`
        Graph to route on
    weight : :obj:`str`
        Edge attribute to use as a weight for optomizing paths
    both_ways : :obj:`bool`, optional, default = ``False``
        If ``True``, finds routes between the nodes in both directions

    Returns
    ----------
    :obj:`list`
        List of lists of node IDs for each route.
    """
    def route(G, O, D, weight='length'):
        try:
            return nx.shortest_path(G, O, D, weight)
        except Exception as e:
            if isinstance(e, KeyError):
                return 'No node {} in the graph'.format(e)
            elif isinstance(e, nx.exception.NetworkXNoPath):
                return 'No path between nodes {} and {}'.format(O,D)
            else:
                return 'Unknown error routing between nodes {} and {}'.format(O,D)
    if both_ways:
        return ([route(G, O, D, weight) for O, D in node_pairs] + 
                [route(G, D, O, weight) for O, D in node_pairs])
    else:
        return [route(G, O, D, weight) for O, D in node_pairs]


def route_between_points(points, G, additional_summaries=None, summarize_links=False,
    search_distance=None, sindex=None, points_to_nodes=True, weight='length', 
    both_ways=False, accessibility_functions=None, verbose=False):
    """Route between pairs of points passed as columns in a DataFrame
    

    """
    # Make a copy of the graph so it doesn't change
    G = G.copy()

    points = points.reset_index(drop=True)
    points_order = ['a_name', 'a_point', 'b_name', 'b_point']
    # Parse DataFrame columns into lists (assumes correct column order)
    if len(points.columns) == 2:
        a_points = points[points.columns[0]].tolist() 
        b_points = points[points.columns[1]].tolist()
        points.columns = ['a_point', 'b_point']
        a_names = ['a{}'.format(i) for i in range(len(a_points))]
        b_names = ['b{}'.format(i) for i in range(len(b_points))]
        names = pd.DataFrame({'a_name': a_names, 'b_name': b_names})
        points = pd.concat([names, points], axis=1)
        points = points[points_order]
    elif len(points.columns) == 4:
        a_names = points[points.columns[0]].tolist()
        a_points = points[points.columns[1]].tolist()
        b_names = points[points.columns[2]].tolist()
        b_points = points[points.columns[3]].tolist()
        points.columns = points_order
   
    # Check whether point locations are unique
    unique_names = []
    unique_points = []
    a_names, _, unique_names, unique_points = find_unique_named_points(
        a_names, a_points, unique_names, unique_points)
    b_names, _, unique_names, unique_points = find_unique_named_points(
        b_names, b_points, unique_names, unique_points)    
    # Check whether names for unique points are unique
    if len(unique_names) != len(set(unique_names)):
        pass
   
    ####################################################################
    # Need to add a system here for providing unique names to repeated #
    # names with stepping integers (e.g., name, name_1, name_2, ...)   #
    ####################################################################
    
    # Add unique points to graph as nodes
    named_unique_points = list(zip(unique_names, unique_points))
    if verbose:
        chunk_time = time()
    unplaced_points, node_aliases = connect_points_to_closest_edges(
        G=G, points=named_unique_points, search_distance=search_distance, 
        sindex=sindex, points_to_nodes=points_to_nodes,
        accessibility_functions=accessibility_functions, verbose=verbose)
    if verbose:
        print('connect to edges time: {}'.format(time()-chunk_time))
        chunk_time = time()
        print('{} points placed on edges'.format(len(named_unique_points)-len(unplaced_points)))
    
    # Replace points names with any aliases 
    a_names = [node_aliases[x] if x in node_aliases else x for x in a_names]
    b_names = [node_aliases[x] if x in node_aliases else x for x in b_names]
    
    # Pair up nodes
    routing_pairs = list(zip(a_names, b_names)) 
    
    # Route between pairs
    routes = route_node_pairs(routing_pairs, G, weight=weight, both_ways=both_ways)  
    
    if verbose:
        print('routing time: {}'.format(time()-chunk_time))
        chunk_time = time()
        print('{} routes found'.format(len(routes)))
    
    # Define default summaries
    default_summaries = OrderedDict(
        [('geometry',  (lambda x: MultiLineString(x) if len(x) > 0 else None, 'geometry')),
         ('length', (lambda x: sum(x) if len(x) > 0 else np.inf, 'length'))])
    
    weight_summary = OrderedDict(
        [('wgt_len',(lambda x: sum(x) if len(x) > 0 else np.inf, 'wgt_len'))])
    
    # Add weight sum if weights are used for routing
    
    if weight is not 'length':
        # Calculate weighted length attributes for all edges
        wgt_lens = {}
        for edge in G.edges(data=True, keys=True):
            weight = edge[-1]['weight'] if 'weight' in edge[-1] else 1
            wgt_lens[edge[0:3]] = edge[-1]['length'] * weight
        nx.set_edge_attributes(G, wgt_lens, 'wgt_len')

        # Add weight summary to summaries dictionary
        default_summaries.update(weight_summary)
        
    # Use default summary alone if no other summaries specified
    if additional_summaries is None:
        summaries = default_summaries
    # Otherwise, append other summaries
    else:
        default_summaries.update(additional_summaries)
        summaries = default_summaries
    # Make a DataFrame to hold summaries
    route_summaries = pd.DataFrame(columns=list(summaries.keys()) + ['log'])
    # Summarize attributes along routes
    for route in routes:
        if not summarize_links:
            if isinstance(route, list):
                route = route[1:-1]
        _, summary = collect_route_attributes(route, G, summaries)
        route_summaries = route_summaries.append(summary, ignore_index=True)  
    if verbose:
        print('summary time: {}'.format(time()-chunk_time))   
    if both_ways:
        # Split forward and reverse columns apart
        forward_summaries = route_summaries.head(len(points)).reset_index(drop=True)
        backward_summaries = route_summaries.tail(len(points)).reset_index(drop=True)
        # Add prefixes to the reverse column names        
        backward_summaries.columns = ['rev_' + column for column in backward_summaries.columns]
        # Concatinate them side-by-side
        route_summaries = pd.concat([forward_summaries, backward_summaries], axis=1)
    # Concatinate with original points
    return_dataframe = pd.concat([points, route_summaries], axis=1) 
    # summary_columns = [x for x in list(return_dataframe) if x not in points_order]
    # return_dataframe = return_dataframe[points_order + remaining]
    unrouted_pairs = [(i, x) for i, x in enumerate(routes) if isinstance(x, str)]
    return return_dataframe, unrouted_pairs
    

def collect_route_attributes(route, G, summaries=None):
    """Collect attributes of edges along a route defined by nodes.

    Parameters
    ----------
    route : :obj:`list`
        List nodes forming route
    G : :class:`networkx.Graph`
        Graph containing route
    summaries : :class:`OrderedDict`, optional, default = None
        Keys are names for summaries, uses as keys in ``collected_summaries``.\
        Values are tuples with the first value being function for calculating\
        the requested summary, and the second value being the name for the\
        edge attribute from which the function makes these calulations.\
        Functions must take a single list-type parameter and operate on the\
        types of attributes for which they are called.

        If default, will return summary of geometry and geometric route length

    Returns
    ----------
    collected_attributes : :class:`numpy.ndarray`
        Structured array containing attributes for each edge along the route.\
        Column names are attributes defined by the keyes of ``summaries``.

    collected_summaries : :obj:`dict`
        Keys are attributes defined in the keys of ``summaries``. Values are\
        products of the functions defined in the values of ``summaries``.
    """
    default_summaries = OrderedDict(
        [('route',  (lambda x: MultiLineString(x) if len(x) > 0 else None, 'geometry')),
         ('rt_len', (lambda x: sum(x) if len(x) > 0 else np.inf, 'length'))])

    if not summaries:
        summaries = default_summaries
    else:
        default_summaries.update(summaries)

    # Get data from edges along route
    route_data = [G.get_edge_data(u, v) for u,v, in list(zip(route[:-1], route[1:]))]    
    # Make a structured array to store collected attributes
    attribute_fields = dict(zip(summaries.keys(), ['object'] * len(summaries)))
    collected_attributes = empty_array(len(route_data), attribute_fields)
    # Iterate through edges along route
    for i, edge in enumerate(route_data):       
        # If there are parallel edges, select shortest one
        if edge is not None:
            if len(edge) > 1:
                keys = []
                lengths = []
                for key, data in edge.items():
                    keys.append(key)
                    lengths.append(data['length'])
                # Identify the shorest option
                _, j = min((length, j) for (j, length) in enumerate(lengths))
                # Remove all dictionary elements except that one
                [route_data[i].pop(x) for x in list(route_data[i]) if x != keys[j]]
            
            # Collect each attribute
            for name, (_, attribute) in summaries.items():
                # Access whatever key remains in the edge dictionary
                for key in edge.keys():
                    if isinstance(attribute, tuple):
                        attributes = []
                        for a in attribute:
                            if a in edge[key]:
                                attributes.append(edge[key][a])
                            else:
                                attributes.append(edge[key][None])
                        collected_attributes[name][i] = tuple(attributes)
                    else:
                        if attribute in edge[key]:
                            collected_attributes[name][i] = edge[key][attribute]
                               
    # Summarize collected attributes
    collected_summaries = []
    for name, (function, attribute)  in summaries.items(): 
        attribute_list = collected_attributes[name].tolist()
        # Remove None values
        attribute_list = [x for x in attribute_list if x is not None]
        collected_summaries.append((name, function(attribute_list)))
    # Add a log entry to maintain original route info
    collected_summaries.append(('log', route))
    # Convert summaries into an ordered dict
    collected_summaries = OrderedDict(collected_summaries)
    
    return collected_attributes, collected_summaries


def route_geometry(geometries):
    """Collect geometries of edges along route. Return None if no edges.

    """
    if len(geometries) > 0:
        x = MultiLineString(geometries)
        return x
    else:
        return None


def route_length(lengths):
    """Collect lengths of edges along route. Return None if no edges.

    """
    if len(lengths) > 0:
        x = sum(lengths)
        return x
    else:
        return np.inf


def find_unique_named_points(names, points, unique_names, unique_points):
    """Find points that are geometrically unique.


    """
    for i, point in enumerate(points):
        if len(unique_names) == 0:
            unique_names.append(names[i])
            unique_points.append(point)
            continue
        unique = True
        for j, unique_point in enumerate(unique_points):
            if unique_point.almost_equals(point, decimal=0):
                unique = False
                names[i] = unique_names[j]
        if unique:
            unique_names.append(names[i])
            unique_points.append(point)
    return names, points, unique_names, unique_points


def build_nodes_from_edges(edges, search_distance=1):
    """Identify common nodes at the ends of LineString edges.
 
    LineString ends are considered equal (i.e., joined at the same network node)
    within the specified search distance.

    Outputs can be used as inputs for OSMnx 'gdfs_to_graph' function.

    Parameters
    ----------
    edges : :class:`geopandas.GeoDataFrame`
        LineString GeoDataFrame to be interpreted as graph edges
    search_distance : :obj:`float`, optional, default = 1.0
        Distance within which edge endpoints will be considered identical nodes

    Returns
    ----------
    nodes : :class:`geopandas.GeoDataFrame`
        New nodes identified at the endpoints of edges

    edges : :class:`geopandas.GeoDataFrame`
        Edges with 'u' and 'v' columns corresponding to nodes IDs
    """

    # Make an empty dataframe to store unique nodes as they are identified
    nodes = pd.DataFrame(columns=['geometry'])

    # Make a dataframe of edges with edge geometry, node geometries, and empty columns for node ids
    original_edges = edges
    edges = gpd.GeoDataFrame(geometry=edges['geometry'], columns=['u_geom','v_geom','u','v'], crs=edges.crs)
    ends = edges['geometry'].apply(lambda x: endpoints(x))
    edges['u_geom'] = ends.apply(lambda x: x[0])
    edges['v_geom'] = ends.apply(lambda x: x[1])

    # Make a spatial index of edges
    edges_sindex = edges.sindex

    # Iterate through edges
    for e in edges.itertuples():

        # Iterate through edge endpoint, i
        for i_end, i_geom in zip(('u','v'), (e.u_geom, e.v_geom)):
            
            # See if a node isn't assigned yet
            if np.isnan(edges.at[e.Index, i_end]):

                # Assign it the next node
                i_node = len(nodes)
                nodes = nodes.append({'geometry':i_geom}, ignore_index=True)
                edges.at[e.Index, i_end] = i_node

                # Figure out which edge ends should also be assigned this node
                i_buffer = i_geom.buffer(search_distance * 2)
                nearby_edge_ids = list(edges_sindex.intersection(i_buffer.bounds))
                nearby_edges = edges.iloc[nearby_edge_ids]
                nearby_edges = nearby_edges[nearby_edges.intersects(i_buffer)].copy()

                # remove the current edge from consideration
                nearby_edges.drop([e.Index], inplace=True)

                # Iterate through nearby edges
                for j in nearby_edges.itertuples():
                
                    # Iterate through nearby edge endpoints, k
                    for k_end, k_node, k_geom in zip(('u','v'), (j.u, j.v), (j.u_geom, j.v_geom)):
                        
                        # See if a node isn't assigned yet
                        if np.isnan(edges.at[j.Index, k_end]):

                            # See if i and k are approximatly equal
                            if i_geom.distance(k_geom) <= search_distance:

                                # If yes, assign it the same node as i                               
                                edges.at[j.Index, k_end] = i_node

    nodes = gpd.GeoDataFrame(data=nodes, geometry='geometry', crs=edges.crs)
    nodes.gdf_name = ''
    nodes['x'] = nodes['geometry'].apply(lambda x: x.x)
    nodes['y'] = nodes['geometry'].apply(lambda x: x.y)
    nodes['osmid'] = nodes.index
    original_edges['u'] = edges['u']
    original_edges['v'] = edges['v']
    edges = original_edges

    return nodes, edges


def fill_missing_graph_geometries(G):
    """Create missing edge geometries by connecting u and v nodes.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        Graph with missing edge geometry attributes

    Returns
    ----------
    G : :class:`networkx.Graph`
        Graph with filled edge geometry attributes
    """
    nodes_x = nx.get_node_attributes(G, 'x')
    nodes_y = nx.get_node_attributes(G, 'y')
    for u, v, key, data in G.edges(data=True, keys=True):
        if 'geometry' not in data:
            G[u][v][key]['geometry'] = sh.geometry.LineString([(nodes_x[u],nodes_y[u]),
                                               (nodes_x[v],nodes_y[v])])

    return G


def make_backward_edges(edges, twoway_column='oneway', twoway_id='False'):
    """Create a duplicate edge in the opposite direction for every two-way edge
    
    """
    
    edges = edges.copy()

    # Get two-way edges
    two_way_edges = edges[edges[twoway_column] == twoway_id].copy()

    # Flip endpoint IDs
    if 'u' in two_way_edges.columns and 'v' in two_way_edges.columns:
        u = two_way_edges['u']
        v = two_way_edges['v']
        two_way_edges['u'] = v
        two_way_edges['v'] = u   
    if 'to' in two_way_edges.columns and 'from' in two_way_edges.columns:
        to = two_way_edges['to']
        _from = two_way_edges['from']
        two_way_edges['to'] = _from
        two_way_edges['from'] = to

    # Flip geometry    
    two_way_edges['geometry'] = two_way_edges['geometry'].apply(lambda x: sh.geometry.LineString(x.coords[::-1]))

    # Append flipped edges back onto edges
    edges = edges.append(two_way_edges, ignore_index=True)
    
    return edges


def remove_extraneous_nodes(G):
    """Simplify a network by removing extraneous nodes and welding edge geometries

    G : networkx graph
    """
    
    # Use a copy of the graph
    G = G.copy()

    # Identify paths that could be simplified
    paths_to_simplify = ox.simplify.get_paths_to_simplify(G, strict=True)

    for path in paths_to_simplify:

        # Construct segment endpoints
        segment_endpoints = [(path[i], path[i + 1]) for i in range(0, len(path) - 1)]

        # Gather segment attributes
        segments = [G.get_edge_data(u, v, 0) for u, v in segment_endpoints]

        # Remove segments whose attributes are None
        segments = [x for x in segments if x] 

        # Combine attribute dictionaries
        new_edge = merge_dictionaries(segments)

        # For non-geometry attributes, collapse like attributes
        for key, value in new_edge.items():
            if key not in ['geometry','length']:
                # Ensure that lists are flat
                value = flatten(value)

                # Collapse similar values
                value = list(set(value))

                # Extract from list if only one value
                if len(value) == 1:
                    value = value[0]
                new_edge[key] = value

        # Weld together geometries
        new_edge['geometry'] = merge_ordered_lines(new_edge['geometry'])

        # Calculate new length
        new_edge['length'] = new_edge['geometry'].length

        # Insert new edge
        G.add_edge(path[0], path[-1], **new_edge)

    # Once all new edges are created, remove old edges
    for path in paths_to_simplify:
        # Construct segment endpoints
        segment_endpoints = [(path[i], path[i + 1]) for i in range(0, len(path) - 1)]
        # Remove each segment
        for (u, v) in segment_endpoints:
            G.remove_edge(u, v)

    # Remove orphan nodes (with no edges connecting anymore)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    return G


def subgraph_by_edge_attribute(G, attribute, values, simplify=True):
    """Make a subgraph of edges that have any of the supplied values
    
    G : networkx graph
    
    attribute : str
    
    values : list

    simply : bool
        True : Extraneous nodes are removed and edge geometries are merged
        False : Original nodes and edge geometries are retained
    
    returns : networkx graph
    """
    
    # Get dictionary of edge attributes
    edge_dict = nx.get_edge_attributes(G, attribute)

    # Identify edges with these highway tags
    def list_overlap(value, values):
        # ensure that value is a list
        value = listify(value)
        # find intersection
        intersection = [x for x in value if x in values]
        if len(intersection) > 0:
            return True
        else:
            return False
    edges_subset = [key for key, value in edge_dict.items() if list_overlap(value, values)]

    # Make subgraph from these edges
    G = G.edge_subgraph(edges_subset).copy()

    if simplify:
        G = remove_extraneous_nodes(G)
    
    return G


##### The following functions are for modeling turns within intersections #####
##### Docstrings needs to be fleshed out #####

def explode_node(G, node):
    """Create non-geometric edges between all 'in' and 'out' edges at each node
    """
    
    # Find all entering edges
    orig_in_edges = list(G.in_edges(node, keys=True))

    # Find all exiting edges
    orig_out_edges = list(G.out_edges(node, keys=True))

    # Get attributes from existing node
    node_attributes = G.node[node]

    def add_node(G, edge, counter, direction):
        # Add a new node
        new_node = '{}_{}{}'.format(node, direction, counter)
        G.add_node(new_node, attr_dict=node_attributes)
        # Add a new edge
        edge_attributes = G.get_edge_data(*edge)
        u, v, key = edge
        if direction == 'in': 
            new_edge = (u, new_node, key)
        elif direction == 'out':
            new_edge = (new_node, v, key)
        # Add the new edge into the graph
        G.add_edge(*new_edge)
        # Add original edge attributes
        new_u, new_v, new_key = new_edge
        for attr in edge_attributes:
            G[new_u][new_v][new_key][attr] = edge_attributes[attr]
        # Remove old edge
        G.remove_edge(*edge)
        # Advance counter
        counter += 1
        return new_edge, new_node, counter 
    
    in_nodes = []
    in_edges = []
    in_i = 0 # start counter 
    for orig_in_edge in orig_in_edges:
        # Add a new node
        in_edge, in_node, in_i = add_node(G, orig_in_edge, in_i, 'in')
        in_nodes.append(in_node)
        in_edges.append(in_edge)

    out_nodes = []
    out_edges = []
    out_i = 0 # start counter 
    for orig_out_edge in orig_out_edges:
        # Add a new node
        out_edge, out_node, out_i = add_node(G, orig_out_edge, out_i, 'out')
        out_nodes.append(out_node)
        out_edges.append(out_edge)
    
    # Remove old node
    G.remove_node(node)
    
    # Connect new nodes with edges
    inter_edges = []
    for in_node in in_nodes:
        for out_node in out_nodes:
            new_edge = (in_node, out_node, 0)
            G.add_edge(*new_edge)
            inter_edges.append(new_edge)
          
    return in_edges, out_edges


def classify_turns(G, in_edges, out_edges, straight_angle=20):
    """Classify turning movements between 'in' and 'out' edges at intersections
    """
    # Only proceed if there are 'in' edges
    if len(out_edges) > 0:
        # Iterate through in edges
        for in_edge in in_edges:
            in_u, in_v, in_key = in_edge
            # Get the azimuth of 'in' edge
            in_geom = G[in_u][in_v][in_key]['geometry']
            in_geom_len = in_geom.length
            in_azimuth = sp.azimuth_at_distance(in_geom, in_geom_len) # Azimuth at the entering line's end

            # Get the azimuths of each 'out' edge
            out_edge_azimuths = []
            for out_edge in out_edges:
                out_u, out_v, out_key = out_edge            
                out_geom = G[out_u][out_v][out_key]['geometry']
                # Azimuth at the out edge start
                out_azimuth = sp.azimuth_at_distance(out_geom, 0)
                relative_azimuth = out_azimuth - in_azimuth
                out_edge_azimuths.append(relative_azimuth)

            # Sort edges and azimuths by azimuth
            try:
                out_edges, out_edge_azimuths = zip(*sorted(zip(out_edges, out_edge_azimuths), key=lambda x: x[1]))
            except:
                print(out_edges, out_edge_azimuths)

            # Classify turn directions
            turn_directions = [classify_turn_direction(x, straight_angle) for x in out_edge_azimuths]

            # Classify turn proximity
            turns_proximities = classify_turn_proximity(turn_directions)

            # Assign turn label to intersection edges
            for (out_u, _, _), turn_direction, turn_proximity in zip(out_edges, turn_directions, turns_proximities):
                G[in_v][out_u][0]['turn_direction'] = turn_direction
                G[in_v][out_u][0]['turn_proximity'] = turn_proximity
                

def classify_turn_direction(relative_azimuth, straight_angle=20):
    """Classify turn directions based on a relative azimuth     
    """
    a, b, c, d = (
        0 + straight_angle, 
        180 - straight_angle, 
        180 + straight_angle, 
        360 - straight_angle)
    if (relative_azimuth > d) or (relative_azimuth < a):
        return 'U'
    elif (relative_azimuth > a) and (relative_azimuth < b):
        return 'right'
    elif (relative_azimuth > b) and (relative_azimuth < c):
        return 'straight'
    elif (relative_azimuth > c) and (relative_azimuth < d):
        return 'left'
    
def classify_turn_proximity(turn_directions):
    """Classify turn proximity based on a list of turn directions  
    """
    # Enumerate turns
    enum_turns = list(enumerate(turn_directions))
    # Identify U-turns
    u_turns = {i: 'near' for i, x in enum_turns if x == 'U'}
    # Remove U-turns
    enum_turns = [(i, x) for i, x in enum_turns if x != 'U']
    # First and last turns are 'near', all others are 'far
    turn_proximity = ['far'] * len(enum_turns)
    if len(turn_proximity) > 0:
        turn_proximity[0] = 'near'
        turn_proximity[-1] = 'near'
        # Enumerate proximities
        enum_proximity = [(i, x) for (i, _), x in zip(enum_turns, turn_proximity)]
        # Convert into dictionary
        enum_proximity = dict(enum_proximity)      
    else:
        enum_proximity = {}
    # Combine non-U-turns and U-turns
    turn_proximities = {**enum_proximity, **u_turns}
    # List proximities in order
    turn_proximities = [turn_proximities[key] for key in sorted(turn_proximities.keys())]
    return turn_proximities


def create_intersection_edges(G, straight_angle=20):
    """Add non-geometric edges to represent turns at intersections
    """
    G = G.copy()
    for node in list(G.nodes()):
        # Explode the node into edges
        in_edges, out_edges = explode_node(G, node)
        # Classify turns on edges
        classify_turns(G, in_edges, out_edges, straight_angle=straight_angle)
    return G


