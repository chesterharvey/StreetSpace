"""Functions to manipulate and analyze NetworkX graphs."""

################################################################################
# Module: network.py
# Description: Functions to manipulate and analyze NetworkX graphs.
# License: MIT
################################################################################

import networkx as nx
import numpy as np
from rtree import index
from networkx import Graph, DiGraph, MultiGraph, MultiDiGraph
from time import time
from pandas import DataFrame
from geopandas import GeoDataFrame

from .geometry import *


def closest_point_along_network(search_point, G, search_distance=None, 
    sindex=None):
    """
    Find the closest point along the edges of a NetworkX graph with Shapely 
    LineString geometry attributes in the same coordinate system.

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

    Returns
    -------
    closest_point : :class:`shapely.geometry.Point`
        Location of closest point
    index : :obj:`tuple`
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
        edge_indices = [x for x in sindex.intersection(search_bounds, 
                                                       objects='raw')]
        # Collect geometries that intersect the search bounds
        if isinstance(G, (MultiGraph, MultiDiGraph)):
            edge_geometries = [G.get_edge_data(u, v, key)['geometry'] 
                               for u, v, key in edge_indices
                               # only query an edge if it exists; some edges may
                               # have been deleted since the sindex was made
                               if G.get_edge_data(u, v, key)]
        else:
            edge_geometries = [G.get_edge_data(u, v)['geometry'] 
                               for u, v in edge_indices
                               # only query an edge if it exists; some edges may
                               # have been deleted since the sindex was made
                               if G.get_edge_data(u, v)]
        # Construct list of edges as (index, geometry) tuples
        edges = list(zip(edge_indices, edge_geometries))
    elif search_distance:
        # Construct search bounds around the search point
        search_area = search_point.buffer(search_distance)
        # Collect edges that intersect the search area as (index, geometry) tuples
        if isinstance(G, (MultiGraph, MultiDiGraph)):
            edges = G.edges(keys=True, data='geometry')
            edges = [((u, v, key), geometry) for u, v, key, geometry
                     in edges if geometry.intersects(search_area)]
        else:
            edges = G.edges(data='geometry')
            edges = [((u, v), geometry) for u, v, geometry
                     in edges if geometry.intersect(search_area)]
    else:
        # Collect all edges as (index, geometry) tuples
        if isinstance(G, (MultiGraph, MultiDiGraph)):
            edges = G.edges(keys=True, data='geometry')
            edges = [((u, v, key), geometry) for u, v, key, geometry in edges]
        else:
            edges = G.edges(data='geometry')
            edges = [((u, v), geometry) for u, v, geometry in edges]
    # Feed edges to general function for finding closest point among lines
    return closest_point_along_lines(search_point, edges)


# def insert_node_along_edge(G, edge, node_point, node_name):
#     """Insert a node along an edge with a geometry attribute.

#     ``edge`` must have a LineString geometry attribute
#     which will be split at the location of the new node.

#     If `G` is :class:`networkx.DiGraph` or :class:`networkx.MultiDiGraph`,
#     the new node will split edges in both directions.

#     Parameters
#     ----------
#     G : :class:`networkx.Graph`, :class:`networkx.DiGraph`,\
#     :class:`networkx.MultiGraph` or :class:`networkx.MultiDiGraph`
#         Graph into which new node will be inserted. Each edge of `G` must\
#         have a :class:`shapely.geometry.LineString` geometry attribute.

#     edge : :obj:`tuple`
#         * if G is Graph or DiGraph: (u, v)
#         * if G is MultiGraph or MultiDiGraph: (u, v, key)

#     node_point : :class:`shapely.geometry.Point`
#         Geometric location of new node

#     node_name : :obj:`str`
#         Name for new node
#     """
#     if isinstance(G, (MultiGraph, MultiDiGraph)):
#         u, v, key = edge
#     else:
#         u, v = edge
#         key = None
#     # get attributes from the existing nodes
#     u_attrs = G.node[u]
#     v_attrs = G.node[v]
#     # assemble attributes for the new node
#     new_node_attrs = {'geometry': node_point, 
#                       'x': node_point.x,
#                       'y': node_point.y}
#     if isinstance(G, (MultiGraph, MultiDiGraph)):
#         if G.has_edge(u, v, key):
#             # get attributes from existing edge
#             attrs = G.get_edge_data(u, v, key)
#             original_geom = attrs['geometry']
#             # delete existing edge
#             G.remove_edge(u, v, key)
#             # specify nodes for the new edges
#             G.add_node(u, **u_attrs)
#             G.add_node(v, **v_attrs)
#             G.add_node(node_name, **new_node_attrs)        
#             # construct attributes for first new edge            
#             attrs['geometry'] = segment(original_geom, 
#                                         endpoints(original_geom)[0], 
#                                         node_point)
#             if 'length' in attrs:
#                 attrs['length'] = attrs['geometry'].length
#             # specify new edge
#             G.add_edge(u, node_name, key = 0, **attrs)
#             # construct attributes for second new edge
#             attrs['geometry'] = segment(original_geom,
#                                         node_point, 
#                                         endpoints(original_geom)[1])
#             if 'length' in attrs:
#                 attrs['length'] = attrs['geometry'].length
#             # specify new edge   
#             G.add_edge(node_name, v, key = 0, **attrs)    
#         if G.has_edge(v, u, key):
#             # get attributes from existing edge
#             attrs = G.get_edge_data(v, u, key)
#             original_geom = attrs['geometry']
#             # delete existing edge
#             G.remove_edge(v, u, key)
#             # specify nodes for the new edges
#             G.add_node(u, **u_attrs)
#             G.add_node(v, **v_attrs)
#             G.add_node(node_name, **new_node_attrs)
#             # construct attributes for first new edge
#             attrs['geometry'] = segment(original_geom, 
#                                         endpoints(original_geom)[0], 
#                                         node_point)
#             if 'length' in attrs:
#                 attrs['length'] = attrs['geometry'].length
#             # specify new edge
#             G.add_edge(v, node_name, key = 0, **attrs)
#             # construct attributes for second new edge
#             attrs['geometry'] = segment(original_geom, 
#                                         node_point, 
#                                         endpoints(original_geom)[1])
#             if 'length' in attrs:
#                 attrs['length'] = attrs['geometry'].length
#             # specify new edge   
#             G.add_edge(node_name, u, key = 0, **attrs)
#     else:
#         if G.has_edge(u, v): # examine the edge from u to v
#             # get attributes from existing edge
#             attrs = G.get_edge_data(u, v)
#             original_geom = attrs['geometry']
#             # delete existing edge
#             G.remove_edge(u, v)
#             # specify nodes for the new edges
#             G.add_node(u, **u_attrs)
#             G.add_node(v, **v_attrs)
#             G.add_node(node_name, **new_node_attrs)
#             # construct attributes for first new edge
#             attrs['geometry'] = segment(original_geom, 
#                                         endpoints(original_geom)[0], 
#                                         node_point)
#             attrs['length'] = attrs['geometry'].length
#             G.add_edge(u, node_name, **attrs)
#             # construct attributes for second new edge
#             attrs['geometry'] = segment(original_geom, 
#                                         node_point, 
#                                         endpoints(original_geom)[1])
#             attrs['length'] = attrs['geometry'].length
#             G.add_edge(node_name, v, **attrs)
#         if G.has_edge(v, u): # examine the edge from v to u
#             # get attributes from existing edge
#             attrs = G.get_edge_data(v, u)
#             original_geom = attrs['geometry']
#             # delete existing edge
#             G.remove_edge(v, u)
#             # specify nodes for the new edges
#             G.add_node(u, **u_attrs)
#             G.add_node(v, **v_attrs)
#             G.add_node(node_name, **new_node_attrs)
#             # construct attributes for first new edge
#             attrs['geometry'] = segment(original_geom, 
#                                         endpoints(original_geom)[0], 
#                                         node_point)
#             if 'length' in attrs:
#                 attrs['length'] = attrs['geometry'].length
#             # specify new edge
#             G.add_edge(v, node_name, **attrs)
#             # construct attributes for second new edge
#             attrs['geometry'] = segment(original_geom, 
#                                         node_point, 
#                                         endpoints(original_geom)[1])
#             if 'length' in attrs:
#                 attrs['length'] = attrs['geometry'].length
#             # specify new edge   
#             G.add_edge(node_name, u, **attrs)


def insert_node_along_edge(G, edge, node_point, node_name):
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
    """  
    u_attrs = G.node[edge[0]]
    v_attrs = G.node[edge[1]]
    # Get attributes from existing edge
    attrs = G.get_edge_data(*edge)
    original_geom = attrs['geometry']
    # Delete existing edge
    G.remove_edge(*edge)
    # Assemble attributes for the new node
    new_node_attrs = {'geometry': node_point, 
                      'x': node_point.x,
                      'y': node_point.y}
    # Add new nodes
    G.add_node(edge[0], **u_attrs)
    G.add_node(edge[1], **v_attrs)
    G.add_node(node_name, **new_node_attrs)
    # Assemble attributes for first new edge            
    attrs['geometry'] = segment(original_geom, 
                                endpoints(original_geom)[0], 
                                node_point)
    if 'length' in attrs:
        attrs['length'] = attrs['geometry'].length
    # Add first new edge
    G.add_edge(edge[0], node_name, key = 0, **attrs)
    # Assemble attributes for second new edge
    attrs['geometry'] = segment(original_geom,
                                node_point, 
                                endpoints(original_geom)[1])
    if 'length' in attrs:
        attrs['length'] = attrs['geometry'].length
    # Add second new edge
    G.add_edge(node_name, edge[1], key = 0, **attrs)


def connect_points_to_closest_edges(G, points, search_distance=None, 
    sindex=None, return_unplaced=False, points_to_nodes=True):
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
    unplaced_points =[]
    # u_point refers to the off-the-graph point which is being connected
    # v_point refers to the on-graph point where the connection is made 
    for name, u_point in points:
        # u_name = '{}u'.format(name)
        # v_name = '{}v'.format(name)
        if points_to_nodes:
            u_name = name
            v_name = '{}_link'.format(name)
        else:
            v_name = name
        v_point, edge, _ = (
            closest_point_along_network(u_point, G,
                                        search_distance = search_distance, 
                                        sindex = sindex))
        if v_point:
            if isinstance(G, (MultiGraph, MultiDiGraph)):
                if G.has_edge(*edge):
                    insert_node_along_edge(G, edge, v_point, v_name)
                reverse_edge = (edge[1], edge[0], edge[2])
                if G.has_edge(*reverse_edge):
                    insert_node_along_edge(G, reverse_edge, 
                                           v_point, v_name)
            else:
                if G.has_edge(*edge):
                    insert_node_along_edge(G, edge, v_point, v_name)
                reverse_edge = (edge[1], edge[0])
                if G.has_edge(*reverse_edge):
                    insert_node_along_edge(G, reverse_edge, 
                                           v_point, v_name)
            # If off-the-graph points are being inserted into the graph as
            # nodes, add them and connecting edges
            if points_to_nodes:
                attrs = {'geometry': u_point,
                         'x': u_point.x,
                         'y': u_point.y}
                G.add_node(u_name, **attrs)
                attrs = {'geometry': LineString([u_point, v_point]),
                         'length': u_point.distance(v_point)}
                G.add_edge(u_name, v_name, key = 0, **attrs)
                # If graph is directed, add another edge going the other way
                if nx.is_directed(G):
                    attrs = {'geometry': LineString([v_point, u_point]),
                             'length': u_point.distance(v_point)}
                    G.add_edge(v_name, u_name, key = 0, **attrs)
        else:
            unplaced_points.append((u_name, u_point))
    if return_unplaced:
        return unplaced_points


# def connect_points_to_closest_edges(G, points, search_distance=None, 
#     sindex=None, return_unplaced=False):
#     """Connect points to a graph by inserting a node along their closest edge.

#     G : :class:`networkx.Graph`, :class:`networkx.DiGraph`,\
#     :class:`networkx.MultiGraph` or :class:`networkx.MultiDiGraph`
#         Graph into which new node will be inserted. Each edge of `G` must\
#         have a :class:`shapely.geometry.LineString` geometry attribute.
#     points : :obj:`list`
#         List of tuples with structure (point_name, point_geometry)
#     search_distance : :obj:`float`, optional, default = ``None``
#         Maximum distance to search for an edge from each point
#     sindex : :class:`rtree.index.Index`, optional, default = ``None``
#         Spatial index for `G`
#     return_unplaced : :obj:`bool`, optional, default = ``False``
#         If ``True``, will return points that are outside the search distance\
#         or have otherwise not been connected to the graph

#     Returns
#     ----------
#     :obj:`list`
#         Points not connected to the graph (if ``return_unplaced`` is ``True``)
#     """
#     unplaced_points =[]
#     for point_name, point_geometry in points:
#         connection_location, edge_info, _ = (
#             closest_point_along_network(point_geometry, G,
#                                         search_distance = search_distance, 
#                                         sindex = sindex))
#         if connection_location:
#             if isinstance(G, (MultiGraph, MultiDiGraph)):
#                 insert_node_along_edge(G, edge_info, connection_location, 
#                                        point_name)
#             else:
#                 insert_node_along_edge(G, edge_info, connection_location, 
#                                        point_name)
#         else:
#             unplaced_points.append((point_name, point_geometry))
#     if return_unplaced:
#         return unplaced_points


def graph_sindex(G, save_path=None):
    """Create a spatial index from a graph with geometry attributes.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        Must include :class:`shapely.geometry.LineString` geometry attributes 

    Returns
    ----------
    :class:`rtree.index.Index`
        Spatial index
    """
    if save_path:
        idx = index.Index(save_path)
    else:
        idx = index.Index()
    if isinstance(G, (MultiGraph, MultiDiGraph)):
        edges = G.edges(keys=True, data='geometry')
        for i, (u, v, key, geometry) in enumerate(edges):
            idx.insert(i, geometry.bounds, (u, v, key))
    else:
        edges = G.edges(data='geometry')
        for i, (u, v, geometry) in enumerate(edges):
            idx.insert(i, geometry.bounds, (u, v))
    if save_path:
        idx.close
    return idx


# def add_edge_attribute(G, attribute, value):
#     """Add or modify an edge attribute based on existing attributes.

#     Adds an attribute to every edge in ``G``.

#     Parameters
#     ----------
#     G : :class:`networkx.Graph`
#         Graph to which to add attribute.

#     attribute : :obj:`str`
#         Attribute name

#     value : any
#         Value for attribute to take. May be expressed in terms of an existing\
#         attribute by calling it as a key of ``data``.\
#         For example: ``data['name']``
#     """
#     for u, v, key, data in G.edges(data=True, keys=True):
#         G[u][v][key][attribute] = value

def route_node_pairs(node_pairs, G, weight=None, both_ways=False):
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
    def route(G, O, D, weight=None):
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


def collect_route_attributes(route, G, summaries):
    """Collect attributes of edges along a route defined by nodes.

    Parameters
    ----------
    route : :obj:`list`
        List nodes forming route
    G : :class:`networkx.Graph`
        Graph containing route
    summaries : :obj:`dict`
        Keys specify attributes to be collected. Values are functions with\
        which each attributed will be summarised. Functions must take a\
        single list-type parameter and be designed to operate on the types\
        of attributes for which they are called.

    Returns
    ----------
    collected_attributes : :class:`numpy.ndarray`
        Structured array containing attributes for each edge along the route.\
        Column names are attributes defined by the keyes of ``summaries``.

    collected_summaries : :obj:`dict`
        Keys are attributes defined in the keys of ``summaries``. Values are\
        products of the functions defined in the values of ``summaries``.

    """
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
            for attribute in summaries.keys():
                # Access whatever key remains in the edge dictionary
                for key in edge.keys():
                    if attribute in edge[key]:
                        collected_attributes[attribute][i] = edge[key][attribute]
    # Summarize collected attributes
    collected_summaries = {}
    for attribute, summary_function in summaries.items(): 
        attribute_list = collected_attributes[attribute].tolist()
        # Remove None values
        attribute_list = [x for x in attribute_list if x is not None]
        collected_summaries[attribute] = summary_function(attribute_list)                  
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
