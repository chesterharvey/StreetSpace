"""Functions to manipulate and analyze NetworkX graphs."""

################################################################################
# Module: network.py
# Description: Functions to manipulate and analyze NetworkX graphs.
# License: MIT
################################################################################

import networkx as nx
import numpy as np
from rtree import index

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
        if nx.is_directed(G):
            edge_geometries = [G.get_edge_data(u, v, key)['geometry'] 
                                for u, v, key in edge_indices]
        else:
            edge_geometries = [G.get_edge_data(u, v)['geometry'] 
                                for u, v in edge_indices]
        # Construct list of edges as (index, geometry) tuples
        edges = list(zip(edge_indices, edge_geometries))
    elif search_distance:
        # Construct search bounds around the search point
        search_area = search_point.buffer(search_distance)
        # Collect edges that intersect the search area as (index, geometry) tuples
        if nx.is_directed(G):
            edges = G.edges(keys=True, data='geometry')
            edges = [((u, v, key), geometry) for u, v, key, geometry
                     in edges if geometry.intersects(search_area)]
        else:
            edges = G.edges(data='geometry')
            edges = [((u, v), geometry) for u, v, geometry
                     in edges if geometry.intersect(search_area)]
    else:
        # Collect all edges as (index, geometry) tuples
        if nx.is_directed(G):
            edges = G.edges(keys=True, data='geometry')
            edges = [((u, v, key), geometry) for u, v, key, geometry in edges]
        else:
            edges = G.edges(data='geometry')
            edges = [((u, v), geometry) for u, v, geometry in edges]
    # Feed edges to general function for finding closest point among lines
    return closest_point_along_lines(search_point, edges)


def insert_node(G, u, v, node_point, node_name, key = None):
    """Insert a node into a graph with edge geometries.

    Each edge must have a LineString geometry attribute, which will be
    split at the location of the new node.

    If `G` is :class:`networkx.DiGraph` or :class:`networkx.MultiDiGraph`,
    the new node will split edges in both directions.

    Parameters
    ----------
    G : :class:`networkx.Graph`, :class:`networkx.DiGraph`,\
    :class:`networkx.MultiGraph` or :class:`networkx.MultiDiGraph`
        Graph into which new node will be inserted. Each edge of `G` must\
        have a :class:`shapely.geometry.LineString` geometry attribute.

    u : :obj:`int`
        First node ID for edge along which node is located

    v : :obj:`int`
        Second node ID for edge along which new node is located

    node_point : :class:`shapely.geometry.Point`
        Geometric location of new node

    node_name : :obj:`str`
        Name for new node

    key : :obj:`int`, optional, default = ``None``
        Key for edge along which node is being inserted

    Returns
    -------
    same type as `G`
        Copy of `G` with new node inserted

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
                                        endpoints(original_geom)[0], 
                                        node_point)
            attrs['length'] = attrs['geometry'].length
            G.add_edge(u, node_name, **attrs)
            # construct attributes for second new edge
            attrs['geometry'] = segment(original_geom, 
                                        node_point, 
                                        endpoints(original_geom)[1])
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
                                        endpoints(original_geom)[0], 
                                        node_point)
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge
            G.add_edge(v, node_name, **attrs)
            # construct attributes for second new edge
            attrs['geometry'] = segment(original_geom, 
                                        node_point, 
                                        endpoints(original_geom)[1])
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
                                        endpoints(original_geom)[0], 
                                        node_point)
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge
            G.add_edge(u, node_name, key = 0, **attrs)
            # construct attributes for second new edge
            attrs['geometry'] = segment(original_geom,
                                        node_point, 
                                        endpoints(original_geom)[1])
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge   
            G.add_edge(node_name, v, key = 0, **attrs)    
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
                                        endpoints(original_geom)[0], 
                                        node_point)
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge
            G.add_edge(v, node_name, key = 0, **attrs)
            # construct attributes for second new edge
            attrs['geometry'] = segment(original_geom, 
                                        node_point, 
                                        endpoints(original_geom)[1])
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge   
            G.add_edge(node_name, u, key = 0, **attrs)
        return G


def graph_sindex(G):
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
    idx = index.Index()
    if nx.is_directed(G):
        edges = G.edges(keys=True, data='geometry')
        for i, (u, v, key, geometry) in enumerate(edges):
            idx.insert(i, geometry.bounds, (u, v, key))
    else:
        edges = G.edges(data='geometry')
        for i, (u, v, geometry) in enumerate(edges):
            idx.insert(i, geometry.bounds, (u, v))
    return idx


def add_edge_attribute(G, attribute, value):
    """Add or modify an edge attribute based on existing attributes.

    Adds attribute directly to the input graph, G.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        Graph to which to add attribute.

    attribute : :obj:`str`
        Attribute name

    value : any
        Value for attribute to take. May be expressed in terms of an existing\
        attribute by calling it as a key of ``data``.\
        For example: ``data['name']``
    """
    for u, v, key, data in G.edges(data=True, keys=True):
        G[u][v][key][attribute] = value