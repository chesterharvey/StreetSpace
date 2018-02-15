"""Functions to manipulate and analyze NetworkX graphs."""

################################################################################
# Module: network.py
# Description: Functions to manipulate and analyze NetworkX graphs.
# License: MIT
################################################################################

from .geometry import *


def closest_point_along_network(G, search_point, search_distance,
    edges_sindex=None):
    """
    Find the closest point along the edges of a NetworkX graph with Shapely 
    LineString geometry attributes in the same coordinate system.

    Parameters
    ----------
    G : :class:`networkx.Graph`, :class:`networkx.DiGraph`, \
    :class:`networkx.MultiGraph` or :class:`networkx.MultiDiGraph`
        Graph along which closest point will be found. Each edge of `G` must\
        have a :class:`shapely.geometry.LineString` geometry attribute.

    search_point : :class:`shapely.geometry.Point`
        Point from which to search

    search_distance : :obj:`float`
        Maximum distance to search from the `search_point`

    edges_sindex : :class:`rtree.index.Index`, optional, default = ``None``
        Spatial index for edges of `G`

    Returns
    -------
    u : :obj:`int`
        First node ID for the edge along which the closest point is located
    v : :obj:`int`
        Second node ID for the edge along which the closest point is located
    key : :obj:`int`
        Key for the edge along which the closest point is located
    point : :class:`shapely.geometry.Point`
        Location of closest point
    """
    # extract edge indices and geometries from the graph
    edge_IDs = [i for i in G.edges]
    edge_geometries = [data['geometry'] for _, _, data in G.edges(data=True)]
    if edges_sindex is None:
        line_index = [data['geometry'] for _, _, data in G.edges(data=True)]
    # find the closest point for connection along the network 
    edge_ID, point = closest_point_along_lines(search_point, edge_geometries, 
        search_distance=search_distance, linestrings_sindex=edges_sindex)
    if edge_ID is not None:
        try: # will not return key if the network is DiGraph
            u, v  = edge_IDs[edge_ID]
            return u, v, None, point
        except:
            pass
        try: # will return key if network is MultiDiGraph
            u, v, key  = edge_IDs[edge_ID]
            return u, v, key, point
        except:
            return None, None, None, None
    else:
        return None, None, None, None


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
            G.add_edge(u = u, v = node_name, key = 0, **attrs)
            # construct attributes for second new edge
            attrs['geometry'] = segment(original_geom,
                                        node_point, 
                                        endpoints(original_geom)[1])
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
                                        endpoints(original_geom)[0], 
                                        node_point)
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge
            G.add_edge(u = v, v = node_name, key = 0, **attrs)
            # construct attributes for second new edge
            attrs['geometry'] = segment(original_geom, 
                                        node_point, 
                                        endpoints(original_geom)[1])
            if 'length' in attrs:
                attrs['length'] = attrs['geometry'].length
            # specify new edge   
            G.add_edge(u = node_name, v = u, key = 0, **attrs)
        return G