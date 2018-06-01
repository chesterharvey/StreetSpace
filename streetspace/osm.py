##############################################################################
# Module: osm.py
# Description: Functions to manipulate OSM data.
# License: MIT
##############################################################################

from pathlib import Path
import json
import re
import numpy as np
import re
import osmnx as ox
from .utils import *

def _key_in_set(key, keys, keys_regex=False):
    """Determine whether ``key`` is among ``keys``, with or without regex
    """
    if keys_regex:
        return any(re.compile(k).match(key) for k in keys)
    else:
        return key in keys

def _parse_osm_number(value, length=False, distance=False, speed=False, none_value=np.nan):
    """Parse OSM strings representing numbers into standardized floats.

    Parameters
    ----------
    value : :obj:`str`
        Raw value from OSM
    length : :obj:`bool`, optional, default=False
        * ``True`` : Value should be treated like a length;
        common length units (m, km, ft, in, mi) will be parsed and converted to meters
        * ``False`` : Units will be stripped but not parsed/converted
    distance : :obj:`bool`, optional, default=False
        * ``True`` : Value should be treated like a distance;
        common distance units (m, km, ft, in, mi) will be parsed and converted to kilometers
        * ``False`` : Units will be stripped but not parsed/converted
    speed : :obj:`bool`, optional, default=False
        * ``True`` : Value should be treated like a speed;
        common speed units (kph, mph) will be parsed and converted to kilometers/hr
        * ``False`` : Units will be stripped but not parsed/converted
    none_value : :obj:`float`, optional, default=np.nan
        Value to return if no number can be parsed.

    Returns
    -------
    :obj:`float`
        Parsed/converted number
    """
    # If value can be converted directly to a float, return that
    try:
        return float(value)
    except:               
        # Otherwise, if it can be split into two parts:
        if len(value.split()) == 2:
            value, unit = value.split()
            try:
                value = float(value)
                if length: # Converts to m
                    if unit in ('km', 'kilometre', 'kilometres', 'kilometer', 'kilometers'):
                        value = value * 0.001
                    if unit in ('ft', 'feet'):
                        value = value * 0.3048
                    if unit in ('in', 'inch', 'inches'):
                        value = value * 0.0254
                    if unit in ('mi', 'mile', 'miles'):
                        value = value * 0.000621371
                elif distance: # Converts to km
                    if unit in ('m', 'metre', 'metres', 'meter', 'meters'):
                        value = value * 1000
                    if unit in ('ft', 'feet'):
                        value = value * 0.0003048
                    if unit in ('in', 'inch', 'inches'):
                        value = value * 2.54e-5
                    if unit in ('mi', 'mile', 'miles'):
                        value = value * 1.60934
                elif speed: # Convert to kph
                    if unit in ('mph'):
                        value = value * 1.60934
                return float(value)
            except:
                try:
                    return float(value)
                except:
                    return none_value
        else:
            return none_value


def _identify_any_value_among_keys(tags_dict, keys, values, false_values=None,
    true_value=True, false_value=False, none_value=None, keys_regex=False, **unused):
    """Identify whether any of `values` is in any of `keys` in `tags_dict`.

    Parameters
    ----------
    tags_dict : :obj:`dict`
        Dictionary of tags related to OSM element

    keys : :obj:`set`
        Keys in ``tags_dict`` to search for ``values``.
        If any ``keys`` are not in ``tags_dict``, returns ``none_value``

    values : :obj:`set`
        Values to find within ``keys``

    false_values : :obj:`set`
        Potential tag values that explicitly indicate falsity (e.g., 'no')

    true_value : any type, optional, default = ``True``
        Value to return if any of ``keys`` are found in any of ``values``.

    false_value : any type, optional, default = ``False``
        Value to return if none of ``keys`` are found in any of ``values`` and
        any of ``false_values` are found in any of ``values``

    none_value : any type, optional, default = ``None``
        Value to return if none of ``keys`` or ``false_values`` are found in
        any of ``values``.

    keys_regex : :obj:`bool`, optional, default = ``False``
        * ``True`` : keys will be identified using regex

    Returns
    -------
    ``true_value``, ``false_value``, or ``none_value``

    """
    findings = []
    for key, value in tags_dict.items():
        if _key_in_set(key, keys, keys_regex=keys_regex):
            if value in values:
                findings.append(True)
            elif false_values:
                if value in false_values:
                    findings.append(False)
    if any(x is True for x in findings):
        return true_value
    elif any(x is False for x in findings):
        return false_value
    else:
        return none_value

def _identify_tag_combination(tags_dict, test_tags_dict, true_value=True, none_value=None,
    **unused):
    """Test whether there all of a particular combination of tags are in ``tags_dict``.

    tags_dict : :obj:`dict`
        Dictionary of tags related to OSM element

    test_tags_dict : :obj:`dict`
        Dictionary of tags (key:value pairs) that must be present in ``tags_dict``

    true_value : any type, optional, default = ``True``
        Value to return if all specified tags are found.

    none_value : any type, optional, default = ``None``
        Value to return if not all of the specified tags are found.

    Returns
    -------
    ``true_value`` or ``none_value``
    """
    findings = []
    for key, value in test_tags_dict.items():
        if key in tags_dict and value == tags_dict[key]:
            findings.append(True)
        else:
            findings.append(False)
    if all(findings):
        return true_value
    else:
        return none_value


def _summarize_number_among_keys(tags_dict, keys, summary_function=first, 
    keys_regex=False, length=False, distance=False, speed=False, 
    none_value=np.nan, **unused):
    """Returns a summary of parsed numbers found in all ``keys``.

    By default, returns the first recognized number, parsed as a float.

    A custom ``summary_function``, operating on a list of floats, can be passed 
    to provide a different summary of available values (e.g., ``sum`` or ``max``).

    Numbers are parsed from strings using ``_parse_osm_number``. If ``length``,
    ``distance``, or ``speed`` are set to ``True``, numbers with units will be
    converted to standard OSM units (m, km, and kph respectively). Otherwise,
    parsed numbers are returned raw, no matter their documented unit.

    Parameters
    ----------
    tags_dict : :obj:`dict`
        Dictionary of tags related to OSM element

    keys : :obj:`set`
        Keys in ``tags_dict`` to search for numbers.
        If any ``keys`` are not in ``tags_dict``, returns ``none_value``

    summary_function : function, optional, default=``first``
        Function with which to summarize multiple numbers.
        Must operate on a list.
        ``first`` is defined in the ``utils`` module. Equivalent to ``list[0]``.

    keys_regex : :obj:`bool`, optional, default = ``False``
        * ``True`` : keys will be identified using regex

    length : :obj:`bool`, optional, default=False
        * ``True`` : Value should be treated like a length;
        common length units (m, km, ft, in, mi) will be parsed and converted to meters
        * ``False`` : Units will be stripped but not parsed/converted
    distance : :obj:`bool`, optional, default=False
        * ``True`` : Value should be treated like a distance;
        common distance units (m, km, ft, in, mi) will be parsed and converted to kilometers
        * ``False`` : Units will be stripped but not parsed/converted
    speed : :obj:`bool`, optional, default=False
        * ``True`` : Value should be treated like a speed;
        common speed units (kph, mph) will be parsed and converted to kilometers/hr
        * ``False`` : Units will be stripped but not parsed/converted

    none_value : :obj:`float`, optional, default = ``np.nan``
        Value to return if none of ``keys`` or ``false_values`` are found in
        any of ``values``.

    Additional keyword arguments will be accepted but left unused.

    Returns
    -------
    :obj:`float`
        Summary of numbers identified in ``keys``
    """
    findings = []
    for key, value in tags_dict.items():
        if _key_in_set(key, keys, keys_regex=keys_regex):
            # If value is a raw number, return that
            if isinstance(value, (int, float)):
                findings.append(float(value))
            # If value is a string, try to parse
            elif isinstance(value, str):
                findings.append(_parse_osm_number(
                    value, length=length, distance=distance, speed=speed, none_value=none_value))
    if len(findings) > 0:
        if None in findings:
            print(findings)
        return summary_function(findings)
    else:
        return none_value


def _count_value_instances_among_keys(tags_dict, keys, values, full_match=True, 
    keys_regex=False, none_value=np.nan, **unused):
    """Returns count of instances where any of `values` occurs in any of `keys`.

    If `full_match` = `False`, will count substring matches (e.g., will count
    an instance of 'right' in 'continue|continue|right')

    """
    findings = []
    for key, value in tags_dict.items():
        if _key_in_set(key, keys, keys_regex=keys_regex):
            if isinstance(value, str):
                if full_match:
                    if any(value == x for x in values):
                        findings.append(1)
                else:
                    for v in values:
                        findings.append(value.count(v))
            else:
                if any(value == x for x in values):
                    findings.append(1)
    if sum(findings) > 0:
        return sum(findings)
    else:
        return none_value


def parse_osm_tags(overpass_json, variable_names, true_value=True, 
    false_value=False, none_value=np.nan):
    """Identify the presense or scalar measure of particular features based on OSM tags.

    Uses pre-specified key:value combinations to identify features.

    overpass_json : :obj:`dict`
        Dictionary representation of JSON from overpass API.
        Designed for output from ``retrieve_overpass_json`` or ``merge_overpass_jsons``.

    variable_names : :obj:`list` or :obj:`dict`
        If list, must contain any of the following variables as strings.
            e.g., ``['bike_lane', 'traffic_signal']``
        If dictionary, keys must be an of the following variables as strings. Values must be strings,
            which will be used in place of these variable names in the output dictionary.
            e.g., ``{'bike_lane': 'bicycle_lane', 'traffic_signal': 'stop_light'}``
        Way and node variables should be included together in ``variable_names`` (see examples, above),
            but variables will only be parsed for their respective element type.
        Available way variables (measure type):
            'bike_lane' (Boolean)
            'separated_bike_lane' (Boolean)
            'sharrow' (Boolean)
            'shoulder' (Boolean)
            'bike_route' (Boolean)
            'bike_blvd' (Boolean)
            'off_street_path' (Boolean)
            'bike_facility_width' (Scalar)
            'bike_facility_buffer_width' (Scalar)
            'parallel_parking' (Boolean)
            'perpendicular_parking' (Boolean)
            'oneway' (Boolean)
            'curb_to_curb_width' (Scalar)
            'lanes' (Count)
            'center_turn_lane' (Boolean)
            'speed_limit' (Scalar)
            'right_turn_lanes' (Count)
            'left_turn_lanes' (Count)
        Available node variables (measure type):
            'traffic_signal' (Boolean)
            'stop_sign' (Boolean)

    true_value : any type, optional, default = ``True``
        Value to return if any of ``keys`` are found in any of ``values``.

    false_value : any type, optional, default = ``False``
        Value to return if none of ``keys`` are found in any of ``values`` and
        any of ``false_values` are found in any of ``values``

    none_value : any type, optional, default = ``np.nan``
        Value to return if none of ``keys`` or ``false_values`` are found in
        any of ``values``.

    Returns
    -------
    :obj:`dict`
        ``overpass_json`` with keys added for each of ``variable_names``
    """
    
    # If variables names is list, make into dictionary
    if isinstance(variable_names, list):
        variable_names = dict(zip(variable_names, variable_names))

    # Set standard result values based on passed parameters
    bool_codes = {'true_value': true_value, 'false_value': false_value, 'none_value': none_value}

    # Parse each element
    for element in overpass_json['elements']:

        # Analyze ways
        if element['type'] == 'way':
            if 'tags' in element:
                
                # Set tags dictionary for element
                tags = element['tags']
                
                # Set standard keys for cycleways
                cycleway_keys = {'cycleway', 'cycleway:backward', 'cycleway:right', 'cycleway:left', 'cycleway:both'}
                
                # Bike lane in any direction (True or nan) 
                if 'bike_lane' in variable_names.keys():
                    keys = cycleway_keys
                    values = {'lane', 'opposite_lane'}
                    tags[variable_names['bike_lane']] = _identify_any_value_among_keys(
                        tags, keys, values, **bool_codes)

                # Separated bike lane in any direction (True or nan)
                if 'separated_bike_lane' in variable_names.keys():
                    keys = cycleway_keys
                    values = {'track', 'opposite_track', 'buffered_lane'}
                    tags[variable_names['separated_bike_lane']] = _identify_any_value_among_keys(
                        tags, keys, values, **bool_codes)
                
                # Sharrow in any direction (True or nan)
                if 'sharrow' in variable_names.keys():
                    keys = cycleway_keys
                    values = {'shared_lane', 'shared_busway'}
                    tags[variable_names['sharrow']] = _identify_any_value_among_keys(
                        tags, keys, values, **bool_codes)

                # Shoulder in any direction (True or nan)
                if 'shoulder' in variable_names.keys():
                    keys = cycleway_keys
                    values = {'shoulder'}
                    tags[variable_names['shoulder']] = _identify_any_value_among_keys(
                        tags, keys, values, **bool_codes)

                # Bike route in any direction (True or nan)
                if 'bike_route' in variable_names.keys():
                    keys = {'bicycle'}
                    values = {'designated'}
                    tags[variable_names['bike_route']] = _identify_any_value_among_keys(
                        tags, keys, values, **bool_codes)

                # Bike boulevard in any direction (True or nan)
                if 'bike_blvd' in variable_names.keys():
                    keys = {'highway'}
                    values = {'living_street'}
                    tags[variable_names['bike_blvd']] = _identify_any_value_among_keys(
                        tags, keys, values, **bool_codes)

                # Off Street Path (True or nan)
                if 'off_street_path' in variable_names.keys():
                    if ((_identify_tag_combination(tags, {'highway':'track', 'bicycle':'designated', 'motor_vehicle':'no'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'track', 'bike':'designated', 'motor_vehicle':'no'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'track', 'bicycle':'official', 'motor_vehicle':'no'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'track', 'bike':'official', 'motor_vehicle':'no'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'path', 'bicycle':'yes'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'path', 'bike':'yes'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'path', 'bicycle':'designated'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'path', 'bike':'designated'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'path', 'bicycle':'official'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'path', 'bike':'official'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'pedestrian', 'bicycle':'yes'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'pedestrian', 'bike':'yes'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'pedestrian', 'bicycle':'official'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'pedestrian', 'bike':'official'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'footway', 'bicycle':'yes'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'footway', 'bike':'yes'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'footway', 'bicycle':'official'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'footway', 'bike':'official'}, **bool_codes) == true_value) or
                        (_identify_tag_combination(tags, {'highway':'cycleway'}, **bool_codes) == true_value)):                       
                        tags[variable_names['off_street_path']] = true_value
                    else:
                        tags[variable_names['off_street_path']] = none_value

                # Minimum bike facility width (in feet)
                if 'bike_facility_width' in variable_names.keys():
                    keys = {'^cycleway.*width$'}
                    width = _summarize_number_among_keys(
                        tags, keys, summary_function=min, keys_regex=True, 
                        length=True, **bool_codes)
                    # Convert from meters to feet
                    if isinstance(width, (int, float)):
                        width = np.round(width * 3.28084)
                    tags[variable_names['bike_facility_width']] = width

                # Minimum bike facility buffer width (in feet)
                if 'bike_facility_buffer_width' in variable_names.keys():
                    keys = {'^cycleway:buffer(:\w+)?$'}
                    width = _summarize_number_among_keys(
                        tags, keys, summary_function=min, keys_regex=True, 
                        length=True, **bool_codes)
                    # Convert from meters to feet
                    if isinstance(width, (int, float)):
                        width = np.round(width * 3.28084)
                    tags[variable_names['bike_facility_buffer_width']] = width

                # Parallel parking on either side (True or nan)
                if 'parallel_parking' in variable_names.keys():
                    keys = {'parking:lane:right', 'parking:lane:left' , 'parking:lane:both'}
                    values = {'marked', 'parallel', 'inline'}
                    tags[variable_names['parallel_parking']] = _identify_any_value_among_keys(
                        tags, keys, values, **bool_codes)

                # Perpendicular or diagonal parking on either side (True or nan)
                if 'perpendicular_parking' in variable_names.keys():
                    keys = {'parking:lane:right', 'parking:lane:left' , 'parking:lane:both'}
                    values = {'perpendicular', 'orthogonal', 'diagonal'}
                    tags[variable_names['perpendicular_parking']] = _identify_any_value_among_keys(
                        tags, keys, values, **bool_codes)

                # Oneway (True, False, or nan)
                if 'oneway' in variable_names.keys():
                    keys = {'oneway'}
                    values = {'yes', -1}
                    false_values = {'no'}
                    tags[variable_names['oneway']] = _identify_any_value_among_keys(
                        tags, keys, values, false_values, **bool_codes)

                # Minimum curb-to-curb width (in feet)
                if 'curb_to_curb_width' in variable_names.keys():
                    keys = {'width','est_width'}
                    width = _summarize_number_among_keys(
                        tags, keys, summary_function=min, length=True, none_value=np.nan)#**bool_codes)
                    # Convert from meters to feet
                    if isinstance(width, (int, float)):
                        width = np.round(width * 3.28084)
                    tags[variable_names['curb_to_curb_width']] = width

                # Number of lanes (count)
                if 'lanes' in variable_names.keys():
                    keys = {'lanes'}
                    lanes = _summarize_number_among_keys(
                        tags, keys, summary_function=max, **bool_codes)
                    tags[variable_names['lanes']] = lanes

                # Center turn lane (True, or nan)
                if 'center_turn_lane' in variable_names.keys():
                    keys = {'turn:lanes:both_ways'}
                    values = {'left'}
                    tags[variable_names['center_turn_lane']] = _identify_any_value_among_keys(
                        tags, keys, values, false_values, **bool_codes)

                # Speed limit (mph)
                if 'speed_limit' in variable_names.keys():
                    keys = {'maxspeed', 'maxspeed:forward', 'maxspeed:backward'}
                    speed = _summarize_number_among_keys(
                        tags, keys, summary_function=max, speed=True, **bool_codes)
                    # Convert from kph to mph
                    if isinstance(speed, (int, float)):
                        speed = np.round(speed * 0.621371)
                    tags[variable_names['speed_limit']] = speed

                # Right turn lanes (count)
                if 'right_turn_lanes' in variable_names.keys():
                    keys = {'turn:lanes', 'turn:lanes:forward', 'turn:lanes:backward'}
                    values = {'right','slight_right'}
                    tags[variable_names['right_turn_lanes']] = _count_value_instances_among_keys(
                        tags, keys, values, full_match=False, **bool_codes)

                # Left turn lanes (count)
                if 'left_turn_lanes' in variable_names.keys():
                    keys = {'turn:lanes', 'turn:lanes:forward', 'turn:lanes:backward'}
                    values = {'left','slight_left'}
                    tags[variable_names['left_turn_lanes']] = _count_value_instances_among_keys(
                        tags, keys, values, full_match=False, **bool_codes)

        # Analyze nodes
        if element['type'] == 'node':
            if 'tags' in element:

                # Set tags dictionary for element
                tags = element['tags']

                # Traffic signal (True, False, or nan)
                if 'traffic_signal' in variable_names.keys():
                    keys = {'highway'}
                    values = {'traffic_signals', 'traffic_signals;crossing', 'crossing;traffic_signals'}
                    tags[variable_names['traffic_signal']] = _identify_any_value_among_keys(
                        tags, keys, values, **bool_codes)

                # Traffic signal (True, False, or nan)
                if 'stop_sign' in variable_names.keys():
                    keys = {'highway'}
                    values = {'stop', 'stop;crossing'}
                    tags[variable_names['stop_sign']] = _identify_any_value_among_keys(
                        tags, keys, values, **bool_codes)

    return overpass_json 


def merge_overpass_jsons(jsons):
    """Merge a list of overpass JSONs into a single JSON.

    Parameters
    ----------
    jsons : :obj:`list`
        List of dictionaries representing Overpass JSONs.

    Returns
    -------
    :obj:`dict`
        Dictionary containing all elements from input JSONS.
    """
    elements = []
    for osm_json in jsons:
        elements.extend(osm_json['elements'])
    return {'elements': elements}


def retrieve_overpass_json(wgs_polygon=None, path=None, network_type='all_private', 
    custom_filter=None):
    """Download and pickle Overpass JSON, or retrieve pickled version.

    Parameters
    ----------
    wgs_polygon : :class:`shapely.geometry.Polygon`, optional, default = ``None``
        Polygon defining area to be downloaded.
        Must be in WGS coordinates (lon/lat)
        If ``None``, ``path`` must be specified and JSON will be retreived from ``path``
    
    path : :obj:`str`, optional, default = ``None``
        Path at which to pickle downloaded JSON
        Include file name and extension.
        e.g., ``'data/overpass_download.json'``
        If ``None``, ``wgs_polygon`` must be specified and JSON will be downloaded but not pickled

    network_type :obj:`str`, optional, default = ``'all_private'``
        Type of network to download.
        Reference OSMnx documentation.

    custom_filter :obj:`str`, optional, default = ``None``
        Custom tag filter for Overpass query.
        Reference OSMnx documentation.

    Returns
    -------
    :obj:`dict`
        Dictionary representing Overpass JSON.
    """
    if path:
        # Check to see if file already exists
        json_path = Path(path)
        if json_path.is_file():
            with json_path.open() as f:
                osm_json = json.load(f)
            return osm_json
    if wgs_polygon:
        # Define function to save JSON
        def save_json(json_, path):
            with open(path, 'w') as outfile:
                json.dump(json_, outfile)
        # Download OSM jsons
        osm_jsons = ox.osm_net_download(
            wgs_polygon, network_type=network_type, custom_filter=custom_filter)
        # Combine the list of jsons
        osm_json = merge_overpass_jsons(osm_jsons)
        # Save to file
        if path:
            save_json(osm_json, path)
        return osm_json
    else:
        print('Must supply at least polygon or file name.')


def examine_tags(overpass_json, specific_tags=None):
    """Examine tags within an Overpass JSON.

    Outputs seperate dictionaries summarizing tags in way and node elements.
    Counts unique key:value pairs.

    Parameters
    ----------
    overpass_json : :obj:`dict`
        Dictionary representing Overpass JSON.

    specific_tags : :obj:`list`
        List of tags to examine. All other tags will be ignored.

    Returns
    -------
        * node_tags : :obj:`dict`
        * way_tags : :obj:`dict`

    """
    def _count_tags(element, tags_dict, specific_tags):
        if 'tags' in element:
            for key, value in element['tags'].items():
                if specific_tags:
                    if not any(re.compile(tag).match(key) for tag in specific_tags):
                        continue
                if key not in tags_dict:
                    tags_dict[key] = {}
                if value not in tags_dict[key]:
                    tags_dict[key][value] = 1
                else:
                    tags_dict[key][value] += 1
    node_tags = {}
    way_tags = {}
    for element in overpass_json['elements']:
        if element['type'] == 'node':
            _count_tags(element, node_tags, specific_tags)
        if element['type'] == 'way':
            _count_tags(element, way_tags, specific_tags)
    return node_tags, way_tags