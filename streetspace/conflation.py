##############################################################################
# Module: conflation.py
# Description: Functions to conflate linestring geometries.
# License: MIT
##############################################################################

from .geometry import *
import usaddress
from fuzzywuzzy import fuzz
import shapely as sh
import itertools

def match_lines_by_midpoint(target_features, match_features, distance_tolerance, 
    match_features_sindex=None, azimuth_tolerance=None, length_tolerance=None,
    incidence_tolerance=None, match_by_score=False, match_fields=False, match_stats=False, 
    constrain_target_features=False, target_features_sindex=None,
    match_vectors=False, verbose=False):
    """Conflate attributes between line features based on midpoint proximity.

    Finds the midpoint of all target features and matches them to the closest
    match feature meeting the specified tolerances.

    Parameters
    ----------
    target_features : :class:`geopandas.GeoDataFrame`
        Features to which ``match_features`` will be matched.
        Must have LineString geometries.
        All ``target_features`` will be included in output, with or without a match.

    match_features : :class:`geopandas.GeoDataFrame`
        Features to be matched to ``target_features``.
        Must have LineString geometries.
        Only successfully matched features will be included in output.
        Multiple ``match_features`` may be matched to a single target feature.
        Must have the same spatial reference as ``target_features``.

    distance_tolerance : :obj:`float`
        Maximum distance from each target feature midpoint to the closest point
        along candidate ``match_features``.
        In spatial unit of ``target_features``.

    match_features_sindex : :class:`rtree.index.Index`, optional, default = ``None``
        Spatial index for ``match_features``.
        If provided, will not have to be constructed for each function call.

    azimuth_tolerance : :obj:`float`, optional, default = ``None``
        Maximum azimuth difference (in degrees) between target feature and potential match features.
        Value of 0 specifies that target and match features must be perfectly aligned.
        Value of 90 specifies that target and match features may be perpendicular.
        Azimuth difference will never exceed 90 degrees.
        If ``None``, azimuth difference will not be used as a criteria for matching.

    length_tolerance: :obj:`float`, optional, default = ``None``
        Maximum length difference between target feature and potential match features.
        Value of 0 specifies that target and match features must be exactly the same length.
        Large value specifies that target and match may be substantially different lengths.
        Length difference is calculated as an absolute value
            (e.g., target or match feature may be longer or shorter than the other)
        If ``None``, length difference will not be used as a criteria for matching.

    incidence_tolerance: :obj:`float`, optional, default = ``None``
        Maximum angle of incidence between target feature midpoint and closest point of potential match features.
        Measured in degrees difference from perpendiular to the target feature at its midpoint.
        Value of 0 specifies that the angle of incidence must be exactly 90 degrees
            (e.g., match feature overlaps target midpoint)
        Value of 90 specifies that match features may be non-overlapping and in-line with target feature.
        If ``None``, angle of incidence will not be used as a criteria for matching.

    match_by_score: :obj:`bool`, optional, default = ``False``
        * ``True``: All specified tolerances will be used to compute a score for each candidate;
            the final match will be selected based on the lowest score. The score will equally weight
            all specified tolerances.
        * ``False``: Specified tolerances will be used to restrict match candidates,
            but final match will be identified solely based on midpoint distance.
        
    match_fields: :obj:`bool`, optional, default = ``False``
        * ``True``: Fields from match features will be included in output.
        * ``False``: Only row indices for match features will be included in output.

    match_stats: :obj:`bool`, optional, default = ``False``
        * ``True``: Statistics related to tolerances will be included in output.
        * ``False``: No match statistics will be included in ouput.

    constrain_target_features: :obj:`bool`, optional, default = ``False``
        * ``True``: Extents of ``match_features``, plus a ``distance_tolerance`` buffer
             will be used to select relevent ``target_features`` prior to matching. 
             When the extent or number of ``match_features`` is small relative to
            ``target_features``, this dramatically improves performance because fewer
            ``target_features`` are analyzed for potential matches.
        * ``False``: All ``target_features`` are analyzed for potential matches.

    target_features_sindex: :class:`rtree.index.Index`, optional, default = ``None``
        If ``constrain_target_features=True``, a spatial index for the ``target_features``
        will be computed unless one is provided. If the same ``target_features`` are specified 
        in multiple function calls, pre-computing a spatial index will improve performance.
        If ``constrain_target_features=False``, ``target_features_sindex`` is unnecessary.

    match_vectors: :obj:`bool`, optional, default = ``False``
        * ``True``: Constructs LineStrings between midpoint of ``target_features`` and the
        closest points along matched ``match_features``. Useful for vizualizing match results.

    verbose: :obj:`bool`, optional, default = ``False``
        * ``True``: Reports status by printing to standard output

    Returns
    -------
    :class:`geopandas.GeoDataFrame`
        Same row indices as ``target_features``
    """
    # Copy input features to the function doesn't modify the originals
    target_features = target_features.copy()
    match_features = match_features.copy()

    if verbose:
        start = time()
        length = len(target_features)
        counter = 0
    
    # Constrain target features to those near available match features
    if constrain_target_features:
        if not target_features_sindex:
            target_features_sindex = target_features.sindex
        nearby_target_idx = []
        
        for match_feature in match_features.geometry:
            nearby_target_idx.extend(
                list(target_features_sindex.intersection(
                    match_feature.buffer(distance_tolerance).bounds)))
        nearby_target_idx = list(set(nearby_target_idx))
        operating_target_features = target_features[['geometry']].iloc[nearby_target_idx].copy()
    else:
        operating_target_features = target_features[['geometry']].copy()

    # Make a spatial index for match features, if one isn't supplied
    if not match_features_sindex:
        match_features_sindex = match_features.sindex 
    
    # Initiate lists to store match results
    match_indices = []
    match_dists = []
    match_lengths = []
    match_azimuths = []
    match_incidences = []
    match_scores = []
    if match_vectors:
        match_vectors = []
       
    # Iterate through target features:
    for target in operating_target_features.geometry:

        # Roughly filter candidates with a spatial index
        target_midpoint = midpoint(target)
        match_area = target_midpoint.buffer(distance_tolerance)
        candidate_IDs = list(match_features_sindex.intersection(match_area.bounds))
        candidates = match_features.geometry.iloc[candidate_IDs].reset_index()
        
        # Calculate distances to closest points along candidates
        match_point_lin_refs = [line.project(target_midpoint) for 
            line in candidates['geometry']]
        match_points = [line.interpolate(ref) for 
            line, ref in zip(candidates['geometry'], match_point_lin_refs)]
        closest_dists = [target_midpoint.distance(point) for 
            point in match_points]
        candidates.loc[:,'match_point'] = pd.Series(match_points)
        candidates.loc[:,'match_dist'] = pd.Series(closest_dists)
        candidates.loc[:,'match_point_lin_ref'] = pd.Series(match_point_lin_refs)
        
        # Filter by distance, unless match stats are being collected
        if not match_stats:
            candidates = candidates[
                candidates['match_dist'] <= distance_tolerance
                ].reset_index() # Save original index in column
        
        # Get lengths of each match feature
        if (length_tolerance is not None) or match_stats:
            _match_lengths = [line.length for line in candidates['geometry']]
            # Compare to length of target feature
            _match_lengths = [x - target.length for x in _match_lengths]
            # Add relative azimuths to the candidates dataframe
            candidates['match_length'] = pd.Series(_match_lengths)
            
            # Filter by length if desired
            if (length_tolerance is not None) and (not match_stats):
                # Filter out match features beyond length tolerance
                candidates = candidates[
                    candidates['match_length'].abs() < 
                    length_tolerance
                    ].reset_index(drop=True)
        
        # Get the azimuth of each match feature at its closest point
        if (azimuth_tolerance is not None) or match_stats:
            _match_azimuths = [azimuth_at_distance(line, ref) 
                for line, ref in zip(
                    candidates['geometry'], candidates['match_point_lin_ref'])]
            # Compare it to the azimuth of the target feature at its midpoint
            target_azimuth = azimuth_at_distance(
                target, target.project(target_midpoint))
            _match_azimuths = [azimuth_difference(
                target_azimuth, match_azimuth, directional=False) 
                for match_azimuth in _match_azimuths]
            # Add relative azimuths to the candidates dataframe
            candidates['match_azimuth'] = pd.Series(_match_azimuths)
            
            # Filter by azimuth, unless match stats are being collected
            if (azimuth_tolerance is not None) and (not match_stats):
                # Filter out match features beyond azimuth tolerance
                candidates = candidates[
                    candidates['match_azimuth'] < 
                    azimuth_tolerance
                    ].reset_index(drop=True)

        # Get angle of incidence between target feature centerpoint and
        # closest points on match features
        if (incidence_tolerance is not None) or match_stats:
            incidence_lines = [LineString([target_midpoint, x]) for x in match_points]
            incidence_azimuths = [azimuth(x) for x in incidence_lines]
            _match_incidences = [
                # Subtract angle of incidence from 90 degrees (the optimal angle)
                90 - azimuth_difference(x, target_azimuth, directional=False)
                for x in incidence_azimuths]
            # Add relative angles of incidence to candidates dataframe
            candidates['match_incidence'] = pd.Series(_match_incidences)
            
            # Filter by angle of incidence, unless match stats are being collected
            if (incidence_tolerance is not None) and (not match_stats):
                candidates = candidates[
                    candidates['match_incidence'] < 
                    incidence_tolerance
                    ].reset_index(drop=True)

        # Identify match feature and attributes
        match_id = np.nan
        match_dist = np.nan
        match_length = np.nan
        match_azimuth = np.nan
        match_incidence = np.nan
        match_score = np.nan
        match_vector = np.nan
        
        if len(candidates) > 0:
            
            # Identify available criteria
            available_criteria = []
            if distance_tolerance:
                available_criteria.append(('match_dist', distance_tolerance))
            if length_tolerance:
                available_criteria.append(('match_length', length_tolerance))
            if azimuth_tolerance:
                available_criteria.append(('match_azimuth', azimuth_tolerance))
            if incidence_tolerance:
                available_criteria.append(('match_incidence', incidence_tolerance))

            # Calculate scores based on available criteria
            if match_by_score:
                scores = pd.Series([0] * len(candidates))
                for value_column, tolerance in available_criteria:
                    # Get absolute values
                    values = candidates[value_column].abs()
                    # Cap values at tolerance
                    values = values.clip(upper=tolerance)
                    # Calculate scores as a proportion of tolerance
                    # and weighted by number of available criteria
                    scores = scores + ((tolerance - values) / tolerance / len(available_criteria))
                candidates['match_score'] = pd.Series(scores)
            # Or calculate scores solely as a function of distance
            else:
                candidates['match_score'] = (distance_tolerance - candidates['match_dist']) / distance_tolerance
            
            # Idenify the candidate with the highest score
            # (Info for this feature will be returned even if no match is made based on tolerances)
            highest_score_idx = candidates['match_score'].idxmax()

            # Identify the candidate with the highest score while also meeting specified tolerances
            tolerance_candidates = candidates
            for value_column, tolerance in available_criteria:
                tolerance_candidates = tolerance_candidates.mask(
                    tolerance_candidates[value_column] > tolerance).copy()
            match_idx = tolerance_candidates['match_score'].idxmax()

            # Assign either match index or highest score to return
            if pd.notnull(match_idx):
                return_idx = match_idx
                match_id = candidates.at[return_idx, 'index']
            else:
                return_idx = highest_score_idx             

            # Get match stats if a candidate is matched
            if pd.notnull(return_idx):
                match_dist = candidates.at[return_idx, 'match_dist']
                if length_tolerance or match_stats:
                    match_length = candidates.at[return_idx, 'match_length']
                if azimuth_tolerance or match_stats:
                    match_azimuth = candidates.at[return_idx, 'match_azimuth']
                if incidence_tolerance or match_stats:
                    match_incidence = candidates.at[return_idx, 'match_incidence']
                if match_by_score:
                    match_score = scores.at[return_idx]
                # Construct match vector
                if isinstance(match_vectors, list):
                    match_vector = LineString([target_midpoint, candidates.at[return_idx, 'match_point']])
        
        # Record match stats
        match_indices.append(match_id)
        match_dists.append(match_dist)
        if length_tolerance or match_stats:
            match_lengths.append(match_length)
        if azimuth_tolerance or match_stats:
            match_azimuths.append(match_azimuth)
        if incidence_tolerance or match_stats:
            match_incidences.append(match_incidence)
        if match_by_score:
            match_scores.append(match_score)
        if isinstance(match_vectors, list):
            match_vectors.append(match_vector)
        
        # Report status
        if verbose:
            if counter % round(length / 10) == 0 and counter > 0:
                percent_complete = (counter // round(length / 10)) * 10
                minutes = (time()-start) / 60
                print('{}% ({} segments) complete after {:04.2f} minutes'.format(percent_complete, counter, minutes))
            counter += 1
    
    # Merge joined data with target features
    operating_target_features.loc[:,'match_id'] = pd.Series(
        match_indices, index=operating_target_features.index)
    if match_stats:
        operating_target_features.loc[:,'match_dist'] = pd.Series(
            match_dists, index=operating_target_features.index)
        operating_target_features.loc[:,'match_length'] = pd.Series(
                match_lengths, index=operating_target_features.index)
        operating_target_features.loc[:,'match_azimuth'] = pd.Series(
                match_azimuths, index=operating_target_features.index)
        operating_target_features.loc[:,'match_incidence'] = pd.Series(
                match_incidences, index=operating_target_features.index)
    if match_by_score:
        operating_target_features.loc[:,'match_score'] = pd.Series(
                match_scores, index=operating_target_features.index)
    if isinstance(match_vectors, list):
        operating_target_features.loc[:,'match_vectors'] = pd.Series(
            match_vectors, index=operating_target_features.index)
    
    # Join operating target features back onto all target features
    target_features = target_features.merge(
        operating_target_features.drop(columns=['geometry']), 
        how='left', left_index=True, right_index=True)

    # Join fields from match features
    if match_fields:
        target_features = target_features.merge(
            match_features.drop(columns=['geometry']),
            how='left', left_on='match_id', right_index=True, suffixes=('', '_match'))

    # Report done
    if verbose:
        print('100% ({} segments) complete after {:04.2f} minutes'.format(counter, (time()-start) / 60))

    return target_features


def find_parallel_segment(a, b, distance_tolerance):
    """Identify a segment of line b that runs parallel to line a.

    Parameters
    ----------
    a : :class:`shapely.geometry.LineString`
        LineString along which to find a parallel segment from ``b``.

    b : :class:`shapely.geometry.LineString`
        LineString from which to find a segment this is parallel to ``a``.

    distance_tolerance : :obj:`float`
        Maximum distance that a potential segment of ``b`` may be from ``a``.

    Returns
    -------
    :class:`shapely.geometry.LineString`
        Segment of ``b`` running parallel to ``a``.
    """
    def endpoint_near_endpoint(a_endpoint, b_endpoints):
        for b_endpoint in b_endpoints:
            if a_endpoint.distance(b_endpoint) <= distance_tolerance:
                return True
            
    def almost_equals_any_endpoint(closest_point, b_endpoints):
        for b_endpoint in b_endpoints:
            if closest_point.almost_equals(b_endpoint):
                return True
    
    # Get endpoints of "a" and "b"
    a_endpoints = endpoints(a)
    b_endpoints = endpoints(b)
    # Identify points along "b" that are closest to endpoints of "a"
    split_points = []
    for a_endpoint in a_endpoints:  
        if not endpoint_near_endpoint(a_endpoint, b_endpoints):
            closest_point = closest_point_along_line(a_endpoint, b)
            # Check whether closest point is within distance tolerance of "a" endpoint
            if closest_point.distance(a_endpoint) < distance_tolerance:
                # Check whether closest point on "b" is the same as an endpoint of "b"
                if not almost_equals_any_endpoint(closest_point, b_endpoints):
                    # If not, save as split point
                    split_points.append(closest_point)
    # Assume there is no segment of "b" parallel to "a"
    adjacent_seg = None
    # Split "b" into segments at these points
    if len(split_points) > 0:
        b_segments = split_line_at_points(b, split_points)
        # Identify which (if any) segment runs parallel to "a" based on hausdorff dist
        along = [segment for segment in b_segments 
                 if directed_hausdorff(segment, a) < distance_tolerance]
        if len(along) > 0:
            adjacent_seg = along[0]
    return adjacent_seg

def segment_linear_reference(line, segment):
    """Calculate linear references of segment endpoints relative to a parent LineString

    """
    return tuple([line.project(x) for x in endpoints(segment)])


def match_lines_by_hausdorff(target_features, match_features, distance_tolerance, 
    azimuth_tolerance=None, match_features_sindex=None, match_fields=False, match_stats=False, 
    field_suffixes=('', '_match'), match_strings=None, constrain_target_features=False, 
    target_features_sindex=None, match_vectors=False, expand_target_features=False, 
    closest_match=False, closest_target=False, verbose=False):
    """Conflate attributes between line features based on Hausdorff distance.
    
    target_features : :class:`geopandas.GeoDataFrame`
        Features to which ``match_features`` will be matched.
        Must have LineString geometries.
        All ``target_features`` will be included in output, with or without a match.

    match_features : :class:`geopandas.GeoDataFrame`
        Features to be matched to ``target_features``.
        Must have LineString geometries.
        Only successfully matched features will be included in output.
        Multiple ``match_features`` may be matched to a single target feature.
        Must have the same spatial reference as ``target_features``.

    distance_tolerance : :obj:`float`
        Maximum Hausdorff distance between each target feature and candidate ``match_features``
        Because directed Hausdorff distances are calculated from target to match
            and match to target, ``distance_tolerance`` will be assessed based on
            the smaller of these two values.
        If feature segments are matched (e.g., 1:n, m:1, or m:n),
            Hausdorff distances are calculated for each segment.
        In spatial unit of ``target_features``.

    azimuth_tolerance : :obj:`float`, optional, default = ``None``
        Maximum azimuth difference (in degrees) between target feature and potential match features.
        Feature azimuths are calculated as the azimuth of the feature's "major axis"
            (the longest axis of the feature's minimum bounding rectangle).
        If feature segments are matched (e.g., 1:n, m:1, or m:n),
            azimuths are calculated for each segment.   

    match_features_sindex : :class:`rtree.index.Index`, optional, default = ``None``
        Spatial index for ``match_features``.
        If provided, will not have to be constructed for each function call.

    match_fields : :obj:`bool`, optional, default = ``False``
        * ``True``: Fields from match features will be included in output.
        * ``False``: Only row indices for match features will be included in output.

    match_stats : :obj:`bool`, optional, default = ``False``
        * ``True``: Statistics related to tolerances will be included in output.
        * ``False``: No match statistics will be included in ouput.

    field_suffixes : :obj:`tuple`, optional, default = ``('', '_match')``
        Suffixes to be appended to output field names for ``target_features`` 
            and ``match_features``, respectively.
        Only used if  ``match_stats=True``.

    match_strings : :obj:`tuple`, optional, default = ``None``
        Fields used to compute fuzzy string comparisions.
        Typically, these are street name fields for the ``target_features`` 
            and ``match_features``, respectively.
        String comparisions do not affect matches, but can be post-processed to
            help assess match quality.

    constrain_target_features : :obj:`bool`, optional, default = ``False``
        * ``True``: Extents of ``match_features``, plus a ``distance_tolerance`` buffer,
            will be used to select relevent ``target_features`` prior to matching. 
            When the extent or number of ``match_features`` is small relative to
            ``target_features``, this dramatically improves performance because fewer
            ``target_features`` are analyzed for potential matches.
        * ``False``: All ``target_features`` are analyzed for potential matches.

    target_features_sindex : :class:`rtree.index.Index`, optional, default = ``None``
        If ``constrain_target_features=True``, a spatial index for the ``target_features``
        will be computed unless one is provided. If the same ``target_features`` are specified 
        in multiple function calls, pre-computing a spatial index will improve performance.
        If ``constrain_target_features=False``, ``target_features_sindex`` is unnecessary.

    match_vectors : :obj:`bool`, optional, default = ``False``
        * ``True``: Constructs LineStrings between midpoint of ``target_features`` and the
        closest points along matched ``match_features``. Useful for vizualizing match results.

    expand_target_features : :obj:`bool`, optional, default = ``False``
        * ``True`` : Target features that match to multiple ``match_features`` will be expanded
        into multiple segments, each corresponding to a single match feature. Each target feature
        segment will be output as a seperate record with an index field identifying original
        row-wise indices from ``target_features``.

    closest_match : :obj:`bool`, optional, default = ``False``
        * ``True`` : Only the closest available match feature will be matched to each target
            feature, based on Hausdorff distance
        * ``False`` : All available match features will match to each target feature

    closest_target : :obj:`bool`, optional, default = ``False``
        * ``True`` : A target feature will only match with a match feature if it is the closest
            available target, based on Hausdorff distance
        * ``False`` : A target feature will match with all available match features, regardless
            of whether it has also matched with other target features

    verbose : :obj:`bool`, optional, default = ``False``
        * ``True`` : Reports status by printing to standard output
    """
    # Copy input features to the function doesn't modify the originals
    target_features = target_features.copy()
    match_features = match_features.copy()
    original_target_feature_columns = target_features.columns
    original_crs = target_features.crs

    if verbose:
        start = time()
        length = len(target_features)
        counter = 0
    
    # Constrain target features to those near available match features
    if constrain_target_features:
        if not target_features_sindex:
            target_features_sindex = target_features.sindex
        nearby_target_idx = []
        
        for match_feature in match_features.geometry:
            nearby_target_idx.extend(
                list(target_features_sindex.intersection(
                    match_feature.buffer(distance_tolerance).bounds)))
        nearby_target_idx = list(set(nearby_target_idx))
        operating_target_features = target_features[['geometry']].iloc[nearby_target_idx].copy()
    else:
        operating_target_features = target_features[['geometry']].copy()

    # Make a spatial index for match features, if one isn't supplied
    if not match_features_sindex:
        match_features_sindex = match_features.sindex 
    
    # Initiate lists to store match results
    match_indices = []
    h_tms_matches = []
    t_props_matches = []
    t_segs_matches = []
    t_linrefs_matches = []
    h_mts_matches = []
    m_props_matches = []
    m_segs_matches = []
    m_linrefs_matches = []
    if match_vectors:
        match_vectors = []
      
    # Iterate through target features:
    for i, target in enumerate(operating_target_features.geometry):

        # Initiate lists to store matches
        match_ids = []
        h_tms = []
        t_props = []
        t_segs = []
        t_linrefs = []
        h_mts = []
        m_props = []
        m_segs = []
        m_linrefs = []

        # Only analyze targets with length
        if target.length > 0:

            # Roughly filter candidates with a spatial index
            search_area = target.buffer(distance_tolerance).bounds
            candidate_IDs = list(match_features_sindex.intersection(search_area))
            candidates = match_features[['geometry']].iloc[candidate_IDs].reset_index()
          
            # Calculate Hausdorff distances from feature to each candidate (h_fc)
            h_tm_list = [directed_hausdorff(target, candidate) for candidate in candidates.geometry]
            candidates['h_tm'] = pd.Series(h_tm_list)

            # Calculate Hausdorff distances from each candidate to feature (h_cf)
            h_mt_list = [directed_hausdorff(candidate, target) for candidate in candidates.geometry]
            candidates['h_mt'] = pd.Series(h_mt_list)

            # Define function to compare major axis azimuths
            def azimuth_match(target, candidate, azimuth_tolerance):
                if azimuth_tolerance:
                    target_azimuth = major_axis_azimuth(target)
                    candidate_azimuth = major_axis_azimuth(candidate)
                    azimuth_difference_ = azimuth_difference(target_azimuth, candidate_azimuth, directional=False)
                    if azimuth_difference_ <= azimuth_tolerance:
                        return True
                    else:
                        return False
                else:
                    return True

            # Examine each candidate's relationship to the target feature
            for candidate in candidates.itertuples():

                # Only analyze candidates with length
                if candidate.geometry.length > 0:
                
                    # Initialize default match values
                    h_tm = None
                    t_prop = None
                    t_seg = None
                    t_linref = None
                    h_mt = None
                    m_prop = None
                    m_seg = None
                    m_linref = None
                    
                    
                    # 1:1
                    if (
                        (candidate.h_tm <= distance_tolerance) and 
                        (candidate.h_mt <= distance_tolerance) and
                        # Check that azimuth is acceptable
                        azimuth_match(target, candidate.geometry, azimuth_tolerance)):
                        # Whole target matches candidate
                        h_tm = candidate.h_tm
                        t_prop = 1
                        t_seg = target
                        t_linref = (0, target.length)
                        # Whole candidate matches target
                        h_mt = candidate.h_mt
                        m_prop = 1
                        m_seg = candidate.geometry
                        m_linref = (0, candidate.geometry.length)
                        
                    # m:1
                    elif (
                        (candidate.h_tm <= distance_tolerance) and 
                        (candidate.h_mt > distance_tolerance)):
                        # Find the candidate segment matching the target
                        candidate_seg = find_parallel_segment(
                            target, candidate.geometry, distance_tolerance)

                        if (candidate_seg and 
                            candidate_seg.length > 0 and
                            azimuth_match(target, candidate_seg, azimuth_tolerance)):
                            # Whole target matches candidate
                            h_tm = directed_hausdorff(target, candidate_seg)
                            t_prop = 1
                            t_seg = target
                            t_linref = (0, target.length)
                            # Calculate proportion of candidate included in segment
                            h_mt = directed_hausdorff(candidate_seg, target)
                            m_prop = candidate_seg.length / candidate.geometry.length
                            m_seg = candidate_seg
                            m_linref = segment_linear_reference(candidate.geometry, candidate_seg)

                    # 1:n
                    elif (
                        (candidate.h_tm > distance_tolerance) and 
                        (candidate.h_mt <= distance_tolerance)):
                        # Find the target segment matching the candidate
                        target_seg = find_parallel_segment(
                            candidate.geometry, target, distance_tolerance)
                        if (target_seg and 
                            target_seg.length > 0 and
                            azimuth_match(target_seg, candidate.geometry, azimuth_tolerance)):
                            # Calculate proportion of target included in segment
                            h_tm = directed_hausdorff(target_seg, candidate.geometry)
                            t_prop = target_seg.length / target.length
                            t_seg = target_seg
                            t_linref = segment_linear_reference(target, target_seg)
                            # Whole candidate matches target
                            h_mt = directed_hausdorff(candidate.geometry, target_seg)
                            m_prop = 1
                            m_seg = candidate.geometry
                            m_linref = (0, candidate.geometry.length)
                    
                    # potential m:n
                    elif (
                        (candidate.h_tm > distance_tolerance) and 
                        (candidate.h_mt > distance_tolerance)):
                        # See if parallel segments can be identified
                        target_seg = find_parallel_segment(
                            candidate.geometry, target, distance_tolerance)
                        candidate_seg = find_parallel_segment(
                            target, candidate.geometry, distance_tolerance)
                        # Measure hausdorff distance (non-directed) between parallel segments
                        if target_seg and candidate_seg:
                            h_tm_seg = directed_hausdorff(target_seg, candidate_seg)
                            h_mt_seg = directed_hausdorff(candidate_seg, target_seg)
                            if ((h_tm_seg <= distance_tolerance) and
                                (h_mt_seg <= distance_tolerance) and
                                target_seg.length > 0 and
                                candidate_seg.length > 0 and
                                azimuth_match(target_seg, candidate_seg, azimuth_tolerance)):
                                h_tm = h_tm_seg
                                t_prop = target_seg.length / target.length
                                t_seg = target_seg
                                t_linref = segment_linear_reference(target, target_seg)
                                h_mt = h_mt_seg
                                m_prop = candidate_seg.length / candidate.geometry.length
                                m_seg = candidate_seg
                                m_linref = segment_linear_reference(candidate.geometry, candidate_seg)
                                             
                    if t_prop is not None:
                        match_ids.append(candidate.index)
                        h_tms.append(h_tm)
                        t_props.append(t_prop)
                        t_segs.append(t_seg)
                        t_linrefs.append(t_linref)
                        h_mts.append(h_mt)
                        m_props.append(m_prop)
                        m_segs.append(m_seg)
                        m_linrefs.append(m_linref) 

        # Record match stats
        match_indices.append(match_ids)
        h_tms_matches.append(h_tms)
        t_props_matches.append(t_props)
        t_segs_matches.append(t_segs)
        t_linrefs_matches.append(t_linrefs)
        h_mts_matches.append(h_mts)
        m_props_matches.append(m_props)
        m_segs_matches.append(m_segs)
        m_linrefs_matches.append(m_linrefs)

        # Construct match vector
        if isinstance(match_vectors, list):
            vectors = []
            for t_seg, m_seg in zip(t_segs_matches, m_segs_matches):
                if t_seg and m_seg:
                    vectors.append(LineString([midpoint(t_seg), midpoint(m_seg)]))
            match_vectors.append(vectors)

        # Report status
        if verbose:
            if counter % round(length / 10) == 0 and counter > 0:
                percent_complete = (counter // round(length / 10)) * 10
                minutes = (time()-start) / 60
                print('{}% ({} segments) complete after {:04.2f} minutes'.format(percent_complete, counter, minutes))
            counter += 1
    
    # Merge joined data with target features
    operating_target_features['match_index'] = pd.Series(
        match_indices, index=operating_target_features.index)
    if match_stats or closest_match or closest_target or expand_target_features:
        operating_target_features['h_tm'] = pd.Series(
            h_tms_matches, index=operating_target_features.index)
        operating_target_features['t_prop'] = pd.Series(
            t_props_matches, index=operating_target_features.index)
        operating_target_features['t_seg'] = pd.Series(
            t_segs_matches, index=operating_target_features.index)
        operating_target_features['t_linref'] = pd.Series(
            t_linrefs_matches, index=operating_target_features.index)
        operating_target_features['h_mt'] = pd.Series(
            h_mts_matches, index=operating_target_features.index)
        operating_target_features['m_prop'] = pd.Series(
            m_props_matches, index=operating_target_features.index)
        operating_target_features['m_seg'] = pd.Series(
            m_segs_matches, index=operating_target_features.index)
        operating_target_features['m_linref'] = pd.Series(
            m_linrefs_matches, index=operating_target_features.index)
    if isinstance(match_vectors, list):
        operating_target_features['match_vectors'] = pd.Series(
            match_vectors, index=operating_target_features.index)

    # Store original target feature IDs
    operating_target_features = operating_target_features.reset_index().rename(columns={'index': 'target_index'})

    # Expand targets matched with more than one candidate
    # This is required if 'closest_match' or 'closest_target' are specified
    if expand_target_features or closest_match or closest_target:

        # Look for lists of match IDs in each row
        expanded_targets = []
        for i, target in enumerate(operating_target_features.itertuples()):
            if isinstance(target.match_index, list):
                # Make duplicate rows for each match ID with respective attributes
                for j, match in enumerate(target.match_index):
                    new_row = target._asdict()
                    new_row.pop('Index', None)
                    for key, value in target._asdict().items():
                        if isinstance(value, list):
                            new_row[key] = value[j]
                    # Append new row to end of dataframe
                    operating_target_features = operating_target_features.append(new_row, ignore_index=True)
                # Mark original row for deletion
                expanded_targets.append(i)
        # Delete expanded targets
        operating_target_features = operating_target_features.drop(expanded_targets)
        # Replace target geometries with target segments (if not NaN)
        def replace_target_segs(row):
            if pd.notnull(row.t_seg):
                return row.t_seg
            else:
                return row.geometry
        operating_target_features['geometry'] = operating_target_features.apply(replace_target_segs, axis=1)

    # For each unique target geometry, delete all matches except the closest one
    # (expanded targets are deleted if they don't have the closest match)
    # Required if 'closest_target'
    if closest_match or closest_target:

        # Identify sets of records with identical targets
        equivalent_target_sets = [d for _, d in operating_target_features.groupby(
            ['target_index','t_linref']) if len(d) > 1]

        # Identify which of these records has the closest match
        equivalent_record_ids = []
        closest_records = gpd.GeoDataFrame(crs=operating_target_features.crs)
        for equivalent_target_set in equivalent_target_sets:
            # Keep track of IDs for equivalent records
            equivalent_record_ids.extend(equivalent_target_set.index.tolist())
            # Identify minimum tc and ct distances and associated indices
            h_tm_min_idx = equivalent_target_set['h_tm'].astype(float).idxmin()
            h_tm_min = equivalent_target_set['h_tm'].astype(float).min()
            h_mt_min_idx = equivalent_target_set['h_mt'].astype(float).idxmin()
            h_mt_min = equivalent_target_set['h_mt'].astype(float).min()
            # Identify overall closest match
            min_idx = h_tm_min_idx if h_tm_min < h_mt_min else h_mt_min_idx
            closest_records = closest_records.append(
                operating_target_features.loc[[min_idx]], ignore_index=True)
        # Drop equivalent records
        operating_target_features = operating_target_features.drop(
            equivalent_record_ids)
        # Add back those with the closest match
        operating_target_features = operating_target_features.append(
            closest_records, ignore_index=True)

    
    # Ensure that each match feature is only matched to one, closest target feature
    # (No targets are deleted, but matches are removed if a given target isn't closest)
    if closest_target:
        # Identify sets of records with the same match id
        match_id_sets = [d for _, d in operating_target_features.groupby(
            'match_index') if len(d) > 1]

        # Within these sets, identify sets with overlapping linear references
        for match_id_set in match_id_sets:
            
            # Get ID for match feature
            match_id = match_id_set.iloc[0]['match_index']

            # Get raw geometry for match feature
            match_geom = match_features.iloc[match_id]['geometry']
            
            # Find overlapping linear reference ranges among the original matches
            lin_ref_ranges = merge_intervals(match_id_set['m_linref'].tolist())
            
            # Identify sets of records within each range
            lin_ref_sets = [match_id_set[match_id_set['m_linref'].apply(
                                lambda x: True if (x[0] >= lower and x[1] <= upper) else False)]
                            for lower, upper in lin_ref_ranges]

            # Analyze each set of targets that has otherlapping matches
            for lin_ref_set, lin_ref_range in zip(lin_ref_sets, lin_ref_ranges):
                
                # Get the portion of the raw match feature within the linear reference range
                original_match_geom = match_features.iloc[match_id]['geometry']
                _, range_match_geom, _ = split_line_at_dists(match_geom, lin_ref_range)
                
                # Break into segments at each target segment endpoint
                t_seg_endpoints = [x for t_seg in lin_ref_set['t_seg'] for x in endpoints(t_seg)]
                t_seg_endpoint_lin_refs = [range_match_geom.project(x) for x in t_seg_endpoints]
                range_match_segments = split_line_at_dists(range_match_geom, t_seg_endpoint_lin_refs)
               
                # For each segment, see which target feature is closest
                closest_targets = [
                    nearest_neighbor(segment, GeoDataFrame(geometry=lin_ref_set['t_seg'])).index[0] 
                    for segment in range_match_segments]
                
                # Merge segments shorter than the tolerance distance to their longest neighboring segment
                min_length = min([x.length for x in range_match_segments])
                while min_length < distance_tolerance:
                    # Identify the shortest segment
                    shortest_idx = np.argmin([x.length for x in range_match_segments])
                    # Identify the segments on either side
                    adjacent_idxs = [] 
                    if shortest_idx > 0:
                        adjacent_idxs.append(shortest_idx - 1)
                    if shortest_idx < (len(range_match_segments) - 1):
                        adjacent_idxs.append(shortest_idx + 1)
                    # Only move forward if there are any segments to merge
                    if len(adjacent_idxs) > 0:
                        # Identify the longer of these                    
                        longest_adjacent_idx = adjacent_idxs[np.argmax(
                            [range_match_segments[x].length for x in adjacent_idxs])]
                        # Merge the shortest segment with its longest neighbor 
                        merged_segments = sh.ops.linemerge(
                            [range_match_segments[shortest_idx], 
                             range_match_segments[longest_adjacent_idx]])
                        # Replace neighbor with mereged segment, if it is a LineString
                        if isinstance(merged_segments, LineString):
                            range_match_segments[longest_adjacent_idx] = merged_segments
                        # Otherwise (e.g., if GeometryCollection), delete neighboring segment
                        else:
                            print('ignoring non-linestring: {}'.format(str(merged_segments)))
                            del range_match_segments[longest_adjacent_idx]
                            del closest_targets[longest_adjacent_idx]
                        # Delete shortest segment and target ID
                        # (short segment effectively inherets target from longer segment)
                        del range_match_segments[shortest_idx]
                        del closest_targets[shortest_idx]                         
                        # Recalculate minimum segment length
                        min_length = min([x.length for x in range_match_segments])
                        # Avoid getting caught in infinite loop if only one segment
                        # and still shorter than distance tolerance
                        if len(range_match_segments) == 1:
                            break
                    else:
                        break

                # Group adjacent segments with the same target
                groups = [list(group) for _, group in itertools.groupby(
                    zip(closest_targets, range_match_segments), key=lambda x: x[0])]
                lin_ref_set_idxs = [group[0][0] for group in groups]
                match_segments = [[x[1] for x in group] for group in groups]
                match_segments = [sh.ops.linemerge(x) for x in match_segments]
                
                # Calculate the match prop and lin_ref bounds for the grouped match segments
                match_props = [x.length/match_geom.length for x in match_segments]
                match_lin_refs = [tuple([match_geom.project(point) for point in endpoints(segment)]) 
                                  for segment in match_segments]

                # Update match info for the chosen target
                for lin_ref_set_idx, match_prop, match_segment, match_lin_ref in zip(
                    lin_ref_set_idxs, match_props, match_segments, match_lin_refs):

                    lin_ref_set.at[lin_ref_set_idx, 'm_prop'] = match_prop
                    lin_ref_set.at[lin_ref_set_idx, 'm_seg'] = match_segment
                    lin_ref_set.at[lin_ref_set_idx, 'm_linref'] = match_lin_ref
                    lin_ref_set.at[lin_ref_set_idx, 'h_tm'] = directed_hausdorff(
                        lin_ref_set.at[lin_ref_set_idx, 't_seg'], match_segment)
                    lin_ref_set.at[lin_ref_set_idx, 'h_mt'] = directed_hausdorff(
                        match_segment, lin_ref_set.at[lin_ref_set_idx, 't_seg'])

                # Remove match info for other targets in set
                other_lin_ref_set_idxs = [x for x in lin_ref_set.index
                                          if x not in lin_ref_set_idxs]
                for lin_ref_set_idx in other_lin_ref_set_idxs:
                    lin_ref_set.at[lin_ref_set_idx, 't_prop'] = np.nan
                    # lin_ref_set.at[lin_ref_set_idx, 't_seg'] = np.nan ########### Maybe don't get rid of the t_seg?
                    lin_ref_set.at[lin_ref_set_idx, 't_linref'] = np.nan
                    lin_ref_set.at[lin_ref_set_idx, 'm_prop'] = np.nan
                    lin_ref_set.at[lin_ref_set_idx, 'm_seg'] = np.nan
                    lin_ref_set.at[lin_ref_set_idx, 'm_linref'] = np.nan
                    lin_ref_set.at[lin_ref_set_idx, 'h_tm'] = np.nan
                    lin_ref_set.at[lin_ref_set_idx, 'h_mt'] = np.nan
                    lin_ref_set.at[lin_ref_set_idx, 'match_index'] = np.nan

                # Remove original lin_ref_set rows from the operating_target_features
                operating_target_features = operating_target_features.drop(lin_ref_set.index)               

                # Append rows from lin_ref_set back onto operating_target_features
                operating_target_features = operating_target_features.append(
                    lin_ref_set, ignore_index=True)

    # Drop stats columns if not specifically requested
    if (closest_match or closest_target) and not match_stats:
        operating_target_features = operating_target_features.drop(
            columns=['h_tm','t_prop','t_seg','h_mt','m_prop','m_seg'])

    # Gather values from fields of match features
    if match_fields and isinstance(match_fields, bool):
        match_fields = match_features.columns.tolist()
        match_fields.remove('geometry')
    elif isinstance(match_fields, list):
        match_fields = match_fields
    else:
        match_fields = []
    if match_strings and (match_strings[1] not in match_fields):
        match_fields.append(match_strings[1])

    # Convert empty lists to np.nan
    operating_target_features['match_index'] = operating_target_features['match_index'].apply(
        lambda x: np.nan if (isinstance(x, list) and len(x) == 0) else x)

    # Copy over each column in match_fields
    for field in match_fields:
        try:
            operating_target_features[field] = (
                operating_target_features['match_index'].apply(
                    lambda x: [
                        match_features.at[i, field] 
                        if pd.notnull(x)  
                        else np.nan
                        for i in listify(x)]))
        except:
            print(operating_target_features['match_index'])
      
    # Join operating target features back onto all target features
    target_features = target_features.merge(
        operating_target_features.drop(columns=['geometry']), 
        how='outer', left_index=True, right_on='target_index', suffixes=field_suffixes)
    
    # Sort by original index
    target_features = target_features.sort_values(['target_index'])

    # Move original index to front if target features expanded (so there are multiple entries for some original indices)
    if expand_target_features:
        target_features = df_first_column(target_features, 'target_index')
        target_features = target_features.reset_index(drop=True)
    # Otherwise, set the index to the original index columns
    else:
        target_features = target_features.set_index('target_index')
        # Delete the index name
        del target_features.index.name

    # Convert empty lists to NaN
    target_features = target_features.applymap(
        lambda x: np.nan if x == [] else x)

    # Convert single-element lists to their sole elements
    target_features = target_features.applymap(
        lambda x: x[0] if (isinstance(x, list) and len(x) == 1) else x)

    # Calculate string matches, if specified
    if match_strings:
        def fuzzy_score(row, col_a, col_b):
            a = row[col_a]
            b = row[col_b]
            def standardize_and_score(a, b):
                a = standardize_streetname(str(a))
                b = standardize_streetname(str(b))
                return (fuzz.token_set_ratio(a, b) / 100)
            # Inputs could be lists, so make them lists if they aren't
            a_list = listify(a)
            b_list = listify(b)
            # Get fuzzy scores for each string combination
            scores = []
            for a in a_list:
                for b in b_list:
                    if (pd.notnull(a) and pd.notnull(b)):
                        scores.append(standardize_and_score(a, b))
            if len(scores) > 0:
                return scores
            else:
                return np.nan

        target_string, match_string = match_strings
        if match_string in original_target_feature_columns:
            target_string = target_string + field_suffixes[0]
            match_string = match_string + field_suffixes[1]
        target_features['match_strings'] = target_features.apply(
            fuzzy_score, args=(target_string, match_string), axis=1)

    # Move the geometry column to the end
    target_features = df_last_column(target_features, 'geometry')
    # Ensure that crs is the same as original
    target_features.crs = original_crs

    # Report done
    if verbose:
        print('100% ({} segments) complete after {:04.2f} minutes'.format(counter, (time()-start) / 60))

    return target_features


def _lookup(key, dictionary):
    """Lookup a key among either a dictionary's keys or values.
    """
    if key in dictionary.values():
        return key
    elif key in dictionary:
        return dictionary[key]
    else:
        return None 

def _lookup_direction(value):
    """Convert directions to abbreviations.
    """
    directions = {
        'north': 'n',
        'northeast': 'ne',
        'east': 'e',
        'southeast': 'se',
        'south': 's',
        'southwest': 'sw',
        'west': 'w',
        'northwest': 'nw'
        }
    value = value.lower()
    if value in directions:
        value = _lookup(value, directions)
    return value.upper()

def _lookup_street_type(value):
    """Convert street types to abbreviations.
    """
    street_types = {
        'allee': 'aly',
        'alley': 'aly',
        'ally': 'aly',
        'anex': 'anx',
        'annex': 'anx',
        'annx': 'anx',
        'arcade': 'arc',
        'av': 'ave',
        'aven': 'ave',
        'avenu': 'ave',
        'avenue': 'ave',
        'avn': 'ave',
        'avnue': 'ave',
        'bayoo': 'byu',
        'bayou': 'byu',
        'beach': 'bch',
        'bend': 'bnd',
        'bluf': 'blf',
        'bluff': 'blf',
        'bluffs': 'blfs',
        'bot': 'btm',
        'bottm': 'btm',
        'bottom': 'btm',
        'boul': 'blvd',
        'boulevard': 'blvd',
        'boulv': 'blvd',
        'branch': 'br',
        'brdge': 'brg',
        'bridge': 'brg',
        'brnch': 'br',
        'brook': 'brk',
        'brooks': 'brks',
        'burg': 'bg',
        'burgs': 'bgs',
        'bypa': 'byp',
        'bypas': 'byp',
        'bypass': 'byp',
        'byps': 'byp',
        'camp': 'cp',
        'canyn': 'cyn',
        'canyon': 'cyn',
        'cape': 'cpe',
        'causeway': 'cswy',
        'causway': 'cswy',
        'cen': 'ctr',
        'cent': 'ctr',
        'center': 'ctr',
        'centers': 'ctrs',
        'centr': 'ctr',
        'centre': 'ctr',
        'circ': 'cir',
        'circl': 'cir',
        'circle': 'cir',
        'circles': 'cirs',
        'ck': 'crk',
        'cliff': 'clf',
        'cliffs': 'clfs',
        'club': 'clb',
        'cmp': 'cp',
        'cnter': 'ctr',
        'cntr': 'ctr',
        'cnyn': 'cyn',
        'common': 'cmn',
        'corner': 'cor',
        'corners': 'cors',
        'course': 'crse',
        'court': 'ct',
        'courts': 'cts',
        'cove': 'cv',
        'coves': 'cvs',
        'cr': 'crk',
        'crcl': 'cir',
        'crcle': 'cir',
        'crecent': 'cres',
        'creek': 'crk',
        'crescent': 'cres',
        'cresent': 'cres',
        'crest': 'crst',
        'crossing': 'xing',
        'crossroad': 'xrd',
        'crscnt': 'cres',
        'crsent': 'cres',
        'crsnt': 'cres',
        'crssing': 'xing',
        'crssng': 'xing',
        'crt': 'ct',
        'curve': 'curv',
        'dale': 'dl',
        'dam': 'dm',
        'div': 'dv',
        'divide': 'dv',
        'driv': 'dr',
        'drive': 'dr',
        'drives': 'drs',
        'drv': 'dr',
        'dvd': 'dv',
        'estate': 'est',
        'estates': 'ests',
        'exp': 'expy',
        'expr': 'expy',
        'express': 'expy',
        'expressway': 'expy',
        'expw': 'expy',
        'extension': 'ext',
        'extensions': 'exts',
        'extn': 'ext',
        'extnsn': 'ext',
        'falls': 'fls',
        'ferry': 'fry',
        'field': 'fld',
        'fields': 'flds',
        'flat': 'flt',
        'flats': 'flts',
        'ford': 'frd',
        'fords': 'frds',
        'forest': 'frst',
        'forests': 'frst',
        'forg': 'frg',
        'forge': 'frg',
        'forges': 'frgs',
        'fork': 'frk',
        'forks': 'frks',
        'fort': 'ft',
        'freeway': 'fwy',
        'freewy': 'fwy',
        'frry': 'fry',
        'frt': 'ft',
        'frway': 'fwy',
        'frwy': 'fwy',
        'garden': 'gdn',
        'gardens': 'gdns',
        'gardn': 'gdn',
        'gateway': 'gtwy',
        'gatewy': 'gtwy',
        'gatway': 'gtwy',
        'glen': 'gln',
        'glens': 'glns',
        'grden': 'gdn',
        'grdn': 'gdn',
        'grdns': 'gdns',
        'green': 'grn',
        'greens': 'grns',
        'grov': 'grv',
        'grove': 'grv',
        'groves': 'grvs',
        'gtway': 'gtwy',
        'harb': 'hbr',
        'harbor': 'hbr',
        'harbors': 'hbrs',
        'harbr': 'hbr',
        'haven': 'hvn',
        'havn': 'hvn',
        'height': 'hts',
        'heights': 'hts',
        'hgts': 'hts',
        'highway': 'hwy',
        'highwy': 'hwy',
        'hill': 'hl',
        'hills': 'hls',
        'hiway': 'hwy',
        'hiwy': 'hwy',
        'hllw': 'holw',
        'hollow': 'holw',
        'hollows': 'holw',
        'holws': 'holw',
        'hrbor': 'hbr',
        'ht': 'hts',
        'hway': 'hwy',
        'inlet': 'inlt',
        'island': 'is',
        'islands': 'iss',
        'isles': 'isle',
        'islnd': 'is',
        'islnds': 'iss',
        'jction': 'jct',
        'jctn': 'jct',
        'jctns': 'jcts',
        'junction': 'jct',
        'junctions': 'jcts',
        'junctn': 'jct',
        'juncton': 'jct',
        'key': 'ky',
        'keys': 'kys',
        'knol': 'knl',
        'knoll': 'knl',
        'knolls': 'knls',
        'la': 'ln',
        'lake': 'lk',
        'lakes': 'lks',
        'landing': 'lndg',
        'lane': 'ln',
        'lanes': 'ln',
        'ldge': 'ldg',
        'light': 'lgt',
        'lights': 'lgts',
        'lndng': 'lndg',
        'loaf': 'lf',
        'lock': 'lck',
        'locks': 'lcks',
        'lodg': 'ldg',
        'lodge': 'ldg',
        'loops': 'loop',
        'manor': 'mnr',
        'manors': 'mnrs',
        'meadow': 'mdw',
        'meadows': 'mdws',
        'medows': 'mdws',
        'mill': 'ml',
        'mills': 'mls',
        'mission': 'msn',
        'missn': 'msn',
        'mnt': 'mt',
        'mntain': 'mtn',
        'mntn': 'mtn',
        'mntns': 'mtns',
        'motorway': 'mtwy',
        'mount': 'mt',
        'mountain': 'mtn',
        'mountains': 'mtns',
        'mountin': 'mtn',
        'mssn': 'msn',
        'mtin': 'mtn',
        'neck': 'nck',
        'orchard': 'orch',
        'orchrd': 'orch',
        'overpass': 'opas',
        'ovl': 'oval',
        'parks': 'park',
        'parkway': 'pkwy',
        'parkways': 'pkwy',
        'parkwy': 'pkwy',
        'passage': 'psge',
        'paths': 'path',
        'pikes': 'pike',
        'pine': 'pne',
        'pines': 'pnes',
        'pk': 'park',
        'pkway': 'pkwy',
        'pkwys': 'pkwy',
        'pky': 'pkwy',
        'place': 'pl',
        'plain': 'pln',
        'plaines': 'plns',
        'plains': 'plns',
        'plaza': 'plz',
        'plza': 'plz',
        'point': 'pt',
        'points': 'pts',
        'port': 'prt',
        'ports': 'prts',
        'prairie': 'pr',
        'prarie': 'pr',
        'prk': 'park',
        'prr': 'pr',
        'rad': 'radl',
        'radial': 'radl',
        'radiel': 'radl',
        'ranch': 'rnch',
        'ranches': 'rnch',
        'rapid': 'rpd',
        'rapids': 'rpds',
        'rdge': 'rdg',
        'rest': 'rst',
        'ridge': 'rdg',
        'ridges': 'rdgs',
        'river': 'riv',
        'rivr': 'riv',
        'rnchs': 'rnch',
        'road': 'rd',
        'roads': 'rds',
        'route': 'rte',
        'rvr': 'riv',
        'shoal': 'shl',
        'shoals': 'shls',
        'shoar': 'shr',
        'shoars': 'shrs',
        'shore': 'shr',
        'shores': 'shrs',
        'skyway': 'skwy',
        'spng': 'spg',
        'spngs': 'spgs',
        'spring': 'spg',
        'springs': 'spgs',
        'sprng': 'spg',
        'sprngs': 'spgs',
        'spurs': 'spur',
        'sqr': 'sq',
        'sqre': 'sq',
        'sqrs': 'sqs',
        'squ': 'sq',
        'square': 'sq',
        'squares': 'sqs',
        'station': 'sta',
        'statn': 'sta',
        'stn': 'sta',
        'str': 'st',
        'strav': 'stra',
        'strave': 'stra',
        'straven': 'stra',
        'stravenue': 'stra',
        'stravn': 'stra',
        'stream': 'strm',
        'street': 'st',
        'streets': 'sts',
        'streme': 'strm',
        'strt': 'st',
        'strvn': 'stra',
        'strvnue': 'stra',
        'sumit': 'smt',
        'sumitt': 'smt',
        'summit': 'smt',
        'terr': 'ter',
        'terrace': 'ter',
        'throughway': 'trwy',
        'tpk': 'tpke',
        'tr': 'trl',
        'trace': 'trce',
        'traces': 'trce',
        'track': 'trak',
        'tracks': 'trak',
        'trafficway': 'trfy',
        'trail': 'trl',
        'trails': 'trl',
        'trk': 'trak',
        'trks': 'trak',
        'trls': 'trl',
        'trnpk': 'tpke',
        'trpk': 'tpke',
        'tunel': 'tunl',
        'tunls': 'tunl',
        'tunnel': 'tunl',
        'tunnels': 'tunl',
        'tunnl': 'tunl',
        'turnpike': 'tpke',
        'turnpk': 'tpke',
        'underpass': 'upas',
        'union': 'un',
        'unions': 'uns',
        'valley': 'vly',
        'valleys': 'vlys',
        'vally': 'vly',
        'vdct': 'via',
        'viadct': 'via',
        'viaduct': 'via',
        'view': 'vw',
        'views': 'vws',
        'vill': 'vlg',
        'villag': 'vlg',
        'village': 'vlg',
        'villages': 'vlgs',
        'ville': 'vl',
        'villg': 'vlg',
        'villiage': 'vlg',
        'vist': 'vis',
        'vista': 'vis',
        'vlly': 'vly',
        'vst': 'vis',
        'vsta': 'vis',
        'walks': 'walk',
        'well': 'wl',
        'wells': 'wls',
        'wy': 'way',
        }
    value = value.lower()
    if value in street_types:
        value = _lookup(value, street_types)
    return value.title()
    
def standardize_streetname(name):
    """Standardize street names with title case and common abbreviations.

    Parameters
    ----------
    name : :obj:`str`
        Street name to standardize

    Results
    -------
    :obj:`str`
        Standardized street name.
        If unable to standardize, original street name.
    """
    try:
        tagged_streetname, _ = usaddress.tag(name)
        for key, value in tagged_streetname.items():
            if key == 'StreetNamePreModifier':
                tagged_streetname[key] = value.title()
            elif key == 'StreetNamePreDirectional':
                tagged_streetname[key] = _lookup_direction(value)
            elif key == 'StreetName':
                tagged_streetname[key] = value.title()
            elif key == 'StreetNamePostType':
                tagged_streetname[key] = _lookup_street_type(value)
            else:
                tagged_streetname[key] = value
        return ' '.join(tagged_streetname.values())
    except usaddress.RepeatedLabelError as e:
        # If tagging streetname parts fails, return original string
        return e.original_string

