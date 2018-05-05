##############################################################################
# Module: conflation.py
# Description: Functions to conflate linestring geometries.
# License: MIT
##############################################################################

from .geometry import *

def match_lines_by_midpoint(target_features, match_features, distance_tolerance, 
    match_features_sindex=None, azimuth_tolerance=None, length_tolerance=None,
    incidence_tolerance=None, match_by_score=False, match_fields=False, match_stats=False, 
    constrain_target_features=False, target_features_sindex=None,
    match_vectors=False, verbose=False):
    """Conflate attributes between line features based on midpoint proximity.
    
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
    adjacent_segment = None
    # Split "b" into segments at these points
    if len(split_points) > 0:
        b_segments = split_line_at_points(b, split_points)
        # Identify which (if any) segment runs parallel to "a" based on hausdorff dist
        along = [segment for segment in b_segments 
                 if directed_hausdorff(segment, a) < distance_tolerance]
        if len(along) > 0:
            adjacent_segment = along[0]
    return adjacent_segment


def match_lines_by_hausdorff(target_features, match_features, distance_tolerance, 
    azimuth_tolerance=None, match_features_sindex=None, match_fields=False, match_stats=False, 
    constrain_target_features=False, target_features_sindex=None,
    match_vectors=False, expand_target_features=False, verbose=False):
    """Conflate attributes between line features based on midpoint proximity.
    
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
    match_types = []
    match_h_tc = []
    match_t_props = []
    match_t_segs = []
    match_h_ct = []
    match_c_props = []
    match_c_segs = []
    if match_vectors:
        match_vectors = []
      
    # Iterate through target features:
    for i, target in enumerate(operating_target_features.geometry):

        # Roughly filter candidates with a spatial index
        search_area = target.buffer(distance_tolerance).bounds
        candidate_IDs = list(match_features_sindex.intersection(search_area))
        candidates = match_features[['geometry']].iloc[candidate_IDs].reset_index()

        # return target, candidates
       
        # Calculate Hausdorff distances from feature to each candidate (h_fc)
        h_tc_list = [directed_hausdorff(target, candidate) for candidate in candidates.geometry]
        candidates['h_tc'] = pd.Series(h_tc_list)

        # Calculate Hausdorff distances from each candidate to feature (h_cf)
        h_ct_list = [directed_hausdorff(candidate, target) for candidate in candidates.geometry]
        candidates['h_ct'] = pd.Series(h_ct_list)        

        # Initiate lists to store matches
        match_ids = []
        h_tcs = []
        t_proportions = []
        t_segments = []
        h_cts = []
        c_proportions = []
        c_segments = []

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
            
            # Initialize default match values
            h_tc = None
            t_proportion = None
            t_segment = None
            h_ct = None
            c_proportion = None
            c_segment = None
            
            # 1:1
            if (
                (candidate.h_tc <= distance_tolerance) and 
                (candidate.h_ct <= distance_tolerance) and
                azimuth_match(target, candidate.geometry, azimuth_tolerance)):
                # Whole target matches candidate
                h_tc = candidate.h_tc
                t_proportion = 1
                t_segment = target
                # Whole candidate matches target
                h_ct = candidate.h_ct
                c_proportion = 1
                c_segment = candidate.geometry
                
            # n:1
            elif (
                (candidate.h_tc <= distance_tolerance) and 
                (candidate.h_ct > distance_tolerance)):
                # Find the candidate segment matching the target
                candidate_segment = find_parallel_segment(
                    target, candidate.geometry, distance_tolerance)

                if (candidate_segment and 
                    azimuth_match(target, candidate_segment, azimuth_tolerance)):
                    # Whole target matches candidate
                    h_tc = candidate.h_tc
                    t_proportion = 1
                    t_segment = target
                    # Calculate proportion of candidate included in segment
                    h_ct = candidate.h_ct
                    c_proportion = candidate_segment.length / candidate.geometry.length
                    c_segment = candidate_segment

            # 1:n
            elif (
                (candidate.h_tc > distance_tolerance) and 
                (candidate.h_ct <= distance_tolerance)):
                # Find the target segment matching the candidate
                target_segment = find_parallel_segment(
                    candidate.geometry, target, distance_tolerance)
                if (target_segment and
                    azimuth_match(target_segment, candidate.geometry, azimuth_tolerance)):
                    # Calculate proportion of target included in segment
                    h_tc = candidate.h_tc
                    t_proportion = target_segment.length / target.length
                    t_segment = target_segment
                    # Whole candidate matches target
                    h_ct = candidate.h_ct
                    c_proportion = 1
                    c_segment = candidate.geometry
            
            # potential m:n
            elif (
                (candidate.h_tc > distance_tolerance) and 
                (candidate.h_ct > distance_tolerance)):
                # See if parallel segments can be identified
                target_segment = find_parallel_segment(
                    candidate.geometry, target, distance_tolerance)
                candidate_segment = find_parallel_segment(
                    target, candidate.geometry, distance_tolerance)
                # Measure hausdorff distance (non-directed) between parallel segments
                if target_segment and candidate_segment:
                    h_tc_segment = directed_hausdorff(target_segment, candidate_segment)
                    h_ct_segment = directed_hausdorff(candidate_segment, target_segment)
                    if ((h_tc_segment <= distance_tolerance) and 
                        (h_ct_segment <= distance_tolerance) and
                        azimuth_match(target_segment, candidate_segment, azimuth_tolerance)):
                        h_tc = h_tc_segment
                        t_proportion = target_segment.length / target.length
                        t_segment = target_segment
                        h_ct = h_ct_segment
                        c_proportion = candidate_segment.length / candidate.geometry.length
                        c_segment = candidate_segment
                                     
            if t_proportion is not None:
                match_ids.append(candidate.index)
                h_tcs.append(h_tc)
                t_proportions.append(t_proportion)
                t_segments.append(t_segment)
                h_cts.append(h_ct)
                c_proportions.append(c_proportion)
                c_segments.append(c_segment)

        # Determine match type
        if len(t_proportions) == 0:
            match_types.append('1:0')
        elif (min(t_proportions) == 1) and (min(c_proportions) == 1):
            match_types.append('1:1')
        elif (min(t_proportions) == 1):
            match_types.append('n:1')
        elif (min(c_proportions) == 1):
            match_types.append('1:n')
        else:
            match_types.append('m:n')

        # Record match stats
        match_indices.append(match_ids)
        match_h_tc.append(h_tcs)
        match_t_props.append(t_proportions)
        match_t_segs.append(t_segments)
        match_h_ct.append(h_cts)
        match_c_props.append(c_proportions)
        match_c_segs.append(c_segments)

        # Construct match vector
        if isinstance(match_vectors, list):
            vectors = []
            for t_seg, c_seg in zip(t_segments, c_segments):
                if t_seg and c_seg:
                    vectors.append(LineString([midpoint(t_seg), midpoint(c_seg)]))
            match_vectors.append(vectors)

        # Report status
        if verbose:
            if counter % round(length / 10) == 0 and counter > 0:
                percent_complete = (counter // round(length / 10)) * 10
                minutes = (time()-start) / 60
                print('{}% ({} segments) complete after {:04.2f} minutes'.format(percent_complete, counter, minutes))
            counter += 1
    
    # Merge joined data with target features
    operating_target_features['match_type'] = pd.Series(
        match_types, index=operating_target_features.index)
    operating_target_features['match_id'] = pd.Series(
        match_indices, index=operating_target_features.index)
    
    if match_stats:
        operating_target_features['match_h_tc'] = pd.Series(
            match_h_tc, index=operating_target_features.index)
        operating_target_features['match_t_prop'] = pd.Series(
            match_t_props, index=operating_target_features.index)
        operating_target_features['match_t_seg'] = pd.Series(
            match_t_segs, index=operating_target_features.index)
        operating_target_features['match_h_ct'] = pd.Series(
            match_h_ct, index=operating_target_features.index)
        operating_target_features['match_c_prop'] = pd.Series(
            match_c_props, index=operating_target_features.index)
        operating_target_features['match_c_seg'] = pd.Series(
            match_c_segs, index=operating_target_features.index)
    if isinstance(match_vectors, list):
        operating_target_features['match_vectors'] = pd.Series(
            match_vectors, index=operating_target_features.index)

    # Gather values from fields of match features
    def lookup_values(match_id, match_features, field):
        values = []
        for i in match_id:
            values.append(match_features.at[i, field])
        return values
    if match_fields:
        # If specified as booleans, get all fields except geometry
        if type(match_fields) == bool:
            match_fields = match_features.columns.tolist()
            match_fields.remove('geometry')
        for field in match_fields:
            operating_target_features[field] = (
                operating_target_features['match_id'].apply(
                    lookup_values, args=(match_features, field)))
       
    # Join operating target features back onto all target features
    target_features = target_features.merge(
        operating_target_features.drop(columns=['geometry']), 
        how='left', left_index=True, right_index=True, suffixes=('', '_match'))

    # Convert empty lists to NaN
    target_features = target_features.applymap(
        lambda x: np.nan if x == [] else x)

    # Convert single-element lists to their sole elements
    target_features = target_features.applymap(
        lambda x: x[0] if (isinstance(x, list) and len(x) == 1) else x)

    # Expand targets matched with more than one candidate
    if expand_target_features:
        # Look for lists of match IDs in each row
        expanded_targets = []
        for i, target in enumerate(target_features.itertuples()):
            if isinstance(target.match_id, list):
                # Make duplicate rows for each match ID with respective attributes
                for j, match in enumerate(target.match_id):
                    new_row = target._asdict()
                    new_row.pop('Index', None)
                    for key, value in target._asdict().items():
                        if isinstance(value, list):
                            new_row[key] = value[j]
                    # Append new row to end of dataframe
                    target_features = target_features.append(new_row, ignore_index=True)
                # Mark original row for deletion
                expanded_targets.append(i)
        # Delete expanded targets
        target_features = target_features.drop(expanded_targets)
        # Replace target geometries with target segments (if not NaN)
        def replace_target_segments(row):
            if pd.notnull(row.match_t_seg):
                return row.match_t_seg
            else:
                return row.geometry
        target_features['geometry'] = target_features.apply(replace_target_segments, axis=1)
        target_features = target_features.drop(columns=['match_t_prop','match_t_seg'])

    # Report done
    if verbose:
        print('100% ({} segments) complete after {:04.2f} minutes'.format(counter, (time()-start) / 60))

    return target_features

