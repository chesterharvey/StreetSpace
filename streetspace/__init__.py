################################################################################
# Module: __init__.py
# Description: StreetSpace - measure and analyze streetscapes and street networks
# License: MIT
################################################################################

from .network import *
from .geometry import *
from .utils import *
from .streetscape import *

__version__ = '0.1.0'

# link old function names to new ones
# pointsAlongLines = points_along_line
# splitLineByPoint = split_line_at_points
# getLineSegment = segment
# lineAzimuth = azimuth
# splitLineAtVertices = split_line_at_vertices
# endPoints = endpoints
# getMidpoint = midpoint
# azimuthAtDistance = azimuth_at_distance
# drawLineAtAngle = line_by_azimuth
# closestPointAlongNetwork = closest_network_point
# insertNode = insert_node
# splitGDFLines = split_gdf_lines
# boundsBox
# centroidGDF