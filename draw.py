################################################################################
# Module: draw.py
# Description: Tools for drawing and plotting
# License: MIT
################################################################################

#### Function to draw shapely polygons in a list
def drawPolygons(polygon_list, subplot):
    '''
    Draw shapely polygons stored in a list

    
    '''
    for polygon in polygon_list:
        coords = list(zip(polygon.exterior.coords.xy[0],
                          polygon.exterior.coords.xy[1]))
        subplot.add_patch(plt.Polygon(coords, 
                                 closed = True, 
                                 fill = True, 
                                 edgecolor = None))