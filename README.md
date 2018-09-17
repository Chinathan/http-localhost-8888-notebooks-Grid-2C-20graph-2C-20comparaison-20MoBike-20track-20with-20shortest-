# http-localhost-8888-notebooks-Grid-2C-20graph-2C-20comparaison-20MoBike-20track-20with-20shortest-



import shapefile
from matplotlib import *
from math import *
import numpy as np
import matplotlib.pyplot as plt
import os
from gdal import*
import shapefile 
import shapely
from shapely.geometry import *

def grid():
    starting_points_within_bbox=[]
    list_tracks=list_of_tracks_composed_by_tuples('MoBike_Tracks_10000_wgs84.csv')
    r=shapefile.Reader('/home/mifsud-couchaux/Documents/drive-download-20180710T045435Z-001/Export_Output_Bounding_Box.shp')
    shapes=r.shapes()

    Bounding_Box=shapes[0].points[:] #tableau contenant tous les points que forme le polygone 

    batchConvertCoordinates(Bounding_Box,3395,4326) #Permet de passer de EPSG3395 = projected coordinate system à l'EPSG4326 = geodetic coordinate system
#meters to degrees 

    Bbox=Polygon(batchConvertCoordinates(Bounding_Box,3395,4326)) #Créer un polygone (format shapely) à partir d'une liste de points

    for track in list_tracks: # a partir d'une liste de couples en gcj (ligne_k ici), ajoute des couples au format WGS84 à un fichier csv
        if Bbox.contains(Point(track[0]))==True: #ne récupère que les point de départ et d'arrivées inclus dans l'Inner Ring
            starting_points_within_bbox.append(track[0])

    starting_points_within_bbox_epsg4479=batchConvertCoordinates(starting_points_within_bbox, 4326,4479)
    X=np.array(starting_points_within_bbox_epsg4479)[:,0]
    Y=np.array(starting_points_within_bbox_epsg4479)[:,1]

    min_x=min(X)
    min_y=min(Y)
    max_x=max(X)
    max_y=max(Y)

    slope_x=0.6143
    step_xx= 81.365
    step_xy=49.98


    slope_y=-1.6332 
    step_yx = 30.09
    step_yy=49.14

    x_cross=(((min_y-slope_y*min_x)-(max_y-slope_x*min_x))/(slope_x-slope_y))
    y_cross=slope_x*x_cross+(max_y-slope_x*min_x)

    x=x_cross
    y=y_cross
    line_1=[]
    while x<max_x:
        line_1.append((x,y))
        x+=step_xx
        y+=step_xy

    
    grid=[]
    for i in line_1:
        x=i[0]
        y=i[1]
        perpendicular_i=[]
        while y>min_y:
            perpendicular_i.append((x,y))
            x+=step_yx
            y-=step_yy
        grid.append(perpendicular_i)
    grid.pop()

    return(grid)
    
    
    //
    
    
    import networkx as nx
import matplotlib.pyplot as plt

def graph():
    
    starting_points_within_bbox=[]
    list_tracks=list_of_tracks_composed_by_tuples('MoBike_Tracks_10000_wgs84.csv')
    r=shapefile.Reader('/home/mifsud-couchaux/Documents/drive-download-20180710T045435Z-001/Export_Output_Bounding_Box.shp')
    shapes=r.shapes()

    Bounding_Box=shapes[0].points[:] #tableau contenant tous les points que forme le polygone 

    batchConvertCoordinates(Bounding_Box,3395,4326) #Permet de passer de EPSG3395 = projected coordinate system à l'EPSG4326 = geodetic coordinate system
#meters to degrees 

    Bbox=Polygon(batchConvertCoordinates(Bounding_Box,3395,4326)) #Créer un polygone (format shapely) à partir d'une liste de points

    for track in list_tracks: # a partir d'une liste de couples en gcj (ligne_k ici), ajoute des couples au format WGS84 à un fichier csv
        if Bbox.contains(Point(track[0]))==True: #ne récupère que les point de départ et d'arrivées inclus dans l'Inner Ring
            starting_points_within_bbox.append(track[0])

    starting_points_within_bbox_epsg4479=batchConvertCoordinates(starting_points_within_bbox, 4326,4479)
    X=np.array(starting_points_within_bbox_epsg4479)[:,0]
    Y=np.array(starting_points_within_bbox_epsg4479)[:,1]

    min_x=min(X)
    min_y=min(Y)
    max_x=max(X)
    max_y=max(Y)

    slope_x=0.6143
    step_xx= 81.365
    step_xy=49.98


    slope_y=-1.6332 
    step_yx = 30.09
    step_yy=49.14

    x_cross=(((min_y-slope_y*min_x)-(max_y-slope_x*min_x))/(slope_x-slope_y))
    y_cross=slope_x*x_cross+(max_y-slope_x*min_x)
    
    G=nx.Graph()
    
    x=x_cross
    y=y_cross
    line_1=[]
    while x<max_x:
        line_1.append((x,y))
        x+=step_xx
        y+=step_xy

    
    grid=[]
    column=0
    
    for i in line_1:
        column+=1
        x=i[0]
        y=i[1]
        row=0
        while y>min_y:
            row+=1
            G.add_node((row,column),longitude=x, latitude=y,pos=(x,y))
            x+=step_yx
            y-=step_yy
    
    max_row=max(np.array(G.nodes())[:,0])
    max_column=max(np.array(G.nodes())[:,0])
    n=G.nodes()
    #créer un tableau ne contenant pas les colonnes ou lignes extremes pour ne pas modifier la taille du dictionnaire
    use=[]
    dustbin=[]
    for coord in n :
        if coord[1]==1:
            dustbin.append(coord)
        elif coord[0]==max_row:
            dustbin.append(coord)
        elif coord[1]==max_column:
            dustbin.append(coord)
        else:
            use.append(coord)
                
    
    for i,j in use:
        G.add_edge((i,j),(i,j+1),weight=95) #vers la droite
        G.add_edge((i,j),(i+1,j+1),weight=115)#diagonale vers la droite
        G.add_edge((i,j),(i+1,j),weight=60) #vers le bas 
        G.add_edge((i,j),(i+1,j-1),weight=115)#diagonale vers la gauche
    
    #nx.draw(G)
    #plt.show()
    #plt.savefig('test_graph.png')
    
    return(G)
    
    
    //
    
    
import numpy as np
import itertools
from osgeo import ogr
from scipy import spatial 
import networkx as nx
from matplotlib import *
from math import *
import matplotlib.pyplot as plt

def analysis(path_csv_file_epsg4326):

    label=[] #create a list containing the labels of the coordinates, having the same index than the coordinates in the self-titled list
    coordinates=[]
    tracks_ordered=[]
    dijkstra_lbl=[] #Stock each list composed of labels of each nodes of the shortest route for each track
    dijkstra_coord_haver=[] #Stock each list composed of haversine coordinates of each nodes of the shortest route for each trac
    dijkstra_coord_epsg4479=[]
    tracks_ordered_grid=[]
    
    G=graph()
    geo_data=G.nodes.data()
    
    #Step_1 : stocking all the coordinates of the nodes of the graph 

    for i in list(geo_data): #i =('label', {'latitude':,'utilisation':, 'longitude':,'label'}) ie type i = tuple with a string and a dic
        label.append(i[0]) 
        try:
            coordinates.append((i[1]['longitude'],i[1]['latitude']))
        except KeyError:
            pass
        
    #Step_2 : Put in order the coordinates of each track 
    
    path_to_study=paths_to_study('/home/mifsud-couchaux/Téléchargements/drive-download-20180716T105534Z-001/Export_Output_Perimeters.shp',path_csv_file_epsg4326) #Return a list of tracks within a predefined area with ArcGis
    
    for track in path_to_study:
        track_ordered=[]
        start=track[0]
        stop=track[-1]
        track_ordered.append(start)
        track=track[1:len(track)]
        tree=spatial.KDTree(track)
        i=start
        while len(track)>1:
            idx=tree.query(i)[1] #stock the index corresponding to the closest point to 'i'
            track_ordered.append(track[idx])
            i=track[idx]
            del track[idx]
            tree=spatial.KDTree(track)
        track_ordered.append(stop) #A list of ordered coordinates (for only one track)
        tracks_ordered.append(track_ordered) # A list of list of ordered coordinates
    
    #Step_3 : Stocking shortest route corresponding to each track
    
    tree=spatial.KDTree(coordinates)
    for track in tracks_ordered:
        datas=tree.query([i for i in track])
        tracks_ordered_grid.append([coordinates[datas[1][j]] for j in range(len(track))])
        shortest_route_lbl=nx.dijkstra_path(G,label[datas[1][0]],label[datas[1][-1]],weight='weight')
        
        dijkstra_lbl.append(shortest_route_lbl)
        shortest_route_epsg4479=[(G.node[i]['longitude'],G.node[i]['latitude']) for i in shortest_route_lbl]
        dijkstra_coord_epsg4479.append(shortest_route_epsg4479)
        
    return 'tracks_ordered_grid=',tracks_ordered_grid, 'dijkstra=',dijkstra_coord_epsg4479 
    
    
//

import osr
from osgeo import ogr
import pandas as pd


a=analysis('MoBike_Tracks_10000_wgs84.csv')
tracks_ordered=a[1]
djikstra=a[3]

lengths_of_ordered_paths=[]
lengths_of_shortest_paths=[]
difference_lengths=[]

fig, ax=plt.subplots()
        
for index, track in enumerate(tracks_ordered):
    line_track=ogr.Geometry(ogr.wkbLineString)
    for j in track:
        line_track.AddPoint(j[0],j[1])
    
    lengths_of_ordered_paths.append(line_track.Length())
    
    if line_track.Length()<2094:
        line_shortest_route=ogr.Geometry(ogr.wkbLineString)
        for coord in djikstra[index]:
            line_shortest_route.AddPoint(coord[0],coord[1])
        
        lengths_of_shortest_paths.append(line_shortest_route.Length())
        if line_track.Length()-line_shortest_route.Length()>0:
            difference_lengths.append(line_track.Length()-line_shortest_route.Length())
        
            x=np.array(track)[:,0]
            y=np.array(track)[:,1]
            plt.plot(x,y, color='grey')
       
            x=np.array(djikstra[index])[:,0]
            y=np.array(djikstra[index])[:,1]
            plt.plot(x,y, color='blue')

    
s_lengths=pd.Series(lengths_of_ordered_paths)
s_difference=pd.Series(difference_lengths)

print(s_lengths.describe(),s_difference.describe())

    
Repères_visuels=[(121.450446,31.251552),(121.495,31.242),(121.440833,31.225),(121.431389,31.193056),(121.498144,31.200808),(121.473056,31.232222)]
Names=['SRS','Oriental Pearl','JA Temple','St Ignacius Cath','Shanghai MOCA','People Park']

References_euclidian=batchConvertCoordinates(Repères_visuels,4326,4479)
References_euclidian=np.array(References_euclidian)
    
k=0
for i in References_euclidian:
    plt.text(i[0],i[1],Names[k], color='black')
    plt.scatter(i[0],i[1],s=10)
    k+=1
    
        
fig.set_size_inches(100,100)
plt.show()

//

import osr
from osgeo import ogr
import pandas as pd

def compare(path_MoBike_Tracks_wgs84_csv):

    a=analysis(path_MoBike_Tracks_wgs84_csv)
    tracks_ordered=a[1]
    djikstra=a[3]

    lengths_of_ordered_paths=[]
    lengths_of_shortest_paths=[]
    difference_lengths=[]
    centroide=[]

    fig, ax=plt.subplots()
        
    for index, track in enumerate(tracks_ordered):
        line_track=ogr.Geometry(ogr.wkbLineString)
        for j in track:
            line_track.AddPoint(j[0],j[1])
    
        lengths_of_ordered_paths.append(line_track.Length())
    
        if line_track.Length()<2094:
            line_shortest_route=ogr.Geometry(ogr.wkbLineString)
            for coord in djikstra[index]:
                line_shortest_route.AddPoint(coord[0],coord[1])
        
            lengths_of_shortest_paths.append(line_shortest_route.Length())
            if line_track.Length()-line_shortest_route.Length()>0:
                difference_lengths.append(line_track.Length()-line_shortest_route.Length())
        
                x_track=np.array(track)[:,0]
                y_track=np.array(track)[:,1]
                #plt.plot(x_track,y_track, color='grey')
       
                x_dij=np.array(djikstra[index])[:,0]
                y_dij=np.array(djikstra[index])[:,1]
                #plt.plot(x_dij,y_dij, color='blue')
                x_centroide_track=x_track.mean()
                x_centroide_dij=x_dij.mean()
                y_centroide_track=y_track.mean()
                y_centroide_dij=y_dij.mean()
                plt.plot([x_centroide_track,x_centroide_dij],[y_centroide_track,y_centroide_dij],color='red')

    
    s_lengths=pd.Series(lengths_of_ordered_paths)
    s_difference=pd.Series(difference_lengths)
    
    Repères_visuels=[(121.450446,31.251552),(121.495,31.242),(121.440833,31.225),(121.431389,31.193056),(121.498144,31.200808),(121.473056,31.232222)]
    Names=['SRS','Oriental Pearl','JA Temple','St Ignacius Cath','Shanghai MOCA','People Park']

    References_euclidian=batchConvertCoordinates(Repères_visuels,4326,4479)
    References_euclidian=np.array(References_euclidian)
    
    k=0
    for i in References_euclidian:
        plt.text(i[0],i[1],Names[k], color='black')
        plt.scatter(i[0],i[1],s=10)
        k+=1
    
        
    fig.set_size_inches(100,100)
    plt.show()
    
    return s_lengths.describe(),s_difference.describe()


    
