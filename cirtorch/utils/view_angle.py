from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

#city = 'london'

#q_postproc = pd.read_csv(f'/Users/alexanderholstrup/git/VisualPlaceRecognition/cnnimageretrieval-pytorch/notebooks/data/IT5/MSEAndContrastive400/Train/Images/{city}/query/postprocessed.csv')
#q_raw = pd.read_csv(f'/Users/alexanderholstrup/git/VisualPlaceRecognition/cnnimageretrieval-pytorch/notebooks/data/IT5/MSEAndContrastive400/Train/Images/{city}/query/raw.csv')

#db_postproc = pd.read_csv(f'/Users/alexanderholstrup/git/VisualPlaceRecognition/cnnimageretrieval-pytorch/notebooks/data/IT5/MSEAndContrastive400/Train/Images/{city}/database/postprocessed.csv')
#db_raw = pd.read_csv(f'/Users/alexanderholstrup/git/VisualPlaceRecognition/cnnimageretrieval-pytorch/notebooks/data/IT5/MSEAndContrastive400/Train/Images/{city}/database/raw.csv')
"""
q_postproc = pd.read_csv(f'/Users/alexanderholstrup/Desktop/miami/query/postprocessed.csv')
q_raw = pd.read_csv(f'/Users/alexanderholstrup/Desktop/miami/query/raw.csv')
db_postproc = pd.read_csv(f'/Users/alexanderholstrup/Desktop/miami/database/postprocessed.csv')
db_raw = pd.read_csv(f'/Users/alexanderholstrup/Desktop/miami/database/raw.csv')
"""
q_postproc = pd.read_csv(f'/Users/alexanderholstrup/Desktop/buenos/query/postprocessed.csv')
q_raw = pd.read_csv(f'/Users/alexanderholstrup/Desktop/buenos/query/raw.csv')
db_postproc = pd.read_csv(f'/Users/alexanderholstrup/Desktop/buenos/database/postprocessed.csv')
db_raw = pd.read_csv(f'/Users/alexanderholstrup/Desktop/buenos/database/raw.csv')

# view_distance: How far out can we look?
VIEW_DISTANCE = 50   #Meters 

# view_angle: what is our field of view?
VIEW_ANGLE = math.pi / 2 #2 * math.pi / 3

def to_radians(angle):
    cartesian_angle = (450 - angle) % 360
    return cartesian_angle * math.pi / 180

def calc_angles(ca, view_angle = VIEW_ANGLE):
    return (ca - view_angle / 2, ca + view_angle / 2)

def calc_next_point(x, y, angle, view_distance=VIEW_DISTANCE):
    return np.column_stack((x + view_distance * np.cos(angle), y + view_distance * np.sin(angle))).tolist()

def iou(polygon1, polygon2):
    intersection = polygon1.intersection(polygon2)
    return intersection.area / polygon1.area

def ious(query_polygon, db_polygon):
    return [iou(query_polygon,polygon) for polygon in db_polygon]

def field_of_view(points, approximation=10, convert_to_radians=True, view_angle=VIEW_ANGLE, view_distance=VIEW_DISTANCE):
    polygons = []
    for point in points:
        northing, easting, angle = point
        if convert_to_radians:
            angle = to_radians(angle)
        starting_angle, end_angle = calc_angles(angle, view_angle)
        points = [[northing, easting]]

        points.extend(calc_next_point(northing, easting, np.linspace(starting_angle, end_angle, approximation), view_distance))
        
        polygons.append(Polygon(points))
    return polygons
    
def plot_fov(polygon_list):
    Xq = np.array([list(i) for i in polygon_list[0].exterior.coords])
    plt.scatter(Xq[0, 0], Xq[0, 1], facecolor=(0,0,1,0.5))
    t1 = plt.Polygon(Xq[:,:], facecolor=(0,1,0,0.5))
    plt.gca().add_patch(t1)

    for polygon in polygon_list[1:]:
        Xp = np.array([list(i) for i in polygon.exterior.coords])
        plt.scatter(Xp[0, 0], Xp[0, 1], facecolor=(0,0,1,0.5))
        t1 = plt.Polygon(Xp[:,:], facecolor=(1,0,0,0.3))
        plt.gca().add_patch(t1)

    plt.xlim((min(Xq[:,0]) - 50, max(Xq[:,0]) + 50))
    plt.ylim((min(Xq[:,1]) - 50, max(Xq[:,1]) + 50))
    plt.show()

def get_coordinate(key, postproc, raw):
    df = postproc.loc[postproc['key'] == key]
    northing, easting = df['northing'].iloc[0], df['easting'].iloc[0]
    df = raw.loc[raw['key'] == key]
    ca = df['ca'].iloc[0]
    #print(df['lat'].iloc[0], ',',df['lon'].iloc[0])
    return [easting, northing, to_radians(ca)]

def get_coordinates(keys):
    points = [get_coordinate(keys[0], q_postproc, q_raw)]
    for key in keys[1:]:
        points.append(get_coordinate(key, db_postproc, db_raw))
    return points

def distance(query, positive):
    return np.linalg.norm(np.array(query) - np.array(positive))

def get_distances(keys):
    dis = []
    qpoint = get_coordinate(keys[0], q_postproc, q_raw)[0:2]
    for key in keys[1:]:
        point = get_coordinate(key, db_postproc, db_raw)[0:2]
        dis.append(distance(qpoint, point))
    return dis


"""
#points = [[0,0,0], [0,0,40]]
#points = [[0,0,0], [25,0,0]]

#keys = ['9X43HXskdI-kJnt4CRMY0Q', 'QfRBMkBuwFCfE28L8xaa8A', 'ZJWzYIRyhmOVBi5v-T7Prw', 'VoXJWp_wtYJ5oTJZ32cCkg', 'GCnrKGs0uLLKDcezDWHd8A', 'XpEitw6dn3ocrCdHaRq35g']
#keys = ['9X43HXskdI-kJnt4CRMY0Q', 'QfRBMkBuwFCfE28L8xaa8A', 'ZJWzYIRyhmOVBi5v-T7Prw', 'VoXJWp_wtYJ5oTJZ32cCkg', 'GCnrKGs0uLLKDcezDWHd8A', 'XpEitw6dn3ocrCdHaRq35g']

#keys = ['olucTfqJt_xMiwr9Eavimg', 'W9Vn-RC_OVPWWxI3Ug63eQ', 'GyIekSEM2Tevd9LX68-wFw', 'cJbU-IH3TbMLIH4yUqQudg', 'bEnnHogqXExPfH9Y5h3ojg', 'jlWWKhZ2SC1LBOt4OVV3cA']

#keys = ['pIaUxMrK2pAF-UG807_SyA', 'PBp7NDYX0oc9VeiMouuybw', '1bWZC2-ORUDJEsMJWgeybA', 'SFn27bl67Evgw0ni1Qgzwg', 'gwfP-H9an_huFicbG0mCyw', '7VeeN10hJeFupkmNd9Qwcw']
keys = ['pIaUxMrK2pAF-UG807_SyA', '1bWZC2-ORUDJEsMJWgeybA', 'PBp7NDYX0oc9VeiMouuybw', 'r9fhP-Su_TRmn6ZjCYYpqg', 'xmvabOMcoGUB3E2bWmwmvQ', 'iWO4Mh7duE7Tyqe6YEUoSg']
#keys = ['pIaUxMrK2pAF-UG807_SyA', '1bWZC2-ORUDJEsMJWgeybA', 'PBp7NDYX0oc9VeiMouuybw', 'SFn27bl67Evgw0ni1Qgzwg', '7VeeN10hJeFupkmNd9Qwcw', '8cKYY6CDij_o74J4-CR4Vw']

#keys = ['z1Tm6jrIkhE4jUSm14rYt8', 'SbAUjyJNMs1mPJXMQJGXOf', 'cdmsgSBuXbPGSnF9S2EevZ', 'um2egmv2F2n9lPobP5vUBj', 'LRVsecPCzqK_npV0aE21ok', 'PqyDOMPbhBuvd-_UuSz8oM']
#keys = ['z1Tm6jrIkhE4jUSm14rYt8', 'ZWuMn4X03a8LgQqC98v5kb', 'LRVsecPCzqK_npV0aE21ok', 'cdmsgSBuXbPGSnF9S2EevZ', 'cvb6uFIfqLMAO0baOoRLB3', 'bvqy0Sfjbf5awBwMFe_-R3']
#keys = ['z1Tm6jrIkhE4jUSm14rYt8', 'ZWuMn4X03a8LgQqC98v5kb', 'LRVsecPCzqK_npV0aE21ok', 'cdmsgSBuXbPGSnF9S2EevZ', 'cvb6uFIfqLMAO0baOoRLB3', 'um2egmv2F2n9lPobP5vUBj']


points = get_coordinates(keys)
VIEW_ANGLE = to_radians(90)
pol = field_of_view(points, convert_to_radians=False)
print('IOUS: ', ious(pol[0], pol[1:]))
print('GPS: ', get_distances(keys))

plt.scatter(get_distances(keys), ious(pol[0], pol[1:]))
plt.xlim((0, 25))
plt.ylim((0, 1))
plt.show()
plot_fov(pol)


iou_scores = []
#points = [[0,0,0], [0,0,40]]
points = [[0,0,0], [25,0,0]]
angles = np.linspace(0.01, 2*math.pi, num=100)
#distances = np.linspace(1, 100, num=100)
for i in range(len(angles)):
    VIEW_ANGLE = angles[i]
    #VIEW_DISTANCE = distances[i]
    pol = field_of_view(points, convert_to_radians=True, view_angle=VIEW_ANGLE, view_distance=VIEW_DISTANCE)
    iou_scores.append(ious(pol[0], pol[1:])[0])
plt.scatter(angles, iou_scores)
plt.ylim((0, 1))
plt.show()
"""