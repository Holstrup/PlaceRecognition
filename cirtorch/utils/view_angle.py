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


# view_distance: How far out can we look?
VIEW_DISTANCE = 50   #Meters 

# view_angle: what is our field of view?
VIEW_ANGLE = math.pi / 2 #math.pi / 4 + math.pi / 8 #+ math.pi / 16 #math.pi / 2


def to_radians(angle):
    cartesian_angle = (450 - angle) % 360
    return cartesian_angle * math.pi / 180

def calc_angles(ca, view_angle = VIEW_ANGLE):
    return (ca - view_angle / 2, ca + view_angle / 2)

def calc_next_point(x, y, angle, view_distance=VIEW_DISTANCE):
    return np.column_stack((x + view_distance * np.cos(angle), y + view_distance * np.sin(angle))).tolist()

def iou(polygon1, polygon2):
    intersection = polygon1.intersection(polygon2)
    return intersection.area / (polygon1.area + polygon2.area - intersection.area)

def ious(query_polygon, db_polygon):
    return [iou(query_polygon,polygon) for polygon in db_polygon]

def field_of_view(points, approximation=10, convert_to_radians=True):
    polygons = []
    for point in points:
        northing, easting, angle = point
        if convert_to_radians:
            angle = to_radians(angle)
        starting_angle, end_angle = calc_angles(angle)
        points = [[northing, easting]]

        points.extend(calc_next_point(northing, easting, np.linspace(starting_angle, end_angle, approximation)))
        
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
    print(df['lat'].iloc[0], ',',df['lon'].iloc[0])
    return [easting, northing, to_radians(ca)]

def get_coordinates(keys):
    points = [get_coordinate(keys[0], q_postproc, q_raw)]
    for key in keys[1:]:
        points.append(get_coordinate(key, db_postproc, db_raw))
    return points

#keys = ['0NvpSEDZd8Ll_N6YDaf8dA' ,'EvWyELiNjmcgPV5Mu6P8ew','LgZgiqaR-Vm4n8Ly8RtI-A','kvRQa8GKJtt73uwhBGNxSw','_rOfyHfpkLW39p1uREzQmA','94GS7xEn7ySg7yLdlVfkKw','Rjptg8UTfmJkIiJjPy-I5w', 'BqBF-96_NViSVwSjc5WeBQ']
#keys = ['KsiCcR_YbcQnNAsKafSOng', 'tFmc-wK7A0eigPf9KhLHVQ', 'g7wfAspdwkiDfvknUdkZgg', 'pAG4DSoggEl5WVYUWjAEIA', 'l40wawAhi2TL-CZuzfrYig'] #1631
#keys = ['MRfIz0MpoUP5LApkt5GwhA', 'TK6RLS3e8Oa7wqciYC75Ow', 'Tjsn1erZ7GdbeAJAZfDYDA', 'tqin7Zzu0dCGFZmrzhQCCw', 'BaaM4Qvf3VMvjiG1apeFWQ'] #3700
#keys = ['DDb7lapO-czjhb6o_J1MxA', 'zFzarHuCvI73RJf_7MlkLQ', 'VrJfd57eglX5LskATygIiQ', 'XF9EaQsEE5V3WyO9sNu-6A', 'KiFXKBjFgBIOondz8Rm2Cg'] #3220
#keys = ['6jOFome0L5-qRaYGQW1doA', 'VrJfd57eglX5LskATygIiQ', 'KiFXKBjFgBIOondz8Rm2Cg', 'zFzarHuCvI73RJf_7MlkLQ', 'AchY4D1tRFwQLT2Bawoc0Q', 'XF9EaQsEE5V3WyO9sNu-6A']

#keys = ['y2xotXR_HrbadIxXk24EkQ', 'u894ydL4oc7a4aLUwk9rwg', 'D6dYsU5W2JZfyMRdH-MZng', 'M06sx0JtP7X5MiPGzBh4yg', 'q9_ykmaKaJDYwKG0B_8CBg', 'E0tp9rcW0ooVyYoyobKvTg', 'f9ng8k9Ngt4UWja_a-mjaw', 'CxYjkOBcps80szVVKzfe5A']


#points = get_coordinates(keys)
#pol = field_of_view(points)
#print(ious(pol[0], pol[1:]))
#plot_fov(pol)
