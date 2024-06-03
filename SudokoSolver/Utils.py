import cv2
import numpy as np

def convert_dounle_tuples_list_to_int(tuples_list):
    new_list = []
    for l in tuples_list:
        new_list.append([])
        for t in l:
            new_list[-1].append((int(t[0]), int(t[1])))
    return new_list

def sort_list(original_list):
    new_list = np.array(original_list,dtype = np.float32)
    print(len(original_list))
    c = new_list.reshape((100,2))
    c2 = c[np.argsort(c[:,1])]

    b = np.vstack([c2[i*10:(i+1)*10][np.argsort(c2[i*10:(i+1)*10,0])] for i in range(10)])
    return b.reshape((10,10,2))


def remove_last_row_and_column(list_2d):
    return [row[:-1] for row in list_2d[:-1]]

def order_points(pts):
    pts = np.array(pts)
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
