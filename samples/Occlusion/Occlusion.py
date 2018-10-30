import numpy as np

camera_calibration_matrix = np.array([[-572.41140, 0, 325.26110], [0, 573.57043, 242.04899], [0, 0, 1]])
camera_projection_matrix = np.array([[-572.41140, 0, 325.26110, 0], [0, 573.57043, 242.04899, 0], [0, 0, 1, 0]])


def occlusion_point_cloud(path):
    pcl = np.loadtxt(path)
    return pcl

def occlusion_pose(path, i):
    r, t, ext = [], [], []
    def str_to_list_of_floats(line):
        m = map(float, line[:-2].split(" "))
        return list(m)
    with open("%sinfo_%05d.txt"%(path, i)) as f:
        for i, line in enumerate(f):
            if i > 3 and i < 7:
                r.append(str_to_list_of_floats(line))
            if i == 8:
                t.extend(str_to_list_of_floats(line))
            if i == 10:
                ext.extend(str_to_list_of_floats(line))
    R = np.float32(r)
    T = np.float32(t)
    Ext = np.float32(ext)
    return R, T, Ext