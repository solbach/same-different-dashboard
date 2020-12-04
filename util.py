import plotly.graph_objs as go
import numpy as np
from stl import mesh
from scipy.spatial.transform import Rotation as R
import math

def stl2mesh3d(stl_mesh):
    # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points)
    # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
    p, q, r = stl_mesh.vectors.shape #(p, 3, 3)
    # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
    # extract unique vertices from all mesh triangles
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p*q, r), return_inverse=True, axis=0)
    I = np.take(ixr, [3*k for k in range(p)])
    J = np.take(ixr, [3*k+1 for k in range(p)])
    K = np.take(ixr, [3*k+2 for k in range(p)])

    return vertices, I, J, K

def prepare_mesh():
    frusta_stl = mesh.Mesh.from_file('assets/frustum_small.stl')
    vertices, I, J, K = stl2mesh3d(frusta_stl)
    return I, J, K, vertices

def get_positioned_frustum(vertices, I, J, K, quat, translation, colorscale, origin=(0, 0, 0)):
    triangles = np.vstack((I, J, K)).T

    #rotate
    r_matrix = R.from_quat(quat)
    euler = r_matrix.as_euler('zxy', degrees=True)
    euler[2] += 180         # Accounting for shift in model.
    r_matrix = R.from_euler('zxy', euler, degrees=True)
    r_matrix = r_matrix.as_matrix()

    o = np.atleast_3d(origin)
    p = np.atleast_3d(vertices)
    p_rotated = np.squeeze((r_matrix @ (p.T-o.T) + o.T).T)

    x, y, z = p_rotated.T
    #translate
    x += translation[0]
    y += translation[1] - 0.1
    z += translation[2]

    tri_points = p_rotated[triangles]

    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k % 3][0] for k in range(4)] + [None])
        Ye.extend([T[k % 3][1] for k in range(4)] + [None])
        Ze.extend([T[k % 3][2] for k in range(4)] + [None])

    # define the trace for triangle sides
    lines = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        showlegend=False,
        name='',
        line=dict(color='#222222', width=2))

    frusta_mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=I,
        j=J,
        k=K,
        flatshading=True,
        colorscale=colorscale,
        intensity=z,
        showlegend=False,
        name='Fixation',
        showscale=False)


    return frusta_mesh, lines

def quat_to_euler(quat):
    # assuming q = (x, y, z, w)
    r = R.from_quat(quat)
    euler = r.as_euler('zxy', degrees=True)  # as roll, pitch and yaw

    return euler

def rotate(p, origin=(0, 0, 0), rotation=(0, 0, 0)):
    alpha = np.deg2rad(rotation[0])
    beta = np.deg2rad(rotation[1])
    gamma = np.deg2rad(rotation[2])

    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha),  np.cos(alpha)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)