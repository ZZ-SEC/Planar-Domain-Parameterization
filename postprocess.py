import time
import numba
import numpy as np
import torch
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, lsqr
import cpp_accelerate
from utils import get_angle, get_uv_bound
from BSplineSurface import BS, BS_Torch
from collections import defaultdict


def get_range(I, N):
    a = np.min(I)
    b = np.max(I)
    if b - a < N / 2:
        return np.array(range(a, b + 1))
    else:
        a = np.max(I[np.argwhere(I < N / 2)[:, 0]])
        b = np.min(I[np.argwhere(I > N / 2)[:, 0]])
        return np.concatenate([np.array(range(0, a + 1)), np.array(range(b, N))])


def GetNewBound(bound, corner_idx=None, corner_choose_range=10):
    N = bound.shape[0]
    if corner_idx is None:
        dis_00 = np.linalg.norm(bound, axis=1)
        dis_10 = np.linalg.norm(bound - np.array([[1, 0]]), axis=1)
        dis_11 = np.linalg.norm(bound - np.array([[1, 1]]), axis=1)
        dis_01 = np.linalg.norm(bound - np.array([[0, 1]]), axis=1)
        corner_00 = np.argsort(dis_00)[:corner_choose_range]
        corner_10 = np.argsort(dis_10)[:corner_choose_range]
        corner_11 = np.argsort(dis_11)[:corner_choose_range]
        corner_01 = np.argsort(dis_01)[:corner_choose_range]
        corner_00 = get_range(corner_00, N)
        corner_10 = get_range(corner_10, N)
        corner_11 = get_range(corner_11, N)
        corner_01 = get_range(corner_01, N)
        angles = get_angle(bound)
        c00 = corner_00[np.argmax(angles[corner_00])]
        c10 = corner_10[np.argmax(angles[corner_10])]
        c11 = corner_11[np.argmax(angles[corner_11])]
        c01 = corner_01[np.argmax(angles[corner_01])]
    else:
        c00, c10, c11, c01 = corner_idx
    bound_new = np.zeros_like(bound)
    extend = 1e-7
    # [0,0]-[1,0]
    if c10 > c00:
        line = bound[c00:c10 + 1, :]
        len_line = np.linalg.norm(line[1:, :] - line[:-1, :], axis=1)
        len_line /= np.sum(len_line)
        len_line = np.cumsum(np.concatenate([np.zeros([1]), len_line])) * (1 + 2 * extend) - extend
        line_new = np.zeros_like(line)
        line_new[:, 1] = -extend
        line_new[:, 0] = len_line
        bound_new[c00:c10 + 1, :] = line_new
    else:
        line = np.concatenate([bound[c00:, :], bound[:c10 + 1, :]])
        len_line = np.linalg.norm(line[1:, :] - line[:-1, :], axis=1)
        len_line /= np.sum(len_line)
        len_line = np.cumsum(np.concatenate([np.zeros([1]), len_line])) * (1 + 2 * extend) - extend
        line_new = np.zeros_like(line)
        line_new[:, 1] = -extend
        line_new[:, 0] = len_line
        bound_new[c00:, :] = line_new[:N - c00, :]
        bound_new[:c10 + 1, :] = line_new[N - c00:, :]
    # [1,0]-[1,1]
    if c11 > c10:
        line = bound[c10:c11 + 1, :]
        len_line = np.linalg.norm(line[1:, :] - line[:-1, :], axis=1)
        len_line /= np.sum(len_line)
        len_line = np.cumsum(np.concatenate([np.zeros([1]), len_line])) * (1 + 2 * extend) - extend
        line_new = np.ones_like(line)
        line_new[:, 0] = 1 + extend
        line_new[:, 1] = len_line
        bound_new[c10:c11 + 1, :] = line_new
    else:
        line = np.concatenate([bound[c10:, :], bound[:c11 + 1, :]])
        len_line = np.linalg.norm(line[1:, :] - line[:-1, :], axis=1)
        len_line /= np.sum(len_line)
        len_line = np.cumsum(np.concatenate([np.zeros([1]), len_line])) * (1 + 2 * extend) - extend
        line_new = np.ones_like(line)
        line_new[:, 0] = 1 + extend
        line_new[:, 1] = len_line
        bound_new[c10:, :] = line_new[:N - c10, :]
        bound_new[:c11 + 1, :] = line_new[N - c10:, :]
    # [1,1]-[0,1]
    if c01 > c11:
        line = bound[c11:c01 + 1, :]
        len_line = np.linalg.norm(line[1:, :] - line[:-1, :], axis=1)
        len_line /= np.sum(len_line)
        len_line = np.cumsum(np.concatenate([np.zeros([1]), len_line[::-1]])) * (1 + 2 * extend) - extend
        line_new = np.ones_like(line)
        line_new[:, 1] = 1 + extend
        line_new[:, 0] = len_line[::-1]
        bound_new[c11:c01 + 1, :] = line_new
    else:
        line = np.concatenate([bound[c11:, :], bound[:c01 + 1, :]])
        len_line = np.linalg.norm(line[1:, :] - line[:-1, :], axis=1)
        len_line /= np.sum(len_line)
        len_line = np.cumsum(np.concatenate([np.zeros([1]), len_line[::-1]])) * (1 + 2 * extend) - extend
        line_new = np.ones_like(line)
        line_new[:, 1] = 1 + extend
        line_new[:, 0] = len_line[::-1]
        bound_new[c11:, :] = line_new[:N - c11, :]
        bound_new[:c01 + 1, :] = line_new[N - c11:, :]
    # [0,1]-[0,0]
    if c00 > c01:
        line = bound[c01:c00 + 1, :]
        len_line = np.linalg.norm(line[1:, :] - line[:-1, :], axis=1)
        len_line /= np.sum(len_line)
        len_line = np.cumsum(np.concatenate([np.zeros([1]), len_line[::-1]])) * (1 + 2 * extend) - extend
        line_new = np.zeros_like(line)
        line_new[:, 0] = - extend
        line_new[:, 1] = len_line[::-1]
        bound_new[c01:c00 + 1, :] = line_new
    else:
        line = np.concatenate([bound[c01:, :], bound[:c00 + 1, :]])
        len_line = np.linalg.norm(line[1:, :] - line[:-1, :], axis=1)
        len_line /= np.sum(len_line)
        len_line = np.cumsum(np.concatenate([np.zeros([1]), len_line[::-1]])) * (1 + 2 * extend) - extend
        line_new = np.zeros_like(line)
        line_new[:, 0] = - extend
        line_new[:, 1] = len_line[::-1]
        bound_new[c01:, :] = line_new[:N - c01, :]
        bound_new[:c00 + 1, :] = line_new[N - c01:, :]

    return bound_new


@numba.njit()
def CalcMVC(C, V):
    # C: center point [1x2] , V: neighbors [N*2]
    vec = V - C
    len_vec = np.sqrt(vec[:, 0] ** 2 + vec[:, 1] ** 2)  # np.linalg.norm(vec, axis=1)
    vec_n = vec / len_vec.reshape((-1, 1))
    cos_ang = vec_n[:, 0]
    sin_ang = vec_n[:, 1]
    cos_ang = np.clip(cos_ang, a_min=-1, a_max=1)
    theta = np.arccos(cos_ang) * ((sin_ang > 0) * 2 - 1)
    alpha = np.concatenate((theta[1:], theta[:1])) - theta
    alpha = alpha % (np.pi * 2)
    tan_alpha2 = np.tan(alpha / 2)
    w = (np.concatenate((tan_alpha2[-1:], tan_alpha2[:-1])) + tan_alpha2) / len_vec
    w /= np.sum(w)
    return w


@numba.njit()
def PostProcess_FillMatrix(points_uv, new_bound, bound_idx, is_bound, neighbors):
    N_points = points_uv.shape[0]
    b = np.zeros((N_points, 2))
    row = []
    col = []
    data = []
    for i in range(bound_idx.shape[0]):
        idx = bound_idx[i]
        b[idx, :] = new_bound[i, :]
    for i in range(N_points):
        row.append(i)
        col.append(i)
        data.append(1)
        if is_bound[i]:
            continue
        neighbor = neighbors[i, :]
        N_neighbor = np.sum(neighbor >= 0)
        neighbor = neighbor[:N_neighbor]
        mvc = CalcMVC(points_uv[i:i + 1, :], points_uv[neighbor, :])
        if mvc.min() < 0:
            mvc[:] = 1
            mvc = mvc / mvc.sum()
        for j in range(N_neighbor):
            row.append(i)
            col.append(neighbors[i][j])
            data.append(-mvc[j])
    return data, row, col, b


def PostProcess(tri_mesh, points_uv_, pre, max_iter=10, device=torch.device("cpu")):
    points_xy = tri_mesh.points()[:, :2].astype(np.float64)
    points_uv = points_uv_.astype(np.float64)
    neighbors = tri_mesh.vertex_vertex_indices()
    triangles = tri_mesh.face_vertex_indices()
    bound_idx, is_bound = cpp_accelerate.get_trimesh_bound(neighbors)
    bound_idx = bound_idx[::-1].copy()
    # bound_idx, is_bound = get_trimesh_bound(tri_mesh)
    new_bound = GetNewBound(points_uv[bound_idx, :], corner_choose_range=15)
    N_points = points_uv.shape[0]
    data, row, col, b = cpp_accelerate.post_process_fill_matrix(points_uv, new_bound, bound_idx, is_bound, neighbors)
    # data, row, col, b = PostProcess_FillMatrix(points_uv, new_bound, bound_idx, is_bound, neighbors)
    M = csr_matrix((data, (row, col)), shape=(N_points, N_points), dtype=np.float64)
    points_square = spsolve(M, b).copy()
    N_sample = 100
    sample_uv = get_uv_bound(N_sample, N_sample)
    sample_xy = cpp_accelerate.trimesh_sample(N_sample, points_square, points_xy, triangles)
    sample_xy = EliminateFoldovers(sample_xy, weight=1)
    bs = BS(Nu=50, Nv=50, order=3)
    sample_xy = pre.backward(sample_xy)
    bs.fit(sample_uv, sample_xy, smooth=0.001)
    N_optim = 200
    uv_optim = get_uv_bound(N_optim, N_optim)
    Dx, Dy = bs.D(uv_optim)
    J = Dx[:, :, 0] * Dy[:, :, 1] - Dx[:, :, 1] * Dy[:, :, 0]
    if J.min() < 1e-3:
        bs_torch = BS_Torch(coeff=torch.from_numpy(bs.coeff).to(device), order=3)
        temp = np.arange(N_sample - 1)
        index_bound = np.concatenate([temp, temp + N_sample * (N_sample - 1), temp * N_sample, temp * N_sample + N_sample - 1])
        fix_xy = (sample_uv.reshape([-1, 2])[index_bound, :]).copy()
        fix_fxy = (sample_xy.reshape([-1, 2])[index_bound, :]).copy()
        bij = bs_torch.BijectiveFitting(fix_xy=torch.from_numpy(fix_xy).to(device), fix_fxy_real=torch.from_numpy(fix_fxy).to(device),
                                        N_optim=N_optim, max_iter=max_iter, min_iter=10)
        return BS(coeff=bs_torch.coeff.detach().cpu().numpy(), order=3), bij
    return bs, True


@numba.jit(nopython=True)
def TriMeshSample(N, points_square, points, triangles):
    xy = np.zeros((N, N, 2))
    h = 1 / (N - 1)
    N_triangles = triangles.shape[0]
    mark = -np.ones((N, N)) * 1e10
    for t in range(N_triangles):
        tri_idx = triangles[t]
        tri = points_square[tri_idx, :] / h
        tri_xy = points[tri_idx, :]
        v1 = tri[1, :] - tri[0, :]
        v2 = tri[2, :] - tri[0, :]
        if v1[0] * v2[1] - v1[1] * v2[0] <= 0:
            continue
        M = np.concatenate((v1.reshape((2, 1)), v2.reshape((2, 1))), axis=1)
        M_inv = np.linalg.inv(M)
        tri_min = np.clip(np.ceil(np.array((np.min(tri[:, 0]), np.min(tri[:, 1]))) - 1e-3).astype(np.int32), a_min=0, a_max=N - 1)
        tri_max = np.clip(np.floor(np.array((np.max(tri[:, 0]), np.max(tri[:, 1]))) + 1e-3).astype(np.int32), a_min=0, a_max=N - 1)
        for x in range(tri_min[0], tri_max[0] + 1):
            for y in range(tri_min[1], tri_max[1] + 1):
                p = np.array([x - tri[0, 0], y - tri[0, 1]])
                coord = np.empty((3))
                coord[1], coord[2] = M_inv[0, 0] * p[0] + M_inv[0, 1] * p[1], M_inv[1, 0] * p[0] + M_inv[1, 1] * p[1]
                coord[0] = 1 - coord[1] - coord[2]
                min_coord = np.min(coord)
                if mark[x, y] > min_coord or min_coord < -0.1:
                    continue
                mark[x, y] = min_coord
                xy[x, y, :] = np.sum(coord.reshape((3, 1)) * tri_xy, axis=0)
    return xy


class EliminateFoldovers_Mdict:
    dict = defaultdict(lambda: None)


def EliminateFoldovers(P, weight=0.1):
    N = P.shape[0]
    opt = "%10d+%.10f" % (N, weight)
    M = EliminateFoldovers_Mdict.dict[opt]
    if M is None:
        mat, shape = cpp_accelerate.foldover_fill_matrix(N, weight)
        # mat, shape = EliminateFoldovers_FillMatrix(N, weight)
        M = csr_matrix((mat[0], (mat[1], mat[2])), shape=shape, dtype=np.float64)
        EliminateFoldovers_Mdict.dict[opt] = M
    b = np.zeros([M.shape[0], 2])
    b[:M.shape[1], :] = P.reshape([-1, 2])
    P_1 = lsqr(M, b[:, 0])[0]
    P_2 = lsqr(M, b[:, 1])[0]
    P_ = np.concatenate([P_1.reshape([-1, 1]), P_2.reshape([-1, 1])], axis=1)
    # MT = M.transpose()
    # P_ = spsolve(MT.dot(M), MT.dot(b))
    return P_.reshape(P.shape)
