import matplotlib.pyplot as plt
import matplotlib.tri
from matplotlib.collections import LineCollection
import numpy as np
import numba
import numba.cuda as cuda
from scipy.interpolate import interp1d
from itertools import combinations
from collections import defaultdict
import torch
import math
import random
import openmesh as OM
from meshpy.triangle import MeshInfo, build

def penalty_pos(x, k):
    return torch.clip(-x, min=0) * k


# def resample(bound, idx1=None, idx2=None, N_sample=256, end=False, kind="linear"):
#     if idx1 == None or idx2 == None:
#         bound5 = np.concatenate([bound[-5:, :], bound, bound[:6, :]], axis=0)
#         N = bound.shape[0]
#         bound_ = np.concatenate([bound[1:, :], bound[:1, :]], axis=0)
#         L = np.linalg.norm(bound_ - bound, axis=1)
#         t = np.cumsum(L)
#         t /= t[-1]
#         t = np.concatenate([t[-6:] - 1, t, t[1:6] + 1])
#         sp = interp1d(t, bound5, kind=kind, axis=0)
#         t_sample = np.linspace(0, 1, N_sample + 1)
#         ret = sp(t_sample)
#         return ret[:-1, :]
#     if idx1 < idx2:
#         N = idx2 - idx1 + 1
#         bound_item = bound[idx1:idx2 + 1, :]
#     else:
#         N = bound.shape[0] - idx1 + idx2 + 1
#         bound_item = np.concatenate([bound[idx1:, :], bound[:idx2 + 1, :]], axis=0)
#     t_sample = np.linspace(0, 1, N_sample + 1)
#     t = np.linspace(0, 1, N)
#     sp = interp1d(t, bound_item, kind=kind, axis=0)
#     ret = sp(t_sample)
#     # ret = np.array([np.interp(t_sample, t, bound_item[:, 0]), np.interp(t_sample, t, bound_item[:, 1])]).transpose()
#     if end:
#         return ret
#     return ret[:-1, :]


def get_uv_bound(M, N):
    # M[i,j]=(ih,jh)
    hx, hy = 1 / (M - 1), 1 / (N - 1)
    Mat = np.zeros([M, N, 2])
    for i in range(M):
        Mat[i, :, 0] = i * hx
    for j in range(N):
        Mat[:, j, 1] = j * hy
    return Mat


def get_uv(M, N):
    # M[i,j]=((i+0.5)h,(j+0.5)h)
    hx, hy = 1 / N, 1 / N
    Mat = np.zeros([M, N, 2])
    for i in range(M):
        Mat[i, :, 0] = (i + 0.5) * hx
    for j in range(N):
        Mat[:, j, 1] = (j + 0.5) * hy
    return Mat


def draw_mesh(Mat, bound=None, save=None, color=["y", "g", "k"]):
    M, N, _ = Mat.shape
    for i in range(M):
        plt.plot(Mat[i, :, 0], Mat[i, :, 1], color[0], linewidth=0.3)
    for i in range(N):
        plt.plot(Mat[:, i, 0], Mat[:, i, 1], color[1], linewidth=0.3)
    corner = np.concatenate([Mat[0:1, 0, :], Mat[-1:, 0, :], Mat[-1:, -1, :], Mat[0:1, -1, :]])
    plt.scatter(corner[:, 0], corner[:, 1], c="g", s=20)
    if bound is not None:
        bound2 = np.concatenate([bound, bound[:1, :]], axis=0)
        plt.plot(bound2[:, 0], bound2[:, 1], color[2])
    plt.axis("off")
    if save is not None:
        plt.savefig(save, dpi=200)


def draw_smooth_result(bs, bound=None, scale=1, colored=None, save=None, N_line=20, N_sample=200, line_plot=True):
    if bound is not None:
        plt.plot(bound[:, 0], bound[:, 1], "gray", alpha=0.7)
    if line_plot or (colored is None):
        uv_x = get_uv_bound(N_line, N_sample)
        xy_x = bs(uv_x)
        uv_y = get_uv_bound(N_sample, N_line)
        xy_y = bs(uv_y)
        corners = bs(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
        plt.scatter(corners[:, 0], corners[:, 1], s=200, c="#04f704", zorder=0)
        if colored is None:
            for i in range(N_line):
                plt.plot(xy_x[i, :, 0], xy_x[i, :, 1], "r", linewidth=1, zorder=2, alpha=0.5)
            for i in range(N_line):
                plt.plot(xy_y[:, i, 0], xy_y[:, i, 1], "r", linewidth=1, zorder=2, alpha=0.5)
        else:
            ax = plt.gca()
            Dx_x, Dy_x = bs.D(uv_x)
            Dx_y, Dy_y = bs.D(uv_y)
            Dx_x *= scale
            Dy_x *= scale
            Dx_y *= scale
            Dy_y *= scale
            J_x = Dx_x[:, :, 0] * Dy_x[:, :, 1] - Dx_x[:, :, 1] * Dy_x[:, :, 0]
            J_y = Dx_y[:, :, 0] * Dy_y[:, :, 1] - Dx_y[:, :, 1] * Dy_y[:, :, 0]
            J_x_ = np.clip(J_x, a_min=1e-16, a_max=None)
            J_y_ = np.clip(J_y, a_min=1e-16, a_max=None)
            JF_x = np.sum(Dx_x ** 2 + Dy_x ** 2, axis=2)
            JF_y = np.sum(Dx_y ** 2 + Dy_y ** 2, axis=2)
            if colored == "area":
                draw_x = 1 - 1 / (J_x_ + 1 / J_x_ - 1)
                draw_y = 1 - 1 / (J_y_ + 1 / J_y_ - 1)
            elif colored == "angle":
                draw_x = (JF_x - 2 * J_x_) / (JF_x + 2 * J_x_)
                draw_y = (JF_y - 2 * J_y_) / (JF_y + 2 * J_y_)
            elif colored == "js":
                draw_x = J_x_ / (np.linalg.norm(Dx_x, axis=2) * np.linalg.norm(Dy_x, axis=2))
                draw_y = J_y_ / (np.linalg.norm(Dx_y, axis=2) * np.linalg.norm(Dy_y, axis=2))
            else:
                draw_x = 2 * J_x_ / JF_x
                draw_y = 2 * J_y_ / JF_y
            for i in range(N_line):
                line = xy_x[i, :, :].reshape([-1, 1, 2])
                color = (draw_x[i, :-1] + draw_x[i, 1:]) / 2
                segment = np.concatenate([line[:-1, :, :], line[1:, :, :]], axis=1)
                lc = LineCollection(segment, linewidths=1, cmap="rainbow", norm=plt.Normalize(0, 1), zorder=2)
                lc.set_array(color)
                line = ax.add_collection(lc)
            for i in range(N_line):
                line = xy_y[:, i, :].reshape([-1, 1, 2])
                color = (draw_y[:-1, i] + draw_y[1:, i]) / 2
                segment = np.concatenate([line[:-1, :, :], line[1:, :, :]], axis=1)
                lc = LineCollection(segment, linewidths=1, cmap="rainbow", norm=plt.Normalize(0, 1), zorder=2)
                lc.set_array(color)
                line = ax.add_collection(lc)
            plt.gcf().colorbar(line, ax=ax)
    else:
        uv = get_uv_bound(N_sample, N_sample)
        xy=bs(uv)
        triangles = []
        for i in range(N_sample - 1):
            for j in range(N_sample - 1):
                idx = i * N_sample + j
                triangles.append([idx, idx + 1, idx + 1 + N_sample])
                triangles.append([idx, idx + N_sample + 1, idx + N_sample])
        xy_flatten = xy.reshape([-1, 2])
        triang = matplotlib.tri.Triangulation(xy_flatten[:, 0], xy_flatten[:, 1], triangles=triangles)
        corners = bs(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
        plt.scatter(corners[:, 0], corners[:, 1], s=200, c="#04f704", zorder=0)
        ax = plt.gca()
        Dx, Dy = bs.D(uv)
        Dx *= scale
        Dy *= scale
        J = Dx[:, :, 0] * Dy[:, :, 1] - Dx[:, :, 1] * Dy[:, :, 0]
        J_ = np.clip(J, a_min=1e-16, a_max=None)
        JF = np.sum(Dx ** 2 + Dy ** 2, axis=2)
        if colored == "area":
            draw = 1 - 1 / (J_ + 1 / J_ - 1)
        elif colored == "angle":
            draw = (JF - 2 * J_) / (JF + 2 * J_)
        elif colored == "js":
            draw = J_ / (np.linalg.norm(Dx, axis=2) * np.linalg.norm(Dy, axis=2))
        else:
            draw = 2 * J_ / JF
        tpc=ax.tripcolor(triang,draw.reshape([-1]),shading="gouraud",cmap="rainbow")
        plt.gcf().colorbar(tpc, ax=ax)
    if bound is None:
        try:
            min = np.min(xy_x.reshape([-1, 2]), axis=0)
            max = np.max(xy_x.reshape([-1, 2]), axis=0)
        except:
            min = np.min(xy.reshape([-1, 2]), axis=0)
            max = np.max(xy.reshape([-1, 2]), axis=0)
    else:
        min = np.min(bound, axis=0)
        max = np.max(bound, axis=0)
    plt.axis("equal")
    plt.axis("off")
    plt.xlim([min[0] * 1.02 - max[0] * 0.02, max[0] * 1.02 - min[0] * 0.02])
    plt.ylim([min[1] * 1.02 - max[1] * 0.02, max[1] * 1.02 - min[1] * 0.02])
    if save is not None:
        plt.savefig(save, dpi=200, bbox_inches="tight", pad_inches=0.0)


def draw_tri_mesh(points, edges, save=None):
    axs = plt.gca()
    segments = np.concatenate([points[edges[:, 0]].reshape([-1, 1, 2]), points[edges[:, 1]].reshape([-1, 1, 2])], axis=1)
    if edges.shape[1] == 3:
        segments2 = np.concatenate([points[edges[:, 1]].reshape([-1, 1, 2]), points[edges[:, 2]].reshape([-1, 1, 2])], axis=1)
        segments3 = np.concatenate([points[edges[:, 0]].reshape([-1, 1, 2]), points[edges[:, 2]].reshape([-1, 1, 2])], axis=1)
        segments = np.concatenate([segments, segments2, segments3], axis=0)
    lc = LineCollection(segments, color="k", linewidth=0.3, alpha=0.8)
    axs.add_collection(lc)
    min = np.min(points, axis=0) - 0.1
    max = np.max(points, axis=0) + 0.1
    plt.xlim([min[0], max[0]])
    plt.ylim([min[1], max[1]])
    plt.axis("equal")
    plt.axis("off")
    if save is not None:
        plt.savefig(save, dpi=200, bbox_inches="tight", pad_inches=0.0)




def get_area(bound):
    vec = (bound[1:, :] - bound[0:1, :]).astype(np.float64)
    area = vec[:-1, 0] * vec[1:, 1] - vec[:-1, 1] * vec[1:, 0]
    area = np.sum(area) / 2
    return area


def get_angle(bound):
    vec = np.concatenate([bound[1:, :], bound[:1, :]], axis=0) - bound
    vec_ = np.concatenate([vec[-1:, :], vec[:-1, :]], axis=0)
    L = np.linalg.norm(vec, axis=1)
    L_ = np.concatenate([L[-1:], L[:-1]])
    sin_ang = vec_[:, 0] * vec[:, 1] - vec_[:, 1] * vec[:, 0]
    cos_ang = vec_[:, 0] * vec[:, 0] + vec_[:, 1] * vec[:, 1]
    LL_ = L * L_
    cos_ang = np.clip(cos_ang / LL_, a_min=-1, a_max=1)
    theta = np.arccos(cos_ang) * ((sin_ang > 0) * 2 - 1)
    return theta





def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class POS_AREA_LOSS(torch.autograd.Function):

    @staticmethod
    def forward(self, J, t):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        # t = 1e-2
        N = J.shape[0]
        loss = J.detach().clone()
        Dloss = J.detach().clone()
        threads_per_block = 128
        blocks_per_grid = math.ceil(N / threads_per_block)
        t_1, t_2 = (1 - 1 / (t * t)), 2 / t - 2
        if J.is_cuda:
            POS_AREA_CUDA[blocks_per_grid, threads_per_block](loss, Dloss, t, t_1, t_2)
        else:
            loss, Dloss = loss.numpy(), Dloss.numpy()
            POS_AREA_CPU(loss, Dloss, t, t_1, t_2)
            loss, Dloss = torch.from_numpy(loss), torch.from_numpy(Dloss)
        self.save_for_backward(Dloss)
        return loss

    @staticmethod
    def backward(self, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_outpu
        Dloss, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input.data = grad_input.data * Dloss.data
        return grad_input, None


@cuda.jit
def POS_AREA_CUDA(J, D, t, t_1, t_2):
    pos = cuda.grid(1)
    if pos >= J.shape[0]:
        return
    x = J[pos]
    if x > t:
        J[pos] = x + 1 / x - 2
        D[pos] = 1 - 1 / (x * x)
        return
    J[pos] = t_1 * x + t_2
    D[pos] = t_1
    return


@numba.jit(nopython=True)
def POS_AREA_CPU(J, D, t, t_1, t_2):
    N = J.shape[0]
    for pos in range(N):
        x = J[pos]
        if x > t:
            J[pos] = x + 1 / x - 2
            D[pos] = 1 - 1 / (x * x)
        else:
            J[pos] = t_1 * x + t_2
            D[pos] = t_1


def pos_area_loss(J, t=1e-2):
    shape = J.shape
    J = J.view([-1])
    loss_func = POS_AREA_LOSS()
    loss = loss_func.apply(J, t).view(shape)
    J = J.view(shape)
    return loss



def simplify(bound, t):
    N = bound.shape[0]
    angle = get_angle(bound)
    idx = [0]
    sum = 0
    count = 0
    for i in range(1, N):
        sum += abs(angle[i])
        count += 1
        if sum > t or count >= 50:
            idx.append(i)
            sum = 0
            count = 0
    return bound[idx, :]


def build_tri_mesh(bound, max_area=0.003):
    mesh_info = MeshInfo()
    bound_simplify = simplify(bound, 1e-2)
    mesh_info.set_points(bound_simplify)
    Len_bound = bound_simplify.shape[0]
    r1024 = np.array(list(range(Len_bound))).reshape([-1, 1])
    facet = np.concatenate([r1024, (r1024 + 1) % Len_bound], axis=1)
    mesh_info.set_facets(facet)
    mesh_py = build(mesh_info, max_volume=max_area, min_angle=20)
    points = np.array(list(mesh_py.points)).astype(np.float32)
    Np = points.shape[0]
    points = np.concatenate([points, np.zeros([Np, 1])], axis=1)
    faces = np.array(list(mesh_py.elements))

    # Triangle 库
    # t = triangle.triangulate({'vertices': bound_simplify, 'segments': facet}, 'pelDq20a%.10f' % max_area)
    # points = t["vertices"]
    # Np = points.shape[0]
    # points = np.concatenate([points, np.zeros([Np, 1])], axis=1)
    # faces = t["triangles"]

    mesh = OM.TriMesh()
    mesh.add_vertices(points)
    mesh.add_faces(faces)
    return mesh


def show_loss(record, epoch=1000000000, save=None):
    for key, value in record.items():
        L = min(len(value), epoch)
        plt.plot(list(range(L)), value[:L], label=key, linewidth=1)
    plt.legend()
    if save is not None:
        plt.savefig(save, dpi=300)


def wrap2pi(x):
    y = (x + np.pi) % (2 * np.pi) - np.pi
    return y



def get_trimesh_bound(tri_mesh):
    # 得到的bound_idx逆时针顺序
    is_bound = []
    for v in tri_mesh.vertices():
        is_bound.append(tri_mesh.is_boundary(v))
    is_bound = np.array(is_bound, dtype=np.int32)
    # 边界idx序列
    for he in tri_mesh.halfedges():
        if tri_mesh.is_boundary(he):
            he_start = he
            break
    bound_idx = []
    he_next = he_start
    while True:
        bound_idx.append(tri_mesh.from_vertex_handle(he_next).idx())
        he_next = tri_mesh.next_halfedge_handle(he_next)
        if he_next == he_start:
            break
    bound_idx = np.array(bound_idx)[::-1].copy()
    return bound_idx, is_bound
