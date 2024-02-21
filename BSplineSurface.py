import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, lsmr, lsqr
import numba
from matplotlib import pyplot as plt
import collections
from utils import get_uv, get_uv_bound, draw_mesh, pos_area_loss, penalty_pos
import cpp_accelerate

numba.config.NUMBA_DEFAULT_NUM_THREADS = 20



# @numba.njit()
def N(X, knots, p, require_D=False):
    # knots[idx]<x<knots[idx+1]
    # idx-p ~ idx 基函数非0，共 p+1 个
    # now [idx,idx+p] non-zero
    idx = np.searchsorted(knots, X, side='left') - (p + 1)
    index = idx.reshape((-1, 1)) + np.arange(p * 2 + 1).reshape((1, -1))
    knot = knots[index]
    N, DN = cpp_accelerate.bspline_basis(X.copy(), knot.copy(), p, require_D)

    return N, DN, idx


def Calc(xy, knotx, knoty, coeff, order):
    X, Y = xy[:, 0], xy[:, 1]
    N_X, _, idx_X = N(X, knotx, order - 1)
    N_Y, _, idx_Y = N(Y, knoty, order - 1)
    fxy = cpp_accelerate.bspline_calc(coeff, N_X, N_Y, idx_X, idx_Y)
    return fxy


def CalcD(xy, knotx, knoty, coeff, order):
    X, Y = xy[:, 0], xy[:, 1]
    N_X, DN_X, idx_X = N(X, knotx, order - 1, require_D=True)
    N_Y, DN_Y, idx_Y = N(Y, knoty, order - 1, require_D=True)
    Dx, Dy = cpp_accelerate.bspline_calc_d(coeff, N_X, N_Y, DN_X, DN_Y, idx_X, idx_Y)

    return Dx, Dy


@numba.jit(nopython=True)
def fill_matrix(N_X, idx_X, N_Y, idx_Y, Nu, Nv, smooth=0.0):
    row = []
    col = []
    data = []
    N_points = N_X.shape[0]
    order = N_X.shape[1]
    for p in range(N_points):
        # \sum coeff_ij N_x[i](x)N_y[j](y)=fxy(x,y)
        N_x, N_y, idx_x, idx_y = N_X[p, :], N_Y[p, :], idx_X[p], idx_Y[p]
        for ii in range(0, order):
            for jj in range(0, order):
                i, j = ii + idx_x, jj + idx_y
                idx = i * Nv + j
                row.append(p)
                col.append(idx)
                data.append(N_x[ii] * N_y[jj])
    # Smooth
    N_constraint = 0
    for i in range(1, Nu - 1):
        for j in range(0, Nv):
            # coeff[i,j]-(c[i-1,j]+c[i+1,j])/2=0
            row.append(N_points + N_constraint)
            col.append(i * Nv + j)
            data.append(1 * smooth)
            row.append(N_points + N_constraint)
            col.append((i - 1) * Nv + j)
            data.append(-0.5 * smooth)
            row.append(N_points + N_constraint)
            col.append((i + 1) * Nv + j)
            data.append(-0.5 * smooth)
            N_constraint += 1
    for i in range(0, Nu):
        for j in range(1, Nv - 1):
            # coeff[i,j]-(c[i,j-1]+c[i+1,j+1])/2=0
            row.append(N_points + N_constraint)
            col.append(i * Nv + j)
            data.append(1 * smooth)
            row.append(N_points + N_constraint)
            col.append(i * Nv + j + 1)
            data.append(-0.5 * smooth)
            row.append(N_points + N_constraint)
            col.append(i * Nv + j - 1)
            data.append(-0.5 * smooth)
            N_constraint += 1
    return data, row, col


class BS():
    def __init__(self, Nu=30, Nv=30, order=3, coeff=None):
        # Nu,Nv为控制点个数(样条空间维数)，前后各添加order-1个端点
        self.order = order
        p = order - 1
        if coeff is not None:
            Nu, Nv = coeff.shape[0], coeff.shape[1]
        self.Nu = Nu
        self.Nv = Nv
        knotx = np.concatenate([np.zeros([p]), np.linspace(0, 1, Nu - p + 1), np.ones([p])])
        knoty = np.concatenate([np.zeros([p]), np.linspace(0, 1, Nv - p + 1), np.ones([p])])
        self.p = p
        self.knotx = knotx
        self.knoty = knoty
        if coeff is not None:
            self.coeff = coeff

    def fit(self, xy, fxy, smooth=0.01):
        xy = xy.reshape([-1, 2])
        fxy = fxy.reshape([-1, 2])
        xy = np.clip(xy.astype(np.float64), a_min=0 + 1e-16, a_max=1 - 1e-16)
        fxy = fxy.astype(np.float64)
        X, Y = xy[:, 0], xy[:, 1]
        N_X, _, idx_X = N(X, self.knotx, self.order - 1)
        N_Y, _, idx_Y = N(Y, self.knoty, self.order - 1)
        # data, row, col = fill_matrix(N_X, idx_X, N_Y, idx_Y, self.Nu, self.Nv, smooth)
        data, row, col = cpp_accelerate.bspline_surface_fitting_fill_matrix(
            N_X.copy(), idx_X.copy(), N_Y.copy(), idx_Y.copy(), self.Nu, self.Nv, smooth)
        N_constraint = 2 * (self.Nu * self.Nv - self.Nu - self.Nv)
        N_points = xy.shape[0]
        N_coeff = self.Nu * self.Nv
        M = csr_matrix((data, (row, col)), shape=(N_points + N_constraint, N_coeff), dtype=np.float64)
        b = np.concatenate([fxy, np.zeros([N_constraint, 2])], axis=0)
        coeff_1 = lsqr(M, b[:, 0])[0]
        coeff_2 = lsqr(M, b[:, 1])[0]
        coeff = np.concatenate([coeff_1.reshape([-1, 1]), coeff_2.reshape([-1, 1])], axis=1)
        self.coeff = coeff.reshape([self.Nu, self.Nv, 2])
        # err = np.linalg.norm(self.__call__(xy) - fxy)
        # return err

    def __call__(self, xy):
        shape = xy.shape
        xy = xy.reshape([-1, 2])
        xy = np.clip(xy, a_min=0 + 1e-10, a_max=1 - 1e-10)
        fxy = Calc(xy, self.knotx, self.knoty, self.coeff, self.order)
        return fxy.reshape(shape)

    def D(self, xy):
        shape = xy.shape
        xy = xy.reshape([-1, 2])
        xy = np.clip(xy, a_min=0 + 1e-10, a_max=1 - 1e-10)
        Dx, Dy = CalcD(xy, self.knotx, self.knoty, self.coeff, self.order)
        return Dx.reshape(shape), Dy.reshape(shape)


def N_Torch(X, knots, p, require_D=False):
    # knots[idx]<x<knots[idx+1]
    idx = torch.searchsorted(knots, X, side='left') - 1
    # idx-p ~ idx 基函数非0，共 p+1 个
    idx -= p
    # now [idx,idx+p] non zero
    Np = X.shape[0]
    Mat = torch.zeros((Np, p + 1, p + 1), dtype=X.dtype, device=X.device)
    Mat[:, 0, p] = 1
    # index = idx.reshape((-1, 1)) + np.arange(p * 2 + 1).reshape((1, -1))
    # knot = knots[index]
    knot = torch.empty((Np, 2 * p + 1), dtype=X.dtype, device=X.device)
    for i in range(2 * p + 1):
        knot[:, i] = knots[idx + i]
    for k in range(p):
        for i in range(p):
            Mat[:, k + 1, i] = (X - knot[:, i]) / (knot[:, i + k + 1] - knot[:, i] + 1e-16) * Mat[:, k, i] \
                               + (knot[:, i + k + 2] - X) / (knot[:, i + k + 2] - knot[:, i + 1] + 1e-16) * Mat[:, k, i + 1]
        Mat[:, k + 1, p] = (X - knot[:, p]) / (knot[:, p + k + 1] - knot[:, p] + 1e-10) * Mat[:, k, p]
    if require_D:
        DN = torch.zeros((Np, p + 1), dtype=X.dtype, device=X.device)
        for i in range(p):
            DN[:, i] = p / (knot[:, p + i] - knot[:, i] + 1e-16) * Mat[:, -2, i] \
                       - p / (knot[:, p + i + 1] - knot[:, i + 1] + 1e-16) * Mat[:, -2, i + 1]
        DN[:, p] = p / (knot[:, p + p] - knot[:, p]) * Mat[:, -2, p]
    else:
        DN = None
    return (Mat[:, -1, :]).contiguous(), DN, idx


def Calc_Torch(xy, knotx, knoty, coeff, order):
    fxy = torch.zeros_like(xy)
    X, Y = xy[:, 0], xy[:, 1]
    N_X, _, idx_X = N_Torch(X, knotx, order - 1)
    N_Y, _, idx_Y = N_Torch(Y, knoty, order - 1)
    for ii in range(0, order):
        for jj in range(0, order):
            i, j = ii + idx_X, jj + idx_Y
            fxy += coeff[i, j, :] * (N_X[:, ii] * N_Y[:, jj]).reshape([-1, 1])
    return fxy


def Calc_D_Torch(grad_out, xy, knotx, knoty, coeff, order):
    grad_xy = torch.zeros_like(xy)
    grad_coeff = torch.zeros_like(coeff)
    X, Y = xy[:, 0], xy[:, 1]
    N_X, DN_X, idx_X = N_Torch(X, knotx, order - 1, require_D=True)
    N_Y, DN_Y, idx_Y = N_Torch(Y, knoty, order - 1, require_D=True)
    Nu, Nv, _ = coeff.shape
    for ii in range(0, order):
        for jj in range(0, order):
            i, j = ii + idx_X, jj + idx_Y
            temp = (grad_out * coeff[i, j, :]).sum(1)
            grad_xy[:, 0] += temp * DN_X[:, ii] * N_Y[:, jj]
            grad_xy[:, 1] += temp * N_X[:, ii] * DN_Y[:, jj]
            # grad_coeff[i, j, :] += grad_out * (N_X[:, ii] * N_Y[:, jj]).reshape([-1, 1])
            temp = grad_out * (N_X[:, ii] * N_Y[:, jj]).reshape([-1, 1])
            index_flatten = (i * Nv + j) * 2
            grad_coeff = grad_coeff.reshape([-1])
            grad_coeff.data.scatter_add_(0, index_flatten, temp[:, 0])
            grad_coeff.data.scatter_add_(0, index_flatten + 1, temp[:, 1])
            grad_coeff = grad_coeff.reshape(coeff.shape)
            # np.add.at(grad_coeff.reshape([-1]), index_flatten, temp[:, 0])
            # np.add.at(grad_coeff.reshape([-1]), index_flatten + 1, temp[:, 1])
    return grad_xy, grad_coeff


class BS_Func_Torch_(torch.autograd.Function):
    @staticmethod
    def forward(self, xy, knotx, knoty, coeff, order):
        fxy = Calc_Torch(xy, knotx, knoty, coeff, order)
        self.save_for_backward(xy, knotx, knoty, coeff, order)

        return fxy

    @staticmethod
    def backward(self, grad_output):
        xy, knotx, knoty, coeff, order = self.saved_tensors
        grad_xy, grad_coeff = Calc_D_Torch(grad_output, xy, knotx, knoty, coeff, order)
        return grad_xy, None, None, grad_coeff, None


def BS_Func_Torch(xy, knotx, knoty, coeff, order):
    xy = torch.clip(xy, min=0 + 1e-16, max=1 - 1e-16)
    fxy = BS_Func_Torch_.apply(xy, knotx, knoty, coeff, torch.tensor(order, device=xy.device))
    return fxy


class BS_Torch():
    def __init__(self, Nu=30, Nv=30, order=3, coeff=None, device=torch.device("cpu"), dtype=torch.float64):
        # Nu,Nv为控制点个数(样条空间维数)，前后各添加order-1个端点
        self.order = order
        p = order - 1
        if coeff is not None:
            Nu, Nv = coeff.shape[0], coeff.shape[1]
            dtype = coeff.dtype
            device = coeff.device
        self.device = device
        self.Nu = Nu
        self.Nv = Nv
        knotx = torch.from_numpy(np.concatenate([np.zeros([p]), np.linspace(0, 1, Nu - p + 1), np.ones([p])])).type(dtype).to(device)
        knoty = torch.from_numpy(np.concatenate([np.zeros([p]), np.linspace(0, 1, Nv - p + 1), np.ones([p])])).type(dtype).to(device)
        self.p = p
        self.knotx = knotx
        self.knoty = knoty
        if coeff is None:
            coeff = torch.from_numpy(get_uv_bound(Nu, Nv)).type(dtype).to(device)
        self.coeff = torch.nn.Parameter(coeff, requires_grad=True)
        self.dtype = dtype

    def BijectiveFitting(self, fix_xy, fix_fxy_real, N_optim=100, max_iter=100, min_iter=5):
        optim = torch.optim.Adam(params=[self.coeff], lr=0.001)
        fix_xy = torch.clip(fix_xy, min=0 + 1e-16, max=1 - 1e-16)
        sample_xy = torch.from_numpy(get_uv_bound(N_optim + 1, N_optim + 1)).type(self.coeff.dtype).to(self.device)
        sample_xy = torch.clip(sample_xy, min=0 + 1e-16, max=1 - 1e-16)
        best = {"E": 1e10, "coeff": self.coeff.detach().clone(), "step": 0}
        bij = False
        extra_steps = 0
        for iter in range(max_iter):
            N_fix = fix_xy.shape[0]
            inputs = torch.cat([fix_xy, sample_xy.view([-1, 2])], dim=0)
            outputs = self(inputs)
            fix_fxy = outputs[:N_fix, :]
            sample_fxy = outputs[N_fix:, :].view(sample_xy.shape)

            E_constraint = torch.sum((fix_fxy - fix_fxy_real) ** 2)

            smooth = sample_fxy[1:-1, 1:-1, :] - \
                     (sample_fxy[:-2, :-2, :] + sample_fxy[2:, :-2, :] + sample_fxy[2:, 2:, :] + sample_fxy[:-2, 2:, :]) / 8 - \
                     (sample_fxy[1:-1, :-2, :] + sample_fxy[2:, 1:-1, :] + sample_fxy[1:-1, 2:, :] + sample_fxy[:-2, 1:-1, :]) / 8
            smooth = smooth.view([-1, 2]) * N_optim
            E_smooth = torch.mean(smooth ** 2)

            dx = (sample_fxy[1:, :, :] - sample_fxy[:-1, :, :]) * N_optim
            dx_p = dx[:, :-1, :]
            dx_n = dx[:, 1:, :]
            dy = (sample_fxy[:, 1:, :] - sample_fxy[:, :-1, :]) * N_optim
            dy_p = dy[:-1, :, :]
            dy_n = dy[1:, :, :]
            dx_p, dx_n, dy_p, dy_n = dx_p.reshape([-1, 2]), dx_n.reshape([-1, 2]), dy_p.reshape([-1, 2]), dy_n.reshape([-1, 2])
            dx = torch.cat([dx_p, dx_n], dim=0)
            dy = torch.cat([dy_p, dy_n], dim=0)
            J = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]
            E_bij = torch.sum(penalty_pos(J - 1e-2, 1))
            J_Fro = torch.sum(dx ** 2 + dy ** 2, dim=1)
            # loss_inv = torch.sum(penalty_pos(J, 100))
            E_area = torch.mean(pos_area_loss(J, t=1e-2))
            E_angle = torch.mean((J_Fro - 2 * J) / (J_Fro + 2 * J))
            E_inner = E_angle * 4 + E_area + E_smooth * 100
            E = E_bij * 1e3 + E_constraint * 1e4 + E_inner
            if not bij and E_bij < 1e-6:
                bij = True
                optim.state = collections.defaultdict(dict)
            if E < best["E"]:
                best["E"] = E
                best["coeff"] = self.coeff.detach().clone()
                best["step"] = iter
                print("\tSTEP = %5d, E = %.06f, E_bij = %.06f, E_cons = %.08f, E_inner = %.06f" %
                      (iter, E.item(), E_bij.item(), E_constraint.item(), E_inner.item()))
            if E_bij < 1e-6 and E_constraint < 1e-3:
                if extra_steps > min_iter:
                    break
                else:
                    extra_steps += 1
            if iter - best["step"] > 50:
                break
            optim.zero_grad()
            E.backward()
            optim.step()
        return bij

    def __call__(self, xy):
        shape = xy.shape
        xy = xy.reshape([-1, 2])
        fxy = BS_Func_Torch(xy, self.knotx, self.knoty, self.coeff, self.order)
        return fxy.reshape(shape)


if __name__ == "__main__":
    uv = get_uv(5, 5)
    spline = BS(30, 30, 3)
    spline.fit(uv, uv, 0.01)
    device = torch.device("cuda")
    spline_torch = BS_Torch(coeff=torch.from_numpy(spline.coeff.astype(np.float32)), device=device)
    uv_sample = torch.from_numpy(get_uv_bound(10, 10)).to(device)
    xy_sample = spline_torch(uv_sample)
    draw_mesh(xy_sample.detach().cpu().numpy())
    plt.show()
