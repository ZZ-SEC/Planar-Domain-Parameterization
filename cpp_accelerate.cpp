#define npf py::array_t<float>
#define npd py::array_t<double>
#define npi32 py::array_t<__int32>
#define npi64 py::array_t<__int64>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

double clamp(const double& value, const double& lower, const double& upper) {
    return (value < lower) ? lower : (value > upper ? upper : value);
}

namespace py = pybind11;

py::tuple post_fill(npd points_uv_, npd new_bound_, npi32 bound_idx_, npi32 is_bound_, npi32 neighbors_) {
    auto points_uv = (double*)points_uv_.request().ptr, new_bound = (double*)new_bound_.request().ptr;
    auto bound_idx = (__int32*)bound_idx_.request().ptr, is_bound = (__int32*)is_bound_.request().ptr, neighbors = (__int32*)neighbors_.request().ptr;
    __int32 N_points = points_uv_.request().shape[0];
    __int32 neighbor_width = neighbors_.shape()[1];
    __int32* N_neighbors = new __int32[N_points];
    // Count non-zero elements
    __int32 N_elements = N_points;
    for (__int32 i = 0; i < N_points; i++) {
        __int32* neighbor = neighbors + i * neighbor_width;
        __int32 N_neighbor = 0;
        for (; neighbor[N_neighbor] >= 0 && N_neighbor < neighbor_width; N_neighbor++);
        N_neighbors[i] = N_neighbor;
        if (!is_bound[i]) {
            N_elements += N_neighbor;
        }
    }
    __int32* row = new __int32[N_elements], * col = new __int32[N_elements];
    double* data = new double[N_elements];
    double* b = new double[N_points * 2]();
    for (__int32 i = 0; i < bound_idx_.size(); i++) {
        __int32 idx = bound_idx[i];
        b[idx * 2] = new_bound[i * 2];
        b[idx * 2 + 1] = new_bound[i * 2 + 1];
    }
    double* len = new double[neighbor_width];
    double* theta = new double[neighbor_width + 1];
    double* tan_alpha2 = new double[neighbor_width + 1];
    double* mvc = new double[neighbor_width];
    __int32 idx_element = 0;
    for (__int32 i = 0; i < N_points; i++) {
        row[idx_element] = i;
        col[idx_element] = i;
        data[idx_element] = 1.0;
        idx_element++;
        if (is_bound[i]) {
            continue;
        }
        __int32* neighbor = neighbors + i * neighbor_width;
        // Calculate MVC
        double x0 = points_uv[i * 2], y0 = points_uv[i * 2 + 1];
        for (__int32 j = 0; j < N_neighbors[i]; j++) {
            double x = points_uv[neighbor[j] * 2] - x0, y = points_uv[neighbor[j] * 2 + 1] - y0;
            len[j] = sqrt(x * x + y * y);
            double cos_theta = clamp(x / len[j], -1.0 + 1e-20, 1.0 - 1e-20);
            theta[j] = acos(cos_theta);
            if (y < 0)
                theta[j] = -theta[j];
        }
        theta[N_neighbors[i]] = theta[0];
        for (__int32 j = 0; j < N_neighbors[i]; j++) {
            tan_alpha2[j + 1] = tan((theta[j + 1] - theta[j]) / 2.0);
        }
        tan_alpha2[0] = tan_alpha2[N_neighbors[i]];
        double mvc_sum = 0, mvc_min = 1e10;
        for (__int32 j = 0; j < N_neighbors[i]; j++) {
            mvc[j] = (tan_alpha2[j] + tan_alpha2[j + 1]) / len[j];
            mvc_sum += mvc[j];
            if (mvc[j] < mvc_min)
                mvc_min = mvc[j];
        }
        if (mvc_min < 0)
            for (__int32 j = 0; j < N_neighbors[i]; j++) {
                mvc[j] /= mvc_sum;
            }
        else
            for (__int32 j = 0; j < N_neighbors[i]; j++) {
                mvc[j] = 1.0 / N_neighbors[i];
            }
        for (__int32 j = 0; j < N_neighbors[i]; j++) {
            row[idx_element] = i;
            col[idx_element] = neighbor[j];
            data[idx_element] = -mvc[j];
            idx_element++;
        }
    }
    delete[] len;
    delete[] theta;
    delete[] tan_alpha2;
    delete[] mvc;
    delete[] N_neighbors;
    npi32 row_(N_elements, row), col_(N_elements, col);
    npd data_(N_elements, data), b_({ N_points,2 }, b);
    delete[] row;
    delete[] col;
    delete[] data;
    delete[] b;
    return py::make_tuple(data_, row_, col_, b_);
    /*double* ptr1 = (double*)buf1.ptr, * ptr2 = (double*)buf2.ptr, * ptr3 = (double*)buf3.ptr;
    for (__int32 i = 0; i < buf1.size; i++) {
        ptr3[i] = ptr1[i] + ptr2[i];
    }
    return c;*/
}

py::tuple get_trimesh_bound(npi32 neighbors_) {
    auto neighbors = (__int32*)neighbors_.request().ptr;
    __int32 N_points = neighbors_.shape()[0];
    __int32 neighbor_width = neighbors_.shape()[1];
    __int32 first_bound_idx = 0, last_bound_idx = 0;
    for (__int32 i = 0; i < N_points; i++) {
        __int32* neighbor = neighbors + i * neighbor_width;
        __int32 N_neighbor = 0;
        for (; neighbor[N_neighbor] >= 0 && N_neighbor < neighbor_width; N_neighbor++);
        __int32 idx_f = neighbor[0], idx_l = neighbor[N_neighbor - 1];
        auto neighbor_f = neighbors + idx_f * neighbor_width;
        __int32 j = 0;
        for (; j < neighbor_width && neighbor_f[j] >= 0 && neighbor_f[j] != idx_l; j++);
        if (j == neighbor_width) {
            first_bound_idx = i;
            break;
        }
    }
    std::vector<__int32> bound_idx;
    bound_idx.reserve(100);
    bound_idx.push_back(first_bound_idx);
    while (true) {
        last_bound_idx = bound_idx.back();
        __int32* neighbor = neighbors + last_bound_idx * neighbor_width;
        __int32 next_idx = neighbor[0];
        if (next_idx == first_bound_idx)
            break;
        bound_idx.push_back(next_idx);
    }
    __int32 N_bound = bound_idx.size();
    __int32* is_bound = new __int32[N_points]();
    for (__int32 i = 0; i < N_bound; i++) {
        is_bound[bound_idx[i]] = 1;
    }
    npi32 bound_idx_(N_bound, &(bound_idx[0]));
    npi32 is_bound_(N_points, is_bound);
    delete[] is_bound;
    return py::make_tuple(bound_idx_, is_bound_);
}

template<typename T> T tri_min(T a, T b, T c) {
    return std::min(std::min(a, b), c);
}
template<typename T> T tri_max(T a, T b, T c) {
    return std::max(std::max(a, b), c);
}

npd trimesh_sample(const int N, npd points_square_, npd points_, npi32 triangles_) {
    __int32 N_points = points_.shape()[0], N_triangles = triangles_.shape()[0];
    auto points_square = (double*)points_square_.request().ptr, points = (double*)points_.request().ptr;
    auto triangles = (__int32*)triangles_.request().ptr;
    double h = 1.0 / (N - 1);
    double* xy = new double[N * N * 2]();
    double* mark = new double[N * N];
    double* points_uv = new double[N_points * 2];
    double tri_uv[3][2], M[2][2], M_inv[2][2];
    std::pair<double, double> v1, v2;
    for (int i = 0; i < N * N; i++)mark[i] = -1e10;
    for (int i = 0; i < N_points * 2; i++) {
        points_uv[i] = points_square[i] / h;
    }
    for (int t = 0; t < N_triangles; t++) {
        auto triangle = triangles + t * 3;
        tri_uv[0][0] = points_uv[triangle[0] * 2]; tri_uv[0][1] = points_uv[triangle[0] * 2 + 1];
        tri_uv[1][0] = points_uv[triangle[1] * 2]; tri_uv[1][1] = points_uv[triangle[1] * 2 + 1];
        tri_uv[2][0] = points_uv[triangle[2] * 2]; tri_uv[2][1] = points_uv[triangle[2] * 2 + 1];
        M[0][0] = tri_uv[1][0] - tri_uv[0][0]; M[0][1] = tri_uv[2][0] - tri_uv[0][0];
        M[1][0] = tri_uv[1][1] - tri_uv[0][1]; M[1][1] = tri_uv[2][1] - tri_uv[0][1];
        double detM = M[0][0] * M[1][1] - M[1][0] * M[0][1];
        if (detM <= 0)
            continue;
        M_inv[0][0] = M[1][1] / detM; M_inv[0][1] = -M[0][1] / detM;
        M_inv[1][0] = -M[1][0] / detM; M_inv[1][1] = M[0][0] / detM;
        int u_min = (int)ceil(tri_min(tri_uv[0][0], tri_uv[1][0], tri_uv[2][0]) - 1e-2);
        int v_min = (int)ceil(tri_min(tri_uv[0][1], tri_uv[1][1], tri_uv[2][1]) - 1e-2);
        int u_max = (int)floor(tri_max(tri_uv[0][0], tri_uv[1][0], tri_uv[2][0]) + 1e-2);
        int v_max = (int)floor(tri_max(tri_uv[0][1], tri_uv[1][1], tri_uv[2][1]) + 1e-2);
        for (int u = u_min; u <= u_max; u++) {
            for (int v = v_min; v <= v_max; v++) {
                double vu = u - tri_uv[0][0], vv = v - tri_uv[0][1];
                double coo1 = M_inv[0][0] * vu + M_inv[0][1] * vv, coo2 = M_inv[1][0] * vu + M_inv[1][1] * vv;
                double coo0 = 1.0 - coo1 - coo2;
                double min_coo = tri_min(coo0, coo1, coo2);
                int idx = u * N + v;
                if ((min_coo < -0.1) || (mark[idx] > min_coo))
                    continue;
                mark[idx] = min_coo;
                xy[idx * 2] = coo0 * points[triangle[0] * 2] + coo1 * points[triangle[1] * 2] + coo2 * points[triangle[2] * 2];
                xy[idx * 2 + 1] = coo0 * points[triangle[0] * 2 + 1] + coo1 * points[triangle[1] * 2 + 1] + coo2 * points[triangle[2] * 2 + 1];
            }
        }
    }
    npd xy_({ N,N,2 }, xy);
    delete[] xy;
    delete[] mark;
    delete[] points_uv;
    return xy_;
}

py::tuple fold_over_fill(const int N, const double weight = 0.1) {
    int N_cons = 0;
    int N_elem = 0;
    auto row = new int[N * N * 7], col = new int[N * N * 7];
    auto data = new double[N * N * 7];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            row[N_elem] = idx;
            col[N_elem] = idx;
            data[N_elem] = 1.0;
            N_elem++;
            N_cons++;
        }
    for (int i = 0; i < N; i++)
        for (int j = 1; j < N - 1; j++) {
            int idx = i * N + j;
            row[N_elem] = N_cons;
            col[N_elem] = idx;
            data[N_elem] = weight;
            N_elem++;
            row[N_elem] = N_cons;
            col[N_elem] = idx - 1;
            data[N_elem] = -0.5 * weight;
            N_elem++;
            row[N_elem] = N_cons;
            col[N_elem] = idx + 1;
            data[N_elem] = -0.5 * weight;
            N_elem++;
            N_cons++;
        }
    for (int i = 1; i < N - 1; i++)
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            row[N_elem] = N_cons;
            col[N_elem] = idx;
            data[N_elem] = weight;
            N_elem++;
            row[N_elem] = N_cons;
            col[N_elem] = idx - N;
            data[N_elem] = -0.5 * weight;
            N_elem++;
            row[N_elem] = N_cons;
            col[N_elem] = idx + N;
            data[N_elem] = -0.5 * weight;
            N_elem++;
            N_cons++;
        }
    //
    row[N_elem] = N_cons;
    col[N_elem] = 0;
    data[N_elem] = 10 * weight;
    N_elem++;
    row[N_elem] = N_cons;
    col[N_elem] = 1;
    data[N_elem] = -10 * weight;
    N_elem++;
    row[N_elem] = N_cons;
    col[N_elem] = N;
    data[N_elem] = -10 * weight;
    N_elem++;
    row[N_elem] = N_cons;
    col[N_elem] = N + 1;
    data[N_elem] = 10 * weight;
    N_elem++;
    N_cons++;
    //
    row[N_elem] = N_cons;
    col[N_elem] = N - 1;
    data[N_elem] = 10 * weight;
    N_elem++;
    row[N_elem] = N_cons;
    col[N_elem] = N - 2;
    data[N_elem] = -10 * weight;
    N_elem++;
    row[N_elem] = N_cons;
    col[N_elem] = 2 * N - 1;
    data[N_elem] = -10 * weight;
    N_elem++;
    row[N_elem] = N_cons;
    col[N_elem] = 2 * N - 2;
    data[N_elem] = 10 * weight;
    N_elem++;
    N_cons++;
    //
    row[N_elem] = N_cons;
    col[N_elem] = N * N - 1;
    data[N_elem] = 10 * weight;
    N_elem++;
    row[N_elem] = N_cons;
    col[N_elem] = N * N - 2;
    data[N_elem] = -10 * weight;
    N_elem++;
    row[N_elem] = N_cons;
    col[N_elem] = N * (N - 1) - 1;
    data[N_elem] = -10 * weight;
    N_elem++;
    row[N_elem] = N_cons;
    col[N_elem] = N * (N - 1) - 2;
    data[N_elem] = 10 * weight;
    N_elem++;
    N_cons++;
    //
    row[N_elem] = N_cons;
    col[N_elem] = N * (N - 1);
    data[N_elem] = 10 * weight;
    N_elem++;
    row[N_elem] = N_cons;
    col[N_elem] = N * (N - 1) + 1;
    data[N_elem] = -10 * weight;
    N_elem++;
    row[N_elem] = N_cons;
    col[N_elem] = N * (N - 2);
    data[N_elem] = -10 * weight;
    N_elem++;
    row[N_elem] = N_cons;
    col[N_elem] = N * (N - 2) + 1;
    data[N_elem] = 10 * weight;
    N_elem++;
    N_cons++;
    npd data_(N_elem, data);
    npi32 row_(N_elem, row), col_(N_elem, col);
    delete[] row;
    delete[] col;
    delete[] data;
    return py::make_tuple(py::make_tuple(data_, row_, col_), py::make_tuple(N_cons, N * N));
}

py::tuple bspline_surface_fitting_fill_matrix(npd N_X_, npi32 idx_X_, npd N_Y_, npi32 idx_Y_, const __int32 Nu, const __int32 Nv, const double smooth) {
    auto N_X = (double*)N_X_.request().ptr, N_Y = (double*)N_Y_.request().ptr;
    auto idx_X = (__int32*)idx_X_.request().ptr, idx_Y = (__int32*)idx_Y_.request().ptr;
    int N_point = N_X_.shape()[0], order = N_X_.shape()[1], N_elem = 0, N_cons = 0;
    int Max_Elem = N_point * order * order + 6 * Nu * Nv;
    auto row = new int[Max_Elem], col = new int[Max_Elem];
    auto data = new double[Max_Elem];
    for (int p = 0; p < N_point; p++) {
        auto N_x = N_X + p * order, N_y = N_Y + p * order;
        int idx_x = idx_X[p], idx_y = idx_Y[p];
        for (int ii = 0; ii < order; ii++)
            for (int jj = 0; jj < order; jj++) {
                int i = ii + idx_x, j = jj + idx_y;
                int idx = i * Nv + j;
                row[N_elem] = p;
                col[N_elem] = idx;
                data[N_elem] = N_x[ii] * N_y[jj];
                N_elem++;
            }
    }
    N_cons = N_point;
    for (int i = 1; i < Nu - 1; i++)
        for (int j = 0; j < Nv; j++) {
            row[N_elem] = N_cons;
            col[N_elem] = i * Nv + j;
            data[N_elem] = smooth;
            N_elem++;
            row[N_elem] = N_cons;
            col[N_elem] = (i - 1) * Nv + j;
            data[N_elem] = -0.5 * smooth;
            N_elem++;
            row[N_elem] = N_cons;
            col[N_elem] = (i + 1) * Nv + j;
            data[N_elem] = -0.5 * smooth;
            N_elem++;
            N_cons++;
        }
    for (int i = 0; i < Nu; i++)
        for (int j = 1; j < Nv - 1; j++) {
            row[N_elem] = N_cons;
            col[N_elem] = i * Nv + j;
            data[N_elem] = smooth;
            N_elem++;
            row[N_elem] = N_cons;
            col[N_elem] = i * Nv + j + 1;
            data[N_elem] = -0.5 * smooth;
            N_elem++;
            row[N_elem] = N_cons;
            col[N_elem] = i * Nv + j - 1;
            data[N_elem] = -0.5 * smooth;
            N_elem++;
            N_cons++;
        }
    npd data_(N_elem, data);
    npi32 row_(N_elem, row), col_(N_elem, col);
    delete[] row;
    delete[] col;
    delete[] data;
    return py::make_tuple(data_, row_, col_);
}

py::tuple bspline_basis(npd X_, npd knots_, __int32 p, bool require_D) {
    int Np = X_.shape()[0];
    auto X = (double*)X_.request().ptr, knots = (double*)knots_.request().ptr;
    int order = p + 1, width_knots = p * 2 + 1;
    auto N = new double[Np * order];
    auto DN = new double[Np * order];
    /*omp_set_num_threads(10);
#pragma omp parallel for*/
    for (int idx = 0; idx < Np; idx++) {
        double x = X[idx];
        auto knot = knots + idx * width_knots;
        auto temp = N + idx * order;
        for (int i = 0; i < p; i++)
            temp[i] = 0;
        temp[p] = 1;
        for (int k = 0; k < p-1; k++) {
            for (int i = 0; i < p; i++) {
                temp[i] = (x - knot[i]) / (knot[i + k + 1] - knot[i] + 1e-20) * temp[i] + (knot[i + k + 2] - x) / (knot[i + k + 2] - knot[i + 1] + 1e-20) * temp[ i + 1];
            }
            temp[p] = (x - knot[p]) / (knot[p + k + 1] - knot[p] + 1e-20) * temp[p];
        }
        if (require_D) {
            for (int i = 0; i < p; i++) {
                DN[idx * order + i] = p / (knot[p + i] - knot[i] + 1e-20) * temp[i] - p / (knot[p + i + 1] - knot[i + 1] + 1e-20) * temp[i + 1];
            }
            DN[idx * order + p] = p / (knot[p + p] - knot[p] + 1e-20) * temp[p];
        }

        for (int i = 0; i < p; i++) {
            temp[i] = (x - knot[i]) / (knot[i + p] - knot[i] + 1e-20) * temp[i] + (knot[i + p + 1] - x) / (knot[i + p + 1] - knot[i + 1] + 1e-20) * temp[i + 1];
        }
        temp[p] = (x - knot[p]) / (knot[p + p] - knot[p] + 1e-20) * temp[p];
    }
    npd N_({ Np,order }, N);
    delete[] N;
    if (require_D) {
        npd DN_({ Np,order }, DN);
        delete[] DN;
        return py::make_tuple(N_, DN_);
    }
    delete[] DN;
    return py::make_tuple(N_, py::none());
}

npd bspline_calc(npd coeff_, npd N_X_, npd N_Y_, npi32 idx_X_, npi32 idx_Y_) {
    int Np = N_X_.shape()[0],dim=coeff_.shape()[2],Nu=coeff_.shape()[0], Nv = coeff_.shape()[1];
    auto coeff = (double*)coeff_.request().ptr, N_X = (double*)N_X_.request().ptr, N_Y = (double*)N_Y_.request().ptr;
    auto idx_X = (__int32*)idx_X_.request().ptr, idx_Y = (__int32*)idx_Y_.request().ptr;
    int order = N_X_.shape()[1];
    auto fxy = (double*) new double[Np * dim]();
    /*omp_set_num_threads(10);
#pragma omp parallel for*/
    for (int idx = 0; idx < Np; idx++) {
        for (int ii = 0; ii < order; ii++) {
            for (int jj = 0; jj < order; jj++) {
                int i = ii + idx_X[idx], j = jj + idx_Y[idx];
                double N_xy = N_X[idx * order + ii] * N_Y[idx * order + jj];
                for (int d = 0; d < dim; d++) {
                    fxy[idx * dim + d] += coeff[(i * Nv + j) * dim + d] * N_xy;
                }
            }
        }
    }
    npd fxy_({ Np,dim }, fxy);
    delete[] fxy;
    return fxy_;
}

py::tuple bspline_calc_d(npd coeff_, npd N_X_, npd N_Y_, npd DN_X_, npd DN_Y_, npi32 idx_X_, npi32 idx_Y_) {
    int Np = N_X_.shape()[0], dim = coeff_.shape()[2], Nu = coeff_.shape()[0], Nv = coeff_.shape()[1];
    auto coeff = (double*)coeff_.request().ptr, N_X = (double*)N_X_.request().ptr, N_Y = (double*)N_Y_.request().ptr, DN_X = (double*)DN_X_.request().ptr, DN_Y = (double*)DN_Y_.request().ptr;
    auto idx_X = (__int32*)idx_X_.request().ptr, idx_Y = (__int32*)idx_Y_.request().ptr;
    int order = N_X_.shape()[1];
    auto Dx = (double*) new double[Np * dim](), Dy = (double*) new double[Np * dim]();
    omp_set_num_threads(10);
#pragma omp parallel for
    for (int idx = 0; idx < Np; idx++) {
        for (int ii = 0; ii < order; ii++) {
            for (int jj = 0; jj < order; jj++) {
                int i = ii + idx_X[idx], j = jj + idx_Y[idx];
                double DNxNy = DN_X[idx * order + ii] * N_Y[idx * order + jj], NxDNy = N_X[idx * order + ii] * DN_Y[idx * order + jj];
                for (int d = 0; d < dim; d++) {
                    Dx[idx * dim + d] += coeff[(i * Nv + j) * dim + d] * DNxNy;
                    Dy[idx * dim + d] += coeff[(i * Nv + j) * dim + d] * NxDNy;
                }
            }
        }
    }
    npd Dx_({ Np,dim }, Dx);
    npd Dy_({ Np,dim }, Dy);
    delete[] Dx;
    delete[] Dy;
    return py::make_tuple(Dx_,Dy_);
}

PYBIND11_MODULE(cpp_accelerate, m) {
    m.def("post_process_fill_matrix", &post_fill);
    m.def("get_trimesh_bound", &get_trimesh_bound);
    m.def("trimesh_sample", &trimesh_sample);
    m.def("foldover_fill_matrix", &fold_over_fill);
    m.def("bspline_surface_fitting_fill_matrix", &bspline_surface_fitting_fill_matrix);
    m.def("bspline_basis", &bspline_basis);
    m.def("bspline_calc", &bspline_calc);
    m.def("bspline_calc_d", &bspline_calc_d);
}
