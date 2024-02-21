import joblib
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import time
from utils import  setup_seed, build_tri_mesh, draw_tri_mesh, draw_smooth_result
from preprocess import PreProcess
from postprocess import PostProcess
import os
from net import Net

def run(idx, net, bound_ori, device=torch.device("cpu"), save_tri=None, max_iter=50):
    t_one = 0
    pre = PreProcess(bound_ori)
    bound = pre.forward(bound_ori).astype(np.float32)

    bound_simplify = np.copy(bound[::4, :])
    bound_simplify_torch = torch.from_numpy(bound_simplify).to(device)
    # print("\tTriangulation Sample")
    T1 = time.time()
    tri_mesh = build_tri_mesh(bound, max_area=0.0005)
    if tri_mesh.points().shape[0] > 5000:
        print("Error ", idx)
        plt.plot(bound_ori[:-3, 0], bound_ori[:-3, 1])
        plt.show()
        exit(1)
    t_one += time.time() - T1
    print("\tTriangulation Finished, Time Cost = %.03f s" % (time.time() - T1))
    # print("\tNetwork")
    T1 = time.time()
    points = tri_mesh.points()[:, :2].astype(np.float32)
    points_uv = net(torch.from_numpy(points).to(device), bound_simplify_torch).detach().cpu().numpy()
    t_one += time.time() - T1
    print("\tNetwork Finished, Time Cost = %.03f s" % (time.time() - T1))
    # print("\tPost Process")
    T1 = time.time()
    bs, bij = PostProcess(tri_mesh, points_uv, pre, max_iter=max_iter, device=device)
    t_one += time.time() - T1
    print("\tPost Process Finished, Time Cost = %.03f s" % (time.time() - T1))
    if save_tri is not None:
        plt.cla()
        draw_tri_mesh(pre.backward(tri_mesh.points()[:, :2].astype(np.float32)), tri_mesh.edge_vertex_indices(), save_tri)
    return bs, t_one


if __name__ == "__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "./trained_parameters"
    BOUND_PATH = "./points_mpeg7.csv"
    # First 150 boundaries of points_mpeg7.csv are the testing set
    N_BOUND = 150
    SAVE_PATH = "./result"
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    bounds = pd.read_csv(BOUND_PATH).values.astype(np.float32).reshape([-1, 2, 1024]).transpose([0, 2, 1])[:N_BOUND, :, :]

    device = torch.device("cuda")
    width = 512
    net = Net(width=width).to(device)
    net.eval()
    try:
        net.load_state_dict(torch.load(MODEL_PATH))
    except:
        net.load_state_dict(torch.load(MODEL_PATH)[0])
    update = False
    fig = plt.figure(figsize=[8, 8])
    test_loss = 0
    count = 0
    t_all = 0
    # pytorch collects information in the first cycle, which leads to more computational time. So we run a sample before time counting.
    run(0, net, bounds[0, :, :], device=device)
    print()
    # test
    inputs = []
    for i in range(N_BOUND):
        setup_seed(9527)
        bound_ori = bounds[i, :, :]
        bs, t_one = run(i, net, bound_ori, device=device, save_tri=None, max_iter=100)  # save_tri="./img/" + str(i) + "_tri.pdf")
        joblib.dump(bs,  os.path.join(SAVE_PATH,str(i)))
        plt.cla()
        plt.axis("equal")
        draw_smooth_result(bs, save=os.path.join(SAVE_PATH,str(i)+ ".pdf"), bound=bound_ori)
        # draw_mesh(xy, bound=None, save="./img/" + str(i) + ".pdf")
        plt.cla()
        t_all += t_one
        print("[%d] Average Time Cost = %.03f s" % (i, t_all / (i + 1)))
