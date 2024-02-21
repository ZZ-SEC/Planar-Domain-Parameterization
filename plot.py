import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from utils import draw_smooth_result, get_area

bounds = pd.read_csv("./points_mpeg7.csv").values.astype(np.float32).reshape([-1, 2, 1024]).transpose([0, 2, 1])
bound_plot = [50]
N_plot = len(bound_plot)
fig = plt.figure()
fig.set_size_inches([10, 10])
save_dir = "./plot"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
for i in range(N_plot):
    item = bound_plot[i]
    print("\b" * 100 + "%d/%d" % (i, N_plot), end="")
    bound = bounds[item, :, :]
    scale = np.sqrt(1 / get_area(bound))
    bs = joblib.load("./result/" + str(item))
    plt.clf()
    bound_ = np.concatenate([bound, bound[:1, :]], axis=0)
    plt.plot(bound_[:, 0], bound_[:, 1], "gray", linewidth=1, zorder=1)
    draw_smooth_result(bs, scale=scale, colored=None, save=os.path.join(save_dir, str(item) + ".pdf"))
    plt.clf()
    draw_smooth_result(bs, scale=scale, colored="area", save=os.path.join(save_dir, str(item) + "_J.pdf"))
    plt.clf()
    draw_smooth_result(bs, scale=scale, colored="angle", save=os.path.join(save_dir, str(item) + "_mu.pdf"))
    plt.clf()
    # plt.plot(bound[:,0],bound[:,1],"k")
