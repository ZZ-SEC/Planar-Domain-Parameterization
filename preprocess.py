import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import get_area
from sklearn.decomposition import PCA


class PreProcess():
    def __init__(self, bound):
        self.bound = bound
        area = get_area(bound)
        pca = PCA(n_components=1)
        pca.fit(bound)
        self.direction = pca.components_
        if self.direction[0, 0] < 0 and self.direction[0, 1] < 0:
            self.direction = -self.direction
        elif self.direction[0, 0] < 0 and self.direction[0, 1] > 0:
            self.direction[0, 0], self.direction[0, 1] = self.direction[0, 1], -self.direction[0, 0]
        elif self.direction[0, 0] > 0 and self.direction[0, 1] < 0:
            self.direction[0, 0], self.direction[0, 1] = -self.direction[0, 1], self.direction[0, 0]
        self.direction /= np.linalg.norm(self.direction)
        rotate_matrix = np.zeros([2, 2])
        rotate_matrix[0, :] = self.direction
        rotate_matrix[1, 0], rotate_matrix[1, 1] = -self.direction[0, 1], self.direction[0, 0]
        self.rotate_matrix = rotate_matrix
        self.inv_rotate_matrix = np.linalg.inv(rotate_matrix)
        bound_rotate = np.matmul(self.rotate_matrix, bound.transpose())
        max = np.max(bound_rotate, axis=1)
        min = np.min(bound_rotate, axis=1)
        self.mid = ((min + max) / 2).reshape([2, 1])
        scale = 1 / (max - min)
        scale /= np.sqrt(area * scale[0] * scale[1])
        self.scale = scale.reshape([2, 1])

    def forward(self, x):
        shape = x.shape
        x = x.reshape([-1, 2])
        y = x.transpose()
        # [2,N]
        # Rotate
        y = np.matmul(self.rotate_matrix, y)
        # Move
        y = (y - self.mid) * self.scale + 0.5
        y = y.transpose().reshape(shape)
        return y

    def backward(self, x):
        shape = x.shape
        x = x.reshape([-1, 2])
        y = x.transpose()
        y = (y - 0.5) / self.scale + self.mid
        y = np.matmul(self.inv_rotate_matrix, y)
        y = y.transpose().reshape(shape)
        return y

class PreProcessFix():
    def __init__(self, bound):
        self.bound = bound
        area = get_area(bound)
        pca = PCA(n_components=1)
        pca.fit(bound)
        self.direction = pca.components_
        if self.direction[0, 0] < 0 and self.direction[0, 1] < 0:
            self.direction = -self.direction
        elif self.direction[0, 0] < 0 and self.direction[0, 1] > 0:
            self.direction[0, 0], self.direction[0, 1] = self.direction[0, 1], -self.direction[0, 0]
        elif self.direction[0, 0] > 0 and self.direction[0, 1] < 0:
            self.direction[0, 0], self.direction[0, 1] = -self.direction[0, 1], self.direction[0, 0]
        self.direction /= np.linalg.norm(self.direction)
        rotate_matrix = np.zeros([2, 2])
        rotate_matrix[0, :] = self.direction
        rotate_matrix[1, 0], rotate_matrix[1, 1] = -self.direction[0, 1], self.direction[0, 0]
        self.rotate_matrix = rotate_matrix
        self.inv_rotate_matrix = np.linalg.inv(rotate_matrix)
        bound_rotate = np.matmul(self.rotate_matrix, bound.transpose())
        max = np.max(bound_rotate, axis=1)
        min = np.min(bound_rotate, axis=1)
        self.mid = ((min + max) / 2).reshape([2, 1])
        scale = 1 / np.sqrt(area)
        self.scale = scale

    def forward(self, x):
        shape = x.shape
        x = x.reshape([-1, 2])
        y = x.transpose()
        # [2,N]
        # Rotate
        y = np.matmul(self.rotate_matrix, y)
        # Move
        y = (y - self.mid) * self.scale + 0.5
        y = y.transpose().reshape(shape)
        return y

    def backward(self, x):
        shape = x.shape
        x = x.reshape([-1, 2])
        y = x.transpose()
        y = (y - 0.5) / self.scale + self.mid
        y = np.matmul(self.inv_rotate_matrix, y)
        y = y.transpose().reshape(shape)
        return y

if __name__ == "__main__":
    bounds = pd.read_csv("points1024.csv").values.reshape([-1, 2, 1024])
    bound = bounds[0, :, :].transpose()
    pre = PreProcess(bound)
    bound_pre = pre.forward(bound)
    plt.axis("equal")
    plt.plot(bound_pre[:, 0], bound_pre[:, 1])
    plt.show()
    bound = pre.backward(bound_pre)
    plt.axis("equal")
    plt.plot(bound[:, 0], bound[:, 1])
    plt.show()
