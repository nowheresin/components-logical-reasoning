import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def calculate_cosine_similarity(vec: list[list]) -> np:
    new_vec = []
    for i in vec:
        if isinstance(i, torch.Tensor):
            i = i.cpu().detach().squeeze().numpy()
        new_vec.append(i)
    return cosine_similarity(new_vec, new_vec)


def calculate_reduce_dim(x: list[list], method: str = 'TSNE') -> list[list]:
    reduce_method = None
    if method == 'PCA':
        reduce_method = PCA
    if method == 'TSNE':
        reduce_method = TSNE
    if reduce_method is None:
        raise ValueError("The param [method] is error.")

    return reduce_method(n_components=2).fit_transform(x)
