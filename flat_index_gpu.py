# Is a support module for the Zeta Equation Search.py module that can speed up a few areas of the script. 
# This is a A library for building and searching a flat index on GPU.

import numpy as np
import torch

class FlatIndexGPU:
    def __init__(self, dim, dtype='float16', device=None):
        """
        dim:     dimension of each vector
        dtype:   'float32' or 'float16' (defaults to half-precision for reduced memory)
        device:  'cuda' or 'cpu'; if None, will pick 'cuda' when available, else 'cpu'
        """
        self.dim = dim
        # set torch dtype based on string
        if dtype == 'float16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        # auto-select GPU if available
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.prototypes = None

    def add(self, vecs):
        """
        Add (or replace) all prototype vectors.
        vecs: numpy array of shape (N, D)
        """
        arr = np.ascontiguousarray(vecs.astype(self.dtype == torch.float16 and np.float16 or np.float32))
        # ensure prototypes are float32 on GPU
        self.prototypes = torch.from_numpy(arr).cuda().float()

    def search(self, queries: np.ndarray, topk: int = 1, batch_size: int = 1024):
        """
        queries: (N, D) numpy array
        topk: number of nearest neighbors to return per query
        batch_size: queries per batch for GPU efficiency
        returns ids: (N, topk), dists: (N, topk)
        """
        ids_batches = []
        dists_batches = []
        total = queries.shape[0]

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            q_np = queries[start:end]
            # move to GPU and match dtype of prototypes
            q = torch.from_numpy(q_np).cuda().to(self.prototypes.dtype)            # (B, D)
            # prototypes: (P, D) on GPU
            d = torch.cdist(q, self.prototypes, p=2)             # (B, P)
            vals, idxs = torch.topk(d, k=topk, largest=False)    # (B, topk)
            ids_batches.append(idxs.cpu().numpy())
            dists_batches.append(vals.cpu().numpy())

        ids   = np.concatenate(ids_batches,   axis=0)
        dists = np.concatenate(dists_batches, axis=0)
        return ids, dists

    def get_device(self):
        return self.device

    def to(self, device):
        """
        Move the prototypes to a different device ('cuda' or 'cpu').
        """
        self.device = device
        if self.prototypes is not None:
            self.prototypes = self.prototypes.to(device)

# Usage Example (remove or comment out for import as module)
if __name__ == "__main__":
    # Create random data for testing
    np.random.seed(42)
    dim = 8
    n_protos = 1000
    n_queries = 10
    protos = np.random.rand(n_protos, dim).astype(np.float32)
    queries = np.random.rand(n_queries, dim).astype(np.float32)

    index = FlatIndexGPU(dim=dim, dtype='float16')   # or 'float32'
    index.add(protos)
    idx, d = index.search(queries, topk=3)
    print("Indices:\n", idx)
    print("Distances:\n", d)
