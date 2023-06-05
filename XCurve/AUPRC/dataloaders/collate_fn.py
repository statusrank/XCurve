r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).
These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    result = dict()
    for k in batch[0].keys():
        result[k] = torch.cat([i[k] for i in batch], 0)
    return result
