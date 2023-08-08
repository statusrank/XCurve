import torch
from torch.utils.data import Sampler
import numpy as np
import pickle as pk
from collections import Counter
from sklearn.utils import shuffle
import math
import pandas as pd


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, 
                class_vector, 
                batch_size, 
                rpos=1, 
                rneg=4):
        '''
        args:
            rpos and rneg: this is only applied to the binary classification problem.
        '''

        self.class_vector = class_vector
        self.batch_size = batch_size

        self.rpos = rpos
        self.rneg = rneg

        if isinstance(class_vector, torch.Tensor):
            y = class_vector.cpu().numpy()
        else:
            y = np.array(class_vector)        

        y_counter = Counter(y)
        self.data = pd.DataFrame({'y': y})

        if len(y_counter.keys()) < 2:
            raise ValueError("class number must be greater or equal than 2, but got {}".format(len(y_counter.keys())))
        # only implemented for binary classification, 1:pos, 0:neg
        elif len(y_counter.keys()) == 2:
            
            ratio = (rneg, rpos)
            
            self.class_batch_size = {
            k: math.ceil(batch_size * ratio[k] / sum(ratio))
            for k in y_counter.keys()
            }

            if rpos / rneg > y_counter[1] / y_counter[0]:
                add_pos = math.ceil(rpos / rneg * y_counter[0]) - y_counter[1]

                # print("-" * 50)
                # print("To balance ratio, add %d pos imgs (with replace = True)" % add_pos)
                # # print(add_pos)
                # print("-" * 50)

                pos_samples = self.data[self.data.y == 1].sample(add_pos, replace=True)

                assert pos_samples.shape[0] == add_pos

                self.data = self.data.append(pos_samples, ignore_index=False)
            else:
                add_neg = math.ceil(rneg / rpos * y_counter[1]) - y_counter[0]

                # print("-" * 50)
                # print("To balance ratio, add %d neg imgs repeatly" % add_neg)
                # print("-" * 50)

                neg_samples = self.data[self.data.y == 0].sample(add_neg, replace=True)

                assert neg_samples.shape[0] == add_neg

                # self.data = self.data.append(neg_samples, ignore_index=False)
                self.data = pd.concat([self.data, neg_samples], ignore_index=False)

        # print("-" * 50)
        # print("after complementary the ratio, having %d images" % self.data.shape[0])
        # print("-" * 50)
        else:
            self.class_batch_size = {
            k: math.ceil(n * batch_size / y.shape[0])
            for k, n in y_counter.items()
            }

        self.real_batch_size = int(sum(self.class_batch_size.values()))

    def gen_sample_array(self):
        # sampling for each class
        def sample_class(group):
            n = self.class_batch_size[group.name]
            return group.sample(n) if group.shape[0] >= n else group.sample(n, replace=True)

        # sampling for each batch
        data = self.data.copy()

        data['idx'] = data.index

        data = data.reset_index()

        result = []
        while True:
            try:
                batch = data.groupby('y', group_keys=False).apply(sample_class)
                assert len(
                    batch) == self.real_batch_size, 'not enough instances!'
            except (ValueError, AssertionError, AttributeError) as e:
                break
            result.extend(shuffle(batch.idx))

            data.drop(index=batch.index, inplace=True)
            # print(len(data), len(result))
        return result

    def __iter__(self):
        self.index_list = self.gen_sample_array()
        return iter(self.index_list)

    def __len__(self):
        try:
            l = len(self.index_list)
        except:
            l = len(self.class_vector)
        return l
