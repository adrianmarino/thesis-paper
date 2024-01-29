import numpy as np
from pytorch_common.modules import CommonMixin
from torch.nn import Module, Embedding
from torch.nn.init import xavier_uniform_
import logging


class MultiFeatureEmbedding(Module, CommonMixin):
    def __init__(
        self,
        features_n_values: list[int],
        embedding_size: int,
        sparse: bool=False
    ):
        """
            This layer allows creates only one embedding for multiples feature variables. Because is not necessary
            create an embedding layer for each input feature allowing save memory.

        :param features_n_values: A list that contains count of all possible values for each categorical feature.
            i.e.: [3, 2, 5], to create an embedding for 3 categorical features with 3, 2, and 5
            possible values respectively. Is important to note that different features have different discrete
            values count. i.e.: user_id has 1000 users and users state have 5 states.

        :param embedding_size: Embedding vector len. Same for all features.

        :param sparse (bool, optional): If True, gradient w.r.t. weight matrix will be a sparse tensor.
        """
        super().__init__()
        self.embedding = Embedding(
            num_embeddings  =   sum(features_n_values),
            embedding_dim   =   embedding_size,
            sparse          =   sparse
        )
        xavier_uniform_(self.embedding.weight.data)

        self.feat_emb_offset = np.concatenate((np.zeros(1), np.cumsum(features_n_values)[:-1]), axis=0)

    def _sum_emb_offset(self, x): return x + x.new_tensor(self.feat_emb_offset).unsqueeze(0)

    def forward(self, x):
        """
            Replace input values for embedding vectors. Find into embedding lookup table indexing by input values
            and get the associated embedding vector.

        :param batches: A list of batches. Each batch contains a list of feature vectors.
        :return: Same batches list but this contains embeddings instead of categorical input values.
        """
        return self.embedding(self._sum_emb_offset(x))

    @property
    def vectors(self): return self.params['embedding.weight']


    @property
    def weights(self): return self.params['embedding.weight']


    @property
    def feature_indexes(self):
        return list(range(0, len(self.feat_emb_offset)))


    def _from_to_by_emb_index(self, index):
        from_pos = int(self.feat_emb_offset[index])

        if (index+1) < len(self.feat_emb_offset):
            next_start_pos = int(self.feat_emb_offset[index+1])
            to_pos = next_start_pos -1
        else:
            to_pos = self.weights.shape[0]-1

        return from_pos, to_pos


    def feature_embeddings_by_index(self, index):
        from_pos, to_pos = self._from_to_by_emb_index(index)

        embeddings =  self.weights[from_pos:(to_pos+1), :]

        logging.info(f'Get feature {index} embeddings {list(embeddings.shape)}. Detail: Embeddings from {from_pos} to {to_pos} from weights {list(self.weights.shape)}')

        return embeddings


    @property
    def feature_embeddings(self):
        return [self.feature_embeddings_by_index(idx) for idx in self.feature_indexes]


    def __repr__(self):
        original_repr = super(MultiFeatureEmbedding, self).__repr__()
        return f'{original_repr}{self._feature_embeddings_metadata()}'


    def _feature_embeddings_metadata(self):
        metadata = ''
        for index in self.feature_indexes:
            from_pos, to_pos = self._from_to_by_emb_index(index)
            metadata += f'\n- FeatureEmbedding{index}(size: {to_pos-from_pos+1}, from: {from_pos}, to: {to_pos})'
        return metadata
