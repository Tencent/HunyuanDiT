import math

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


class BlockDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False,
                 batch_size=-1, start_index=0):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        if batch_size == -1:
            raise ValueError("batch_size should be specified")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self._start_index = start_index
        self.recompute_sizes()

    @property
    def start_index(self):
        return self._start_index

    @start_index.setter
    def start_index(self, value):
        self._start_index = value
        self.recompute_sizes()

    def recompute_sizes(self):
        self.num_samples = len(self.dataset) // self.batch_size * self.batch_size // self.num_replicas \
                           - self._start_index
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        raw_num_samples = len(indices) // self.batch_size * self.batch_size // self.num_replicas
        raw_total_size = raw_num_samples * self.num_replicas
        indices = indices[:raw_total_size]

        # We require that the dataset size is divisible by batch_size * num_replicas
        # This is naturally satisfied when using index_kits.
        # In future, we can remove this assertion.
        assert len(indices) == raw_total_size, f"{len(indices)} vs {raw_total_size}"

        # subsample with start_index
        indices = indices[self.rank * raw_num_samples + self.start_index:(self.rank + 1) * raw_num_samples]
        assert len(indices) + self.start_index == raw_num_samples, \
            f"{len(indices) + self.start_index} vs {raw_num_samples}"

        # This is a sequential sampler. The shuffle operation is done by the dataset itself.
        return iter(indices)


class DistributedSamplerWithStartIndex(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False,
                 start_index=0):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self._start_index = start_index
        self.recompute_sizes()
        self.shuffle = shuffle
        self.seed = seed

    @property
    def start_index(self):
        return self._start_index

    @start_index.setter
    def start_index(self, value):
        self._start_index = value
        self.recompute_sizes()

    def recompute_sizes(self):
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and (len(self.dataset) - self._start_index) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                ((len(self.dataset) - self._start_index) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil((len(self.dataset) - self._start_index) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(self._start_index, len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample with start_index
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
