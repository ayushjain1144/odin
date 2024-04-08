# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import itertools
import logging
import torch.utils.data
from typing import Optional
from torch.utils.data.sampler import Sampler, BatchSampler
import torch.utils.data as torchdata
import random
import copy
import numpy as np
from tabulate import tabulate
from termcolor import colored
from detectron2.utils.logger import log_first_n
import random

from detectron2.config import CfgNode, configurable
from detectron2.data.build import (
    build_batch_data_loader,
    load_proposals_into_dataset,
    trivial_batch_collator,
)
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import DatasetFromList, MapDataset, _shard_iterator_dataloader_worker
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import InferenceSampler, TrainingSampler
from detectron2.utils.comm import get_world_size
from detectron2.data.build import worker_init_reset_seed
from detectron2.data.detection_utils import check_metadata_consistency


from detectron2.utils import comm

import ipdb
st = ipdb.set_trace

def _compute_num_images_per_worker(cfg: CfgNode):
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers
    return images_per_worker



def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        if type(annos) == list and type(annos[0] == list):
            classes =  np.asarray(
                [x[0]["category_id"] for x in annos if len(x) > 0 and not x[0].get("iscrowd", 0)], dtype=int
            )
        else:
            classes = np.asarray(
                [x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=int
            )
        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            assert (
                classes.max() < num_classes
            ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]
    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan"),
        key="message",
    )



def get_detection_dataset_dicts(
    dataset_names, proposal_files=None, subsample_data=None
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names)
    print("Number of datasets: ", len(dataset_names))
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]
        
    if subsample_data is not None:
        # set seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        # subsample data
        dataset_dicts = [
            random.sample(dataset_i_dicts, subsample_data)
            for dataset_i_dicts in dataset_dicts
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
    has_instances = "annotations" in dataset_dicts[0]
    if has_instances:
        try:
            class_names = MetadataCatalog.get(dataset_names[0]).thing_classes
            check_metadata_consistency("thing_classes", dataset_names)
            print_instances_class_histogram(dataset_dicts, class_names)
        except:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(dataset_names))
    return dataset_dicts


def _train_loader_from_config(cfg, mapper, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

    # if mapper is None:
    #     mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        print("len(dataset)", len(dataset))
        sampler = TrainingSampler(len(dataset))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }



def _train_loader_from_config_multitask(cfg):
    return {
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "total_batch_size_2d": cfg.SOLVER.IMS_PER_BATCH_2D,
        "total_batch_size_3d": cfg.SOLVER.IMS_PER_BATCH_3D,
        "cfg": cfg
    }


def collate_fn(batch):
    """
    Collate function for training. is called ONLY in train loader
    Args:
        batch (list[dict]): a list of samples, each sample is a dict of image, instances, etc.
        do_view_aug (bool): whether to do view augmentation
        max_frames (int): maximum number of frames to use for view augmentation
        max_total_images (int): maximum number of images to use FOR BATCHING
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    
    do_view_aug, max_frames = batch[0]['do_camera_drop'], batch[0]['max_frames']
    use_ghost = batch[0]['use_ghost']

    special_keys = ['multi_scale_xyz']

    if not do_view_aug or np.random.random() < 0.5:
        return batch

    batch_copy = copy.deepcopy(batch)
    
    num_frames = np.random.randint(1, max_frames + 1)
    num_batch_instances = 0
    for i, sample in enumerate(batch):
        keep = np.random.choice(max_frames, num_frames, replace=False)

        # check if after keeping keep we will end up with empty labels
        for key in sample.keys():
            if type(sample[key]) == list and len(sample[key]) == max_frames:
                sample[key] = [sample[key][index] for index in keep]
                if key == "instances_all":
                    num_keys = sum([len(sample[key][i]) for i in range(len(sample[key]))])
                    num_batch_instances += num_keys

            if key in special_keys:
                assert len(sample[key][0]) == max_frames
                sample[key] = [sample[key][j][torch.from_numpy(keep)] for j in range(len(sample[key]))]

            if type(sample[key]) == torch.Tensor and sample[key].shape[0] == max_frames:
                sample[key] = sample[key][torch.from_numpy(keep)]
        
        sample['keep'] = keep

    if not use_ghost and num_batch_instances == 0:
        print("WARNING: no instances in batch, returning original batch")
        return batch_copy

    return batch



def merge_datasets(datasets, mappers, balance=False):
    if balance:
        max_len = max([len(dataset) for dataset in datasets])
        for dataset in datasets:
            dataset.extend(dataset * ((max_len // len(dataset)) - 1))
    datasets = [MapDataset(dataset, mapper) for dataset, mapper in zip(datasets, mappers)]
            
    dataset = torch.utils.data.ConcatDataset(datasets)
    return dataset


# TODO can allow dataset as an iterable or IterableDataset to make this function more general
@configurable(from_config=_train_loader_from_config_multitask)
def build_detection_train_loader_multi_task(
    dataset_3d, dataset_2d, *, mapper_3d, mapper_2d,
    sampler=None, total_batch_size, total_batch_size_2d, total_batch_size_3d,
    aspect_ratio_grouping=True, num_workers=0, cfg=None
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that
            produces indices to be applied on ``dataset``.
            Default to :class:`TrainingSampler`, which coordinates a random shuffle
            sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader: a dataloader. Each output from it is a
            ``list[mapped_element]`` of length ``total_batch_size / num_workers``,
            where ``mapped_element`` is produced by the ``mapper``.
    """
    if mapper_3d is not None:
        dataset_3d = MapDataset(dataset_3d, mapper_3d)
    if mapper_2d is not None:
        dataset_2d = MapDataset(dataset_2d, mapper_2d)
    size_list = [len(dataset_3d), len(dataset_2d)]
    print("size_list", size_list)
    # convert size list to cumulative sum
    size_list = [0] + size_list
    size_list = np.cumsum(size_list).tolist()
    dataset = torch.utils.data.ConcatDataset([dataset_3d, dataset_2d])
    # if sampler is None:
    sampler = MultiTaskTrainingSampler(len(dataset), size_list=size_list)
    # sampler = TrainingSampler(len(dataset))
    group_sizes = []
    for i in range(len(size_list) - 1):
        group_sizes.extend([i] * (size_list[i + 1] - size_list[i]))
        
    total_batch_size = [total_batch_size_3d // get_world_size(), total_batch_size_2d // get_world_size()]
    # print("total_batch_size", total_batch_size)
    # total_batch_size = [total_batch_size_3d, total_batch_size_2d]
    sampler = GroupedBatchSampler(sampler, group_sizes, total_batch_size, prob=cfg.PROB)
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader_multitask(
        dataset,
        sampler,
        # aspect_ratio_grouping=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


# TODO can allow dataset as an iterable or IterableDataset to make this function more general
@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0, cfg=None
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that
            produces indices to be applied on ``dataset``.
            Default to :class:`TrainingSampler`, which coordinates a random shuffle
            sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader: a dataloader. Each output from it is a
            ``list[mapped_element]`` of length ``total_batch_size / num_workers``,
            where ``mapped_element`` is produced by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def _test_loader_from_config(cfg, mapper=None, dataset=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.TEST_NUM_WORKERS,
        "total_batch_size": cfg.SOLVER.TEST_IMS_PER_BATCH,
        }


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(dataset, *, mapper, total_batch_size, num_workers=0):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        num_workers (int): number of parallel data loading workers

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False, serialize=True)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(len(dataset))

    world_size = get_world_size()
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, total_batch_size // world_size, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    """

    def __init__(self, sampler, group_ids, batch_size, prob=None):
        """
        Args:
            sampler (Sampler): Base sampler.
            group_ids (list[int]): If the sampler produces indices in range [0, N),
                `group_ids` must be a list of `N` ints which contains the group id of each sample.
                The group ids must be a set of integers in the range [0, num_groups).
            batch_size (int): Size of mini-batch.
        """
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = np.asarray(group_ids)
        assert self.group_ids.ndim == 1
        self.batch_size = batch_size
        self.groups = np.unique(self.group_ids).tolist()
        self.prob = torch.tensor(prob) if prob is not None else None

        seed = comm.shared_random_seed()
        self._seed = int(seed)
        
        self.max_buffer_size = max(self.batch_size) * 10

        # buffer the indices of each group until batch size is reached
        # self.buffer_per_group = {k: [] for k in self.groups}

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        
        if self.prob is not None:
            selected_group = torch.multinomial(self.prob, 1, generator=g).item()
        else:
            selected_group = torch.randint(0, len(self.groups), (1,), generator=g).item()
            
        # selected_group = torch.randint(0, len(self.groups), (1,), generator=g).item()
        
        buffer_per_group = {k: [] for k in self.groups}

        for idx in self.sampler:
            # print("idx", idx, "selected_group", selected_group, flush=True)
            group_id = self.group_ids[idx]
            group_buffer = buffer_per_group[group_id]
            group_buffer.append(idx)
            if len(buffer_per_group[selected_group]) >= self.batch_size[selected_group]:
                # print("yielding", flush=True)
                data = buffer_per_group[selected_group][:self.batch_size[selected_group]]
                buffer_per_group[selected_group] = buffer_per_group[selected_group][self.batch_size[selected_group]:self.max_buffer_size]
                # selected_group = torch.randint(0, len(self.groups), (1,), generator=g).item()
                if self.prob is not None:
                    selected_group = torch.multinomial(self.prob, 1, generator=g).item()
                else:
                    selected_group = torch.randint(0, len(self.groups), (1,), generator=g).item()
         
                yield data  # yield a copy of the list

    def __len__(self):
        raise NotImplementedError("len() of GroupedBatchSampler is not well-defined.")


def build_batch_data_loader_multitask(
    dataset,
    sampler,
    *,
    num_workers=0,
    collate_fn=None,
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces indices.
            Must be provided iff. ``dataset`` is a map-style dataset.
        total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    # dataset = ToIterableDatasetMultiTask(dataset, sampler)

    data_loader = torchdata.DataLoader(
        dataset,
        batch_sampler=sampler,
        # drop_last=True,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        worker_init_fn=worker_init_reset_seed,
    )
    return data_loader


class ToIterableDatasetMultiTask(torchdata.IterableDataset):
    """
    Convert an old indices-based (also called map-style) dataset
    to an iterable-style dataset.
    """

    def __init__(self, dataset: torchdata.Dataset, sampler: Sampler, shard_sampler: bool = True):
        """
        Args:
            dataset: an old-style dataset with ``__getitem__``
            sampler: a cheap iterable that produces indices to be applied on ``dataset``.
            shard_sampler: whether to shard the sampler based on the current pytorch data loader
                worker id. When an IterableDataset is forked by pytorch's DataLoader into multiple
                workers, it is responsible for sharding its data based on worker id so that workers
                don't produce identical data.

                Most samplers (like our TrainingSampler) do not shard based on dataloader worker id
                and this argument should be set to True. But certain samplers may be already
                sharded, in that case this argument should be set to False.
        """
        assert not isinstance(dataset, torchdata.IterableDataset), dataset
        assert isinstance(sampler, Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler
        self.shard_sampler = shard_sampler

    def __iter__(self):
        if not self.shard_sampler:
            sampler = self.sampler
        else:
            # With map-style dataset, `DataLoader(dataset, sampler)` runs the
            # sampler in main process only. But `DataLoader(ToIterableDataset(dataset, sampler))`
            # will run sampler in every of the N worker. So we should only keep 1/N of the ids on
            # each worker. The assumption is that sampler is cheap to iterate so it's fine to
            # discard ids in workers.
            sampler = _shard_iterator_dataloader_worker(self.sampler)
        
        for idx_list in sampler:
            yield collate_fn([self.dataset[idx] for idx in idx_list])
            # for idx in idx_list:
            #     yield self.dataset[idx]

    def __len__(self):
        return len(self.sampler)




class MultiTaskTrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)

    Note that this sampler does not shard based on pytorch DataLoader worker id.
    A sampler passed to pytorch DataLoader is used only with map-style dataset
    and will not be executed inside workers.
    But if this sampler is used in a way that it gets execute inside a dataloader
    worker, then extra work needs to be done to shard its outputs based on worker id.
    This is required so that workers don't produce identical data.
    :class:`ToIterableDataset` implements this logic.
    This note is true for all samplers in detectron2.
    """

    def __init__(
            self, size: int, shuffle: bool = True,
            seed: Optional[int] = None, size_list=None
        ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        if not isinstance(size, int):
            raise TypeError(f"TrainingSampler(size=) expects an int. Got type {type(size)}.")
        if size <= 0:
            raise ValueError(f"TrainingSampler(size=) expects a positive int. Got {size}.")
        self._size = size
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        self.size_list = size_list

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        
        actual_size_list = np.array(self.size_list[1:]) - np.array(self.size_list[:-1])
        repeat_factor = np.array([max(actual_size_list)] * len(actual_size_list)) // actual_size_list
        indices = np.concatenate(
            [np.tile(np.arange(self.size_list[i], self.size_list[i+1]), repeat_factor[i]) for i in range(len(self.size_list) - 1)]
        )

        while True:
            rand_perm = torch.randperm(len(indices), generator=g)
            yield from indices[rand_perm].tolist()