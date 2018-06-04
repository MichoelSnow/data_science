import os, collections, cv2, torch, math, threading, random, copy, bcolz, contextlib
import numpy as np
from tqdm import tqdm, tqdm_notebook, tnrange
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from abc import abstractmethod
from PIL import Image
import pandas as pd
from enum import IntEnum
from .fastai_models import resnext_50_32x4d, resnext_101_32x4d, resnext_101_64x4d, wrn_50_2f, InceptionResnetV2, \
    inceptionv4
from collections import Iterable, OrderedDict
from itertools import chain
from distutils.version import LooseVersion
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, vgg16_bn, vgg19_bn, \
    densenet121, densenet161, densenet169, densenet201
from torch.nn.init import kaiming_normal
import torch.nn.functional as F

string_classes = (str, bytes)
USE_GPU = torch.cuda.is_available()
IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')


def sum_geom(a, r, n): return a * n if r == 1 else math.ceil(a * (1 - r ** n) / (1 - r))


class ModelData(object):
    def __init__(self, path, trn_dl, val_dl, test_dl=None):
        self.path, self.trn_dl, self.val_dl, self.test_dl = path, trn_dl, val_dl, test_dl

    @classmethod
    def from_dls(cls, path, trn_dl, val_dl, test_dl=None):
        return cls(path, trn_dl, val_dl, test_dl)

    @property
    def is_reg(self): return self.trn_ds.is_reg

    @property
    def is_multi(self): return self.trn_ds.is_multi

    @property
    def trn_ds(self): return self.trn_dl.dataset

    @property
    def val_ds(self): return self.val_dl.dataset

    @property
    def test_ds(self): return self.test_dl.dataset

    @property
    def trn_y(self): return self.trn_ds.y

    @property
    def val_y(self): return self.val_ds.y


class ImageData(ModelData):
    def __init__(self, path, datasets, bs, num_workers, classes):
        trn_ds, val_ds, fix_ds, aug_ds, test_ds, test_aug_ds = datasets
        self.path, self.bs, self.num_workers, self.classes = path, bs, num_workers, classes
        self.trn_dl, self.val_dl, self.fix_dl, self.aug_dl, self.test_dl, self.test_aug_dl = [
            self.get_dl(ds, shuf) for ds, shuf in [
                (trn_ds, True), (val_ds, False), (fix_ds, False), (aug_ds, False),
                (test_ds, False), (test_aug_ds, False)
            ]
        ]

    def get_dl(self, ds, shuffle):
        if ds is None:
            return None
        return DataLoader(ds, batch_size=self.bs, shuffle=shuffle,
                          num_workers=self.num_workers, pin_memory=False)

    @property
    def sz(self):
        return self.trn_ds.sz

    @property
    def c(self):
        return self.trn_ds.c

    def resized(self, dl, targ, new_path):
        return dl.dataset.resize_imgs(targ, new_path) if dl else None

    def resize(self, targ_sz, new_path='tmp'):
        new_ds = []
        dls = [self.trn_dl, self.val_dl, self.fix_dl, self.aug_dl]
        if self.test_dl:
            dls += [self.test_dl, self.test_aug_dl]
        else:
            dls += [None, None]
        t = tqdm_notebook(dls)
        for dl in t:
            new_ds.append(self.resized(dl, targ_sz, new_path))
        t.close()
        return self.__class__(new_ds[0].path, new_ds, self.bs, self.num_workers, self.classes)

    @staticmethod
    def get_ds(fn, trn, val, tfms, test=None, **kwargs):
        res = [
            fn(trn[0], trn[1], tfms[0], **kwargs),  # train
            fn(val[0], val[1], tfms[1], **kwargs),  # val
            fn(trn[0], trn[1], tfms[1], **kwargs),  # fix
            fn(val[0], val[1], tfms[0], **kwargs)  # aug
        ]
        if test is not None:
            if isinstance(test, tuple):
                test_lbls = test[1]
                test = test[0]
            else:
                test_lbls = np.zeros((len(test), trn[1].shape[1]))
            res += [
                fn(test, test_lbls, tfms[1], **kwargs),  # test
                fn(test, test_lbls, tfms[0], **kwargs)  # test_aug
            ]
        else:
            res += [None, None]
        return res


class ImageClassifierData(ImageData):
    @classmethod
    def from_arrays(cls, path, trn, val, bs=64, tfms=(None, None), classes=None, num_workers=4, test=None):
        """ Read in images and their labels given as numpy arrays

        Arguments:
            path: a root path of the data (used for storing trained models, precomputed values, etc)
            trn: a tuple of training data matrix and target label/classification array (e.g. `trn=(x,y)` where `x`
                has the shape of `(5000, 784)` and `y` has the shape of `(5000,)`)
            val: a tuple of validation data matrix and target label/classification array.
            bs: batch size
            tfms: transformations (for data augmentations). e.g. output of `tfms_from_model`
            classes: a list of all labels/classifications
            num_workers: a number of workers
            test: a matrix of test data (the shape should match `trn[0]`)

        Returns:
            ImageClassifierData
        """
        datasets = cls.get_ds(ArraysIndexDataset, trn, val, tfms, test=test)
        return cls(path, datasets, bs, num_workers, classes=classes)

    @classmethod
    def from_paths(cls, path, bs=64, tfms=(None, None), trn_name='train', val_name='valid', test_name=None,
                   test_with_labels=False, num_workers=8):
        """ Read in images and their labels given as sub-folder names

        Arguments:
            path: a root path of the data (used for storing trained models, precomputed values, etc)
            bs: batch size
            tfms: transformations (for data augmentations). e.g. output of `tfms_from_model`
            trn_name: a name of the folder that contains training images.
            val_name:  a name of the folder that contains validation images.
            test_name:  a name of the folder that contains test images.
            num_workers: number of workers

        Returns:
            ImageClassifierData
        """
        assert not (
                tfms[0] is None or tfms[1] is None), "please provide transformations for your train and validation sets"
        trn, val = [folder_source(path, o) for o in (trn_name, val_name)]
        if test_name:
            test = folder_source(path, test_name) if test_with_labels else read_dir(path, test_name)
        else:
            test = None
        datasets = cls.get_ds(FilesIndexArrayDataset, trn, val, tfms, path=path, test=test)
        return cls(path, datasets, bs, num_workers, classes=trn[2])

    @classmethod
    def from_csv(cls, path, folder, csv_fname, bs=64, tfms=(None, None),
                 val_idxs=None, suffix='', test_name=None, continuous=False, skip_header=True, num_workers=8):
        """ Read in images and their labels given as a CSV file.

        This method should be used when training image labels are given in an CSV file as opposed to
        sub-directories with label names.

        Arguments:
            path: a root path of the data (used for storing trained models, precomputed values, etc)
            folder: a name of the folder in which training images are contained.
            csv_fname: a name of the CSV file which contains target labels.
            bs: batch size
            tfms: transformations (for data augmentations). e.g. output of `tfms_from_model`
            val_idxs: index of images to be used for validation. e.g. output of `get_cv_idxs`.
                If None, default arguments to get_cv_idxs are used.
            suffix: suffix to add to image names in CSV file (sometimes CSV only contains the file name without file
                    extension e.g. '.jpg' - in which case, you can set suffix as '.jpg')
            test_name: a name of the folder which contains test images.
            continuous:
            skip_header: skip the first row of the CSV file.
            num_workers: number of workers

        Returns:
            ImageClassifierData
        """
        assert not (
                tfms[0] is None or tfms[1] is None), "please provide transformations for your train and validation sets"
        assert not (os.path.isabs(folder)), "folder needs to be a relative path"
        fnames, y, classes = csv_source(folder, csv_fname, skip_header, suffix, continuous=continuous)
        return cls.from_names_and_array(path, fnames, y, classes, val_idxs, test_name,
                                        num_workers=num_workers, suffix=suffix, tfms=tfms, bs=bs, continuous=continuous)

    @classmethod
    def from_names_and_array(cls, path, fnames, y, classes, val_idxs=None, test_name=None,
                             num_workers=8, suffix='', tfms=(None, None), bs=64, continuous=False):
        val_idxs = get_cv_idxs(len(fnames)) if val_idxs is None else val_idxs
        ((val_fnames, trn_fnames), (val_y, trn_y)) = split_by_idx(val_idxs, np.array(fnames), y)

        test_fnames = read_dir(path, test_name) if test_name else None
        if continuous:
            f = FilesIndexArrayRegressionDataset
        else:
            f = FilesIndexArrayDataset if len(trn_y.shape) == 1 else FilesNhotArrayDataset
        datasets = cls.get_ds(f, (trn_fnames, trn_y), (val_fnames, val_y), tfms,
                              path=path, test=test_fnames)
        return cls(path, datasets, bs, num_workers, classes=classes)


class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, pad_idx=0,
                 num_workers=None, pin_memory=False, drop_last=False, pre_pad=True, half=False,
                 transpose=False, transpose_y=False):
        self.dataset, self.batch_size, self.num_workers = dataset, batch_size, num_workers
        self.pin_memory, self.drop_last, self.pre_pad = pin_memory, drop_last, pre_pad
        self.transpose, self.transpose_y, self.pad_idx, self.half = transpose, transpose_y, pad_idx, half

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        if num_workers is None:
            self.num_workers = num_cpus()

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __len__(self):
        return len(self.batch_sampler)

    def jag_stack(self, b):
        if len(b[0].shape) not in (1, 2):
            return np.stack(b)
        ml = max(len(o) for o in b)
        if min(len(o) for o in b) == ml:
            return np.stack(b)
        res = np.zeros((len(b), ml), dtype=b[0].dtype) + self.pad_idx
        for i, o in enumerate(b):
            if self.pre_pad:
                res[i, -len(o):] = o
            else:
                res[i, :len(o)] = o
        return res

    def np_collate(self, batch):
        b = batch[0]
        if isinstance(b, (np.ndarray, np.generic)):
            return self.jag_stack(batch)
        elif isinstance(b, (int, float)):
            return np.array(batch)
        elif isinstance(b, string_classes):
            return batch
        elif isinstance(b, collections.Mapping):
            return {key: self.np_collate([d[key] for d in batch]) for key in b}
        elif isinstance(b, collections.Sequence):
            return [self.np_collate(samples) for samples in zip(*batch)]
        raise TypeError(("batch must contain numbers, dicts or lists; found {}".format(type(b))))

    def get_batch(self, indices):
        res = self.np_collate([self.dataset[i] for i in indices])
        if self.transpose:
            res[0] = res[0].T
        if self.transpose_y:
            res[1] = res[1].T
        return res

    def __iter__(self):
        if self.num_workers == 0:
            for batch in map(self.get_batch, iter(self.batch_sampler)):
                yield get_tensor(batch, self.pin_memory, self.half)
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as e:
                # avoid py3.6 issue where queue is infinite and can result in memory exhaustion
                for c in chunk_iter(iter(self.batch_sampler), self.num_workers * 10):
                    for batch in e.map(self.get_batch, c):
                        yield get_tensor(batch, self.pin_memory, self.half)


class BaseDataset(Dataset):
    """An abstract class representing a fastai dataset, it extends torch.utils.data.Dataset."""

    def __init__(self, transform=None):
        self.transform = transform
        self.n = self.get_n()
        self.c = self.get_c()
        self.sz = self.get_sz()

    def get1item(self, idx):
        x, y = self.get_x(idx), self.get_y(idx)
        return self.get(self.transform, x, y)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            xs, ys = zip(*[self.get1item(i) for i in range(*idx.indices(self.n))])
            return np.stack(xs), ys
        return self.get1item(idx)

    def __len__(self):
        return self.n

    def get(self, tfm, x, y):
        return (x, y) if tfm is None else tfm(x, y)

    @abstractmethod
    def get_n(self):
        """Return number of elements in the dataset == len(self)."""
        raise NotImplementedError

    @abstractmethod
    def get_c(self):
        """Return number of classes in a dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_sz(self):
        """Return maximum size of an image in a dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_x(self, i):
        """Return i-th example (image, wav, etc)."""
        raise NotImplementedError

    @abstractmethod
    def get_y(self, i):
        """Return i-th label."""
        raise NotImplementedError

    @property
    def is_multi(self):
        """Returns true if this data set contains multiple labels per sample."""
        return False

    @property
    def is_reg(self):
        """True if the data set is used to train regression models."""
        return False


class ArraysDataset(BaseDataset):
    def __init__(self, x, y, transform):
        self.x, self.y = x, y
        assert (len(x) == len(y))
        super().__init__(transform)

    def get_x(self, i): return self.x[i]

    def get_y(self, i): return self.y[i]

    def get_n(self): return len(self.y)

    def get_sz(self): return self.x.shape[1]


class FilesDataset(BaseDataset):
    def __init__(self, fnames, transform, path):
        self.path, self.fnames = path, fnames
        super().__init__(transform)

    def get_sz(self):
        return self.transform.sz

    def get_x(self, i):
        return open_image(os.path.join(self.path, self.fnames[i]))

    def get_n(self):
        return len(self.fnames)

    def resize_imgs(self, targ, new_path):
        dest = resize_imgs(self.fnames, targ, self.path, new_path)
        return self.__class__(self.fnames, self.y, self.transform, dest)

    def denorm(self, arr):
        """Reverse the normalization done to a batch of images.
        Arguments:
            arr: of shape/size (N,3,sz,sz)
        """
        if type(arr) is not np.ndarray:
            arr = to_np(arr)
        if len(arr.shape) == 3:
            arr = arr[None]
        return self.transform.denorm(np.rollaxis(arr, 1, 4))


class ArraysIndexDataset(ArraysDataset):
    def get_c(self): return int(self.y.max()) + 1

    def get_y(self, i): return self.y[i]


class FilesArrayDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y = y
        assert (len(fnames) == len(y))
        super().__init__(fnames, transform, path)

    def get_y(self, i): return self.y[i]

    def get_c(self):
        return self.y.shape[1] if len(self.y.shape) > 1 else 0


class FilesIndexArrayDataset(FilesArrayDataset):
    def get_c(self): return int(self.y.max()) + 1


class FilesIndexArrayRegressionDataset(FilesArrayDataset):
    def is_reg(self): return True


class FilesNhotArrayDataset(FilesArrayDataset):
    @property
    def is_multi(self): return True


class Learner(object):
    def __init__(self, data, models, opt_fn=None, tmp_name='tmp', models_name='models', metrics=None, clip=None,
                 crit=None):
        """
        Combines a ModelData object with a nn.Module object, such that you can train that
        module.
        data (ModelData): An instance of ModelData.
        models(module): chosen neural architecture for solving a supported problem.
        opt_fn(function): optimizer function, uses SGD with Momentum of .9 if none.
        tmp_name(str): output name of the directory containing temporary files from training process
        models_name(str): output name of the directory containing the trained model
        metrics(list): array of functions for evaluating a desired metric. Eg. accuracy.
        clip(float): gradient clip chosen to limit the change in the gradient to prevent exploding gradients Eg. .3
        """
        self.data_, self.models, self.metrics = data, models, metrics
        self.sched = None
        self.wd_sched = None
        self.clip = None
        self.opt_fn = opt_fn or SGD_Momentum(0.9)
        self.tmp_path = tmp_name if os.path.isabs(tmp_name) else os.path.join(self.data.path, tmp_name)
        self.models_path = models_name if os.path.isabs(models_name) else os.path.join(self.data.path, models_name)
        os.makedirs(self.tmp_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        self.crit = crit if crit else self._get_crit(data)
        self.reg_fn = None
        self.fp16 = False

    @classmethod
    def from_model_data(cls, m, data, **kwargs):
        self = cls(data, BasicModel(to_gpu(m)), **kwargs)
        self.unfreeze()
        return self

    def __getitem__(self, i):
        return self.children[i]

    @property
    def children(self):
        return children(self.model)

    @property
    def model(self):
        return self.models.model

    @property
    def data(self):
        return self.data_

    def summary(self):
        return model_summary(self.model, [3, self.data.sz, self.data.sz])

    def __repr__(self):
        return self.model.__repr__()

    def lsuv_init(self, needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=False):
        x = V(next(iter(self.data.trn_dl))[0])
        self.models.model = apply_lsuv_init(self.model, x, needed_std=needed_std, std_tol=std_tol,
                                            max_attempts=max_attempts, do_orthonorm=do_orthonorm,
                                            cuda=USE_GPU and torch.cuda.is_available())

    def set_bn_freeze(self, m, do_freeze):
        if hasattr(m, 'running_mean'):
            m.bn_freeze = do_freeze

    def bn_freeze(self, do_freeze):
        apply_leaf(self.model, lambda m: self.set_bn_freeze(m, do_freeze))

    def freeze_to(self, n):
        c = self.get_layer_groups()
        for l in c:
            set_trainable(l, False)
        for l in c[n:]:
            set_trainable(l, True)

    def freeze_all_but(self, n):
        c = self.get_layer_groups()
        for l in c:
            set_trainable(l, False)
        set_trainable(c[n], True)

    def unfreeze(self):
        self.freeze_to(0)

    def get_model_path(self, name):
        return os.path.join(self.models_path, name) + '.h5'

    def save(self, name):
        save_model(self.model, self.get_model_path(name))
        if hasattr(self, 'swa_model'):
            save_model(self.swa_model, self.get_model_path(name)[:-3] + '-swa.h5')

    def load(self, name):
        load_model(self.model, self.get_model_path(name))
        if hasattr(self, 'swa_model'):
            load_model(self.swa_model, self.get_model_path(name)[:-3] + '-swa.h5')

    def set_data(self, data):
        self.data_ = data

    def get_cycle_end(self, name):
        if name is None:
            return None
        return lambda sched, cycle: self.save_cycle(name, cycle)

    def save_cycle(self, name, cycle):
        self.save(f'{name}_cyc_{cycle}')

    def load_cycle(self, name, cycle):
        self.load(f'{name}_cyc_{cycle}')

    def half(self):
        if self.fp16:
            return
        self.fp16 = True
        if type(self.model) != FP16:
            self.models.model = FP16(self.model)

    def float(self):
        if not self.fp16:
            return
        self.fp16 = False
        if type(self.model) == FP16:
            self.models.model = self.model.module
        self.model.float()

    def fit_gen(self, model, data, layer_opt, n_cycle, cycle_len=None, cycle_mult=1, cycle_save_name=None,
                best_save_name=None,
                use_clr=None, use_clr_beta=None, metrics=None, callbacks=None, use_wd_sched=False, norm_wds=False,
                wds_sched_mult=None, use_swa=False, swa_start=1, swa_eval_freq=5, **kwargs):

        """Method does some preparation before finally delegating to the 'fit' method for
        fitting the model. Namely, if cycle_len is defined, it adds a 'Cosine Annealing'
        scheduler for varying the learning rate across iterations.

        Method also computes the total number of epochs to fit based on provided 'cycle_len',
        'cycle_mult', and 'n_cycle' parameters.

        Args:
            model (Learner):  Any neural architecture for solving a supported problem.
                Eg. ResNet-34, RNN_Learner etc.

            data (ModelData): An instance of ModelData.

            layer_opt (LayerOptimizer): An instance of the LayerOptimizer class

            n_cycle (int): number of cycles

            cycle_len (int):  number of cycles before lr is reset to the initial value.
                E.g if cycle_len = 3, then the lr is varied between a maximum
                and minimum value over 3 epochs.

            cycle_mult (int): additional parameter for influencing how the lr resets over
                the cycles. For an intuitive explanation, please see
                https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb

            cycle_save_name (str): use to save the weights at end of each cycle

            best_save_name (str): use to save weights of best model during training.

            metrics (function): some function for evaluating a desired metric. Eg. accuracy.

            callbacks (list(Callback)): callbacks to apply during the training.

            use_wd_sched (bool, optional): set to True to enable weight regularization using
                the technique mentioned in https://arxiv.org/abs/1711.05101. When this is True
                alone (see below), the regularization is detached from gradient update and
                applied directly to the weights.

            norm_wds (bool, optional): when this is set to True along with use_wd_sched, the
                regularization factor is normalized with each training cycle.

            wds_sched_mult (function, optional): when this is provided along with use_wd_sched
                as True, the value computed by this function is multiplied with the regularization
                strength. This function is passed the WeightDecaySchedule object. And example
                function that can be passed is:
                            f = lambda x: np.array(x.layer_opt.lrs) / x.init_lrs

            use_swa (bool, optional): when this is set to True, it will enable the use of
                Stochastic Weight Averaging (https://arxiv.org/abs/1803.05407). The learner will
                include an additional model (in the swa_model attribute) for keeping track of the
                average weights as described in the paper. All testing of this technique so far has
                been in image classification, so use in other contexts is not guaranteed to work.

            swa_start (int, optional): if use_swa is set to True, then this determines the epoch
                to start keeping track of the average weights. It is 1-indexed per the paper's
                conventions.

            swa_eval_freq (int, optional): if use_swa is set to True, this determines the frequency
                at which to evaluate the performance of the swa_model. This evaluation can be costly
                for models using BatchNorm (requiring a full pass through the data), which is why the
                default is not to evaluate after each epoch.

        Returns:
            None
        """

        if callbacks is None:
            callbacks = []
        if metrics is None:
            metrics = self.metrics

        if use_wd_sched:
            # This needs to come before CosAnneal() because we need to read the initial learning rate from
            # layer_opt.lrs - but CosAnneal() alters the layer_opt.lrs value initially (divides by 100)
            if np.sum(layer_opt.wds) == 0:
                print('fit() warning: use_wd_sched is set to True, but weight decay(s) passed are 0. Use wds to '
                      'pass weight decay values.')
            batch_per_epoch = len(data.trn_dl)
            cl = cycle_len if cycle_len else 1
            self.wd_sched = WeightDecaySchedule(layer_opt, batch_per_epoch, cl, cycle_mult, n_cycle,
                                                norm_wds, wds_sched_mult)
            callbacks += [self.wd_sched]

        if use_clr is not None:
            clr_div, cut_div = use_clr[:2]
            moms = use_clr[2:] if len(use_clr) > 2 else None
            cycle_end = self.get_cycle_end(cycle_save_name)
            self.sched = CircularLR(layer_opt, len(data.trn_dl) * cycle_len, on_cycle_end=cycle_end, div=clr_div,
                                    cut_div=cut_div,
                                    momentums=moms)
        elif use_clr_beta is not None:
            div, pct = use_clr_beta[:2]
            moms = use_clr_beta[2:] if len(use_clr_beta) > 3 else None
            cycle_end = self.get_cycle_end(cycle_save_name)
            self.sched = CircularLR_beta(layer_opt, len(data.trn_dl) * cycle_len, on_cycle_end=cycle_end, div=div,
                                         pct=pct, momentums=moms)
        elif cycle_len:
            cycle_end = self.get_cycle_end(cycle_save_name)
            cycle_batches = len(data.trn_dl) * cycle_len
            self.sched = CosAnneal(layer_opt, cycle_batches, on_cycle_end=cycle_end, cycle_mult=cycle_mult)
        elif not self.sched:
            self.sched = LossRecorder(layer_opt)
        callbacks += [self.sched]

        if best_save_name is not None:
            callbacks += [SaveBestModel(self, layer_opt, metrics, best_save_name)]

        if use_swa:
            # make a copy of the model to track average weights
            self.swa_model = copy.deepcopy(model)
            callbacks += [SWA(model, self.swa_model, swa_start)]

        n_epoch = int(sum_geom(cycle_len if cycle_len else 1, cycle_mult, n_cycle))
        return fit(model, data, n_epoch, layer_opt.opt, self.crit,
                   metrics=metrics, callbacks=callbacks, reg_fn=self.reg_fn, clip=self.clip, fp16=self.fp16,
                   swa_model=self.swa_model if use_swa else None, swa_start=swa_start,
                   swa_eval_freq=swa_eval_freq, **kwargs)

    def get_layer_groups(self):
        return self.models.get_layer_groups()

    def get_layer_opt(self, lrs, wds):

        """Method returns an instance of the LayerOptimizer class, which
        allows for setting differential learning rates for different
        parts of the model.

        An example of how a model maybe differentiated into different parts
        for application of differential learning rates and weight decays is
        seen in ../.../courses/dl1/fastai/conv_learner.py, using the dict
        'model_meta'. Currently, this seems supported only for convolutional
        networks such as VGG-19, ResNet-XX etc.

        Args:
            lrs (float or list(float)): learning rate(s) for the model

            wds (float or list(float)): weight decay parameter(s).

        Returns:
            An instance of a LayerOptimizer
        """
        return LayerOptimizer(self.opt_fn, self.get_layer_groups(), lrs, wds)

    def fit(self, lrs, n_cycle, wds=None, **kwargs):

        """Method gets an instance of LayerOptimizer and delegates to self.fit_gen(..)

        Note that one can specify a list of learning rates which, when appropriately
        defined, will be applied to different segments of an architecture. This seems
        mostly relevant to ImageNet-trained models, where we want to alter the layers
        closest to the images by much smaller amounts.

        Likewise, a single or list of weight decay parameters can be specified, which
        if appropriate for a model, will apply variable weight decay parameters to
        different segments of the model.

        Args:
            lrs (float or list(float)): learning rate for the model

            n_cycle (int): number of cycles (or iterations) to fit the model for

            wds (float or list(float)): weight decay parameter(s).

            kwargs: other arguments

        Returns:
            None
        """
        self.sched = None
        layer_opt = self.get_layer_opt(lrs, wds)
        return self.fit_gen(self.model, self.data, layer_opt, n_cycle, **kwargs)

    def warm_up(self, lr, wds=None):
        layer_opt = self.get_layer_opt(lr / 4, wds)
        self.sched = LR_Finder(layer_opt, len(self.data.trn_dl), lr, linear=True)
        return self.fit_gen(self.model, self.data, layer_opt, 1)

    def lr_find(self, start_lr=1e-5, end_lr=10, wds=None, linear=False, **kwargs):
        """Helps you find an optimal learning rate for a model.

         It uses the technique developed in the 2015 paper
         `Cyclical Learning Rates for Training Neural Networks`, where
         we simply keep increasing the learning rate from a very small value,
         until the loss starts decreasing.

        Args:
            start_lr (float/numpy array) : Passing in a numpy array allows you
                to specify learning rates for a learner's layer_groups
            end_lr (float) : The maximum learning rate to try.
            wds (iterable/float)

        Examples:
            As training moves us closer to the optimal weights for a model,
            the optimal learning rate will be smaller. We can take advantage of
            that knowledge and provide lr_find() with a starting learning rate
            1000x smaller than the model's current learning rate as such:

            >> learn.lr_find(lr/1000)

            >> lrs = np.array([ 1e-4, 1e-3, 1e-2 ])
            >> learn.lr_find(lrs / 1000)

        Notes:
            lr_find() may finish before going through each batch of examples if
            the loss decreases enough.

        .. _Cyclical Learning Rates for Training Neural Networks:
            http://arxiv.org/abs/1506.01186

        """
        self.save('tmp')
        layer_opt = self.get_layer_opt(start_lr, wds)
        self.sched = LR_Finder(layer_opt, len(self.data.trn_dl), end_lr, linear=linear)
        self.fit_gen(self.model, self.data, layer_opt, 1, **kwargs)
        self.load('tmp')

    def lr_find2(self, start_lr=1e-5, end_lr=10, num_it=100, wds=None, linear=False, stop_dv=True, **kwargs):
        """A variant of lr_find() that helps find the best learning rate. It doesn't do
        an epoch but a fixed num of iterations (which may be more or less than an epoch
        depending on your data).
        At each step, it computes the validation loss and the metrics on the next
        batch of the validation data, so it's slower than lr_find().

        Args:
            start_lr (float/numpy array) : Passing in a numpy array allows you
                to specify learning rates for a learner's layer_groups
            end_lr (float) : The maximum learning rate to try.
            num_it : the number of iterations you want it to run
            wds (iterable/float)
            stop_dv : stops (or not) when the losses starts to explode.
        """
        self.save('tmp')
        layer_opt = self.get_layer_opt(start_lr, wds)
        self.sched = LR_Finder2(layer_opt, num_it, end_lr, linear=linear, metrics=self.metrics, stop_dv=stop_dv)
        self.fit_gen(self.model, self.data, layer_opt, num_it // len(self.data.trn_dl) + 1, all_val=True, **kwargs)
        self.load('tmp')

    def predict(self, is_test=False, use_swa=False):
        dl = self.data.test_dl if is_test else self.data.val_dl
        m = self.swa_model if use_swa else self.model
        return predict(m, dl)

    def predict_with_targs(self, is_test=False, use_swa=False):
        dl = self.data.test_dl if is_test else self.data.val_dl
        m = self.swa_model if use_swa else self.model
        return predict_with_targs(m, dl)

    def predict_dl(self, dl):
        return predict_with_targs(self.model, dl)[0]

    def predict_array(self, arr):
        self.model.eval()
        return to_np(self.model(to_gpu(V(T(arr)))))

    def TTA(self, n_aug=4, is_test=False):
        """ Predict with Test Time Augmentation (TTA)

        Additional to the original test/validation images, apply image augmentation to them
        (just like for training images) and calculate the mean of predictions. The intent
        is to increase the accuracy of predictions by examining the images using multiple
        perspectives.

        Args:
            n_aug: a number of augmentation images to use per original image
            is_test: indicate to use test images; otherwise use validation images

        Returns:
            (tuple): a tuple containing:

                log predictions (numpy.ndarray): log predictions (i.e. `np.exp(log_preds)` will return probabilities)
                targs (numpy.ndarray): target values when `is_test==False`; zeros otherwise.
        """
        dl1 = self.data.test_dl if is_test else self.data.val_dl
        dl2 = self.data.test_aug_dl if is_test else self.data.aug_dl
        preds1, targs = predict_with_targs(self.model, dl1)
        preds1 = [preds1] * math.ceil(n_aug / 4)
        preds2 = [predict_with_targs(self.model, dl2)[0] for i in tqdm(range(n_aug), leave=False)]
        return np.stack(preds1 + preds2), targs

    def fit_opt_sched(self, phases, cycle_save_name=None, best_save_name=None, stop_div=False, data_list=None,
                      callbacks=None,
                      cut=None, use_swa=False, swa_start=1, swa_eval_freq=5, **kwargs):
        """Wraps us the content of phases to send them to model.fit(..)

        This will split the training in several parts, each with their own learning rates/
        wds/momentums/optimizer detailed in phases.

        Additionaly we can add a list of different data objets in data_list to train
        on different datasets (to change the size for instance) for each of these groups.

        Args:
            phases: a list of TrainingPhase objects
            stop_div: when True, stops the training if the loss goes too high
            data_list: a list of different Data objects.
            kwargs: other arguments
            use_swa (bool, optional): when this is set to True, it will enable the use of
                Stochastic Weight Averaging (https://arxiv.org/abs/1803.05407). The learner will
                include an additional model (in the swa_model attribute) for keeping track of the
                average weights as described in the paper. All testing of this technique so far has
                been in image classification, so use in other contexts is not guaranteed to work.
            swa_start (int, optional): if use_swa is set to True, then this determines the epoch
                to start keeping track of the average weights. It is 1-indexed per the paper's
                conventions.
            swa_eval_freq (int, optional): if use_swa is set to True, this determines the frequency
                at which to evaluate the performance of the swa_model. This evaluation can be costly
                for models using BatchNorm (requiring a full pass through the data), which is why the
                default is not to evaluate after each epoch.
        Returns:
            None
        """
        if data_list is None:
            data_list = []
        if callbacks is None:
            callbacks = []
        layer_opt = LayerOptimizer(phases[0].opt_fn, self.get_layer_groups(), 1e-2, phases[0].wds)
        if len(data_list) == 0:
            nb_batches = [len(self.data.trn_dl)] * len(phases)
        else:
            nb_batches = [len(data.trn_dl) for data in data_list]
        self.sched = OptimScheduler(layer_opt, phases, nb_batches, stop_div)
        callbacks.append(self.sched)
        metrics = self.metrics
        if best_save_name is not None:
            callbacks += [SaveBestModel(self, layer_opt, metrics, best_save_name)]
        if use_swa:
            # make a copy of the model to track average weights
            self.swa_model = copy.deepcopy(self.model)
            callbacks += [SWA(self.model, self.swa_model, swa_start)]
        n_epochs = [phase.epochs for phase in phases] if cut is None else cut
        if len(data_list) == 0:
            data_list = [self.data]
        return fit(self.model, data_list, n_epochs, layer_opt, self.crit,
                   metrics=metrics, callbacks=callbacks, reg_fn=self.reg_fn, clip=self.clip, fp16=self.fp16,
                   swa_model=self.swa_model if use_swa else None, swa_start=swa_start,
                   swa_eval_freq=swa_eval_freq, **kwargs)

    def _get_crit(self, data):
        return F.mse_loss


class Callback:
    """
    An abstract class that all callback(e.g., LossRecorder) classes extends from.
    Must be extended before usage.
    """

    def on_train_begin(self): pass

    def on_batch_begin(self): pass

    def on_phase_begin(self): pass

    def on_epoch_end(self, metrics): pass

    def on_phase_end(self): pass

    def on_batch_end(self, metrics): pass

    def on_train_end(self): pass


class WeightDecaySchedule(Callback):
    def __init__(self, layer_opt, batch_per_epoch, cycle_len, cycle_mult, n_cycles, norm_wds=False,
                 wds_sched_mult=None):
        """
        Implements the weight decay schedule as mentioned in https://arxiv.org/abs/1711.05101

        :param layer_opt: The LayerOptimizer
        :param batch_per_epoch: Num batches in 1 epoch
        :param cycle_len: Num epochs in initial cycle. Subsequent cycle_len = previous cycle_len * cycle_mult
        :param cycle_mult: Cycle multiplier
        :param n_cycles: Number of cycles to be executed
        """
        super().__init__()

        self.layer_opt = layer_opt
        self.batch_per_epoch = batch_per_epoch
        self.init_wds = np.array(layer_opt.wds)  # Weights as set by user
        self.init_lrs = np.array(layer_opt.lrs)  # Learning rates as set by user
        self.new_wds = None  # Holds the new weight decay factors, calculated in on_batch_begin()
        self.param_groups_old = None  # Caches the old parameter values in on_batch_begin()
        self.iteration = 0
        self.epoch = 0
        self.wds_sched_mult = wds_sched_mult
        self.norm_wds = norm_wds
        self.wds_history = list()

        # Pre calculating the number of epochs in the cycle of current running epoch
        self.epoch_to_num_cycles, i = dict(), 0
        for cycle in range(n_cycles):
            for _ in range(cycle_len):
                self.epoch_to_num_cycles[i] = cycle_len
                i += 1
            cycle_len *= cycle_mult

    def on_train_begin(self):
        self.iteration = 0
        self.epoch = 0

    def on_batch_begin(self):
        # Prepare for decay of weights

        # Default weight decay (as provided by user)
        wdn = self.init_wds

        # Weight decay multiplier (The 'eta' in the paper). Optional.
        wdm = 1.0
        if self.wds_sched_mult is not None:
            wdm = self.wds_sched_mult(self)

        # Weight decay normalized. Optional.
        if self.norm_wds:
            wdn = wdn / np.sqrt(self.batch_per_epoch * self.epoch_to_num_cycles[self.epoch])

        # Final wds
        self.new_wds = wdm * wdn

        # Record the wds
        self.wds_history.append(self.new_wds)

        # Set weight_decay with zeros so that it is not applied in Adam, we will apply it outside in on_batch_end()
        self.layer_opt.set_wds(torch.zeros(self.new_wds.size))
        # We have to save the existing weights before the optimizer changes the values
        self.param_groups_old = copy.deepcopy(self.layer_opt.opt.param_groups)
        self.iteration += 1

    def on_batch_end(self, loss):
        # Decay the weights
        for group, group_old, wds in zip(self.layer_opt.opt.param_groups, self.param_groups_old, self.new_wds):
            for p, p_old in zip(group['params'], group_old['params']):
                if p.grad is None:
                    continue
                p.data = p.data.add(-wds, p_old.data)

    def on_epoch_end(self, metrics):
        self.epoch += 1


class LossRecorder(Callback):
    """
    Saves and displays loss functions and other metrics.
    Default sched when none is specified in a learner.
    """

    def __init__(self, layer_opt, save_path='', record_mom=False, metrics=[]):
        super().__init__()
        self.layer_opt = layer_opt
        self.init_lrs = np.array(layer_opt.lrs)
        self.save_path, self.record_mom, self.metrics = save_path, record_mom, metrics

    def on_train_begin(self):
        self.losses, self.lrs, self.iterations = [], [], []
        self.val_losses, self.rec_metrics = [], []
        if self.record_mom:
            self.momentums = []
        self.iteration = 0
        self.epoch = 0

    def on_epoch_end(self, metrics):
        self.epoch += 1
        self.save_metrics(metrics)

    def on_batch_end(self, loss):
        self.iteration += 1
        self.lrs.append(self.layer_opt.lr)
        self.iterations.append(self.iteration)
        if isinstance(loss, list):
            self.losses.append(loss[0])
            self.save_metrics(loss[1:])
        else:
            self.losses.append(loss)
        if self.record_mom: self.momentums.append(self.layer_opt.mom)

    def save_metrics(self, vals):
        self.val_losses.append(vals[0][0] if isinstance(vals[0], Iterable) else vals[0])
        if len(vals) > 2:
            self.rec_metrics.append(vals[1:])
        elif len(vals) == 2:
            self.rec_metrics.append(vals[1])

    def plot_loss(self, n_skip=10, n_skip_end=5):
        """
        plots loss function as function of iterations.
        When used in Jupyternotebook, plot will be displayed in notebook. Else, plot will be displayed in console and both plot and loss are saved in save_path.
        """
        if not in_ipynb(): plt.switch_backend('agg')
        plt.plot(self.iterations[n_skip:-n_skip_end], self.losses[n_skip:-n_skip_end])
        if not in_ipynb():
            plt.savefig(os.path.join(self.save_path, 'loss_plot.png'))
            np.save(os.path.join(self.save_path, 'losses.npy'), self.losses[10:])

    def plot_lr(self):
        """Plots learning rate in jupyter notebook or console, depending on the enviroment of the learner."""
        if not in_ipynb():
            plt.switch_backend('agg')
        if self.record_mom:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            for i in range(0, 2): axs[i].set_xlabel('iterations')
            axs[0].set_ylabel('learning rate')
            axs[1].set_ylabel('momentum')
            axs[0].plot(self.iterations, self.lrs)
            axs[1].plot(self.iterations, self.momentums)
        else:
            plt.xlabel("iterations")
            plt.ylabel("learning rate")
            plt.plot(self.iterations, self.lrs)
        if not in_ipynb():
            plt.savefig(os.path.join(self.save_path, 'lr_plot.png'))


class LR_Updater(LossRecorder):
    """
    Abstract class where all Learning Rate updaters inherit from. (e.g., CirularLR)
    Calculates and updates new learning rate and momentum at the end of each batch.
    Have to be extended.
    """

    def on_train_begin(self):
        super().on_train_begin()
        self.update_lr()
        if self.record_mom:
            self.update_mom()

    def on_batch_end(self, loss):
        res = super().on_batch_end(loss)
        self.update_lr()
        if self.record_mom:
            self.update_mom()
        return res

    def update_lr(self):
        new_lrs = self.calc_lr(self.init_lrs)
        self.layer_opt.set_lrs(new_lrs)

    def update_mom(self):
        new_mom = self.calc_mom()
        self.layer_opt.set_mom(new_mom)

    @abstractmethod
    def calc_lr(self, init_lrs):
        raise NotImplementedError

    @abstractmethod
    def calc_mom(self):
        raise NotImplementedError


class CircularLR(LR_Updater):
    """
    An learning rate updater that implements the CirularLearningRate (CLR) scheme.
    Learning rate is increased then decreased linearly.
    """

    def __init__(self, layer_opt, nb, div=4, cut_div=8, on_cycle_end=None, momentums=None):
        self.nb, self.div, self.cut_div, self.on_cycle_end = nb, div, cut_div, on_cycle_end
        if momentums is not None:
            self.moms = momentums
        super().__init__(layer_opt, record_mom=(momentums is not None))

    def on_train_begin(self):
        self.cycle_iter, self.cycle_count = 0, 0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        cut_pt = self.nb // self.cut_div
        if self.cycle_iter > cut_pt:
            pct = 1 - (self.cycle_iter - cut_pt) / (self.nb - cut_pt)
        else:
            pct = self.cycle_iter / cut_pt
        res = init_lrs * (1 + pct * (self.div - 1)) / self.div
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res

    def calc_mom(self):
        cut_pt = self.nb // self.cut_div
        if self.cycle_iter > cut_pt:
            pct = (self.cycle_iter - cut_pt) / (self.nb - cut_pt)
        else:
            pct = 1 - self.cycle_iter / cut_pt
        res = self.moms[1] + pct * (self.moms[0] - self.moms[1])
        return res


class CircularLR_beta(LR_Updater):
    def __init__(self, layer_opt, nb, div=10, pct=10, on_cycle_end=None, momentums=None):
        self.nb, self.div, self.pct, self.on_cycle_end = nb, div, pct, on_cycle_end
        self.cycle_nb = int(nb * (1 - pct / 100) / 2)
        if momentums is not None:
            self.moms = momentums
        super().__init__(layer_opt, record_mom=(momentums is not None))

    def on_train_begin(self):
        self.cycle_iter, self.cycle_count = 0, 0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        if self.cycle_iter > 2 * self.cycle_nb:
            pct = (self.cycle_iter - 2 * self.cycle_nb) / (self.nb - 2 * self.cycle_nb)
            res = init_lrs * (1 + (pct * (1 - 100) / 100)) / self.div
        elif self.cycle_iter > self.cycle_nb:
            pct = 1 - (self.cycle_iter - self.cycle_nb) / self.cycle_nb
            res = init_lrs * (1 + pct * (self.div - 1)) / self.div
        else:
            pct = self.cycle_iter / self.cycle_nb
            res = init_lrs * (1 + pct * (self.div - 1)) / self.div
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res

    def calc_mom(self):
        if self.cycle_iter > 2 * self.cycle_nb:
            res = self.moms[0]
        elif self.cycle_iter > self.cycle_nb:
            pct = 1 - (self.cycle_iter - self.cycle_nb) / self.cycle_nb
            res = self.moms[0] + pct * (self.moms[1] - self.moms[0])
        else:
            pct = self.cycle_iter / self.cycle_nb
            res = self.moms[0] + pct * (self.moms[1] - self.moms[0])
        return res


class BasicModel(object):
    def __init__(self, model, name='unnamed'): self.model, self.name = model, name

    def get_layer_groups(self, do_fc=False): return children(self.model)


class ConvnetBuilder():
    """Class representing a convolutional network.

    Arguments:
        f: a model creation function (e.g. resnet34, vgg16, etc)
        c (int): size of the last layer
        is_multi (bool): is multilabel classification?
            (def here http://scikit-learn.org/stable/modules/multiclass.html)
        is_reg (bool): is a regression?
        ps (float or array of float): dropout parameters
        xtra_fc (list of ints): list of hidden layers with # hidden neurons
        xtra_cut (int): # layers earlier than default to cut the model, default is 0
        custom_head : add custom model classes that are inherited from nn.modules at the end of the model
                      that is mentioned on Argument 'f'
    """

    def __init__(self, f, c, is_multi, is_reg, ps=None, xtra_fc=None, xtra_cut=0, custom_head=None, pretrained=True):
        self.f, self.c, self.is_multi, self.is_reg, self.xtra_cut = f, c, is_multi, is_reg, xtra_cut
        if xtra_fc is None: xtra_fc = [512]
        if ps is None: ps = [0.25] * len(xtra_fc) + [0.5]
        self.ps, self.xtra_fc = ps, xtra_fc

        if f in model_meta:
            cut, self.lr_cut = model_meta[f]
        else:
            cut, self.lr_cut = 0, 0
        cut -= xtra_cut
        layers = cut_model(f(pretrained), cut)
        self.nf = model_features[f] if f in model_features else (num_features(layers) * 2)
        if not custom_head: layers += [AdaptiveConcatPool2d(), Flatten()]
        self.top_model = nn.Sequential(*layers)

        n_fc = len(self.xtra_fc) + 1
        if not isinstance(self.ps, list): self.ps = [self.ps] * n_fc

        if custom_head:
            fc_layers = [custom_head]
        else:
            fc_layers = self.get_fc_layers()
        self.n_fc = len(fc_layers)
        self.fc_model = to_gpu(nn.Sequential(*fc_layers))
        if not custom_head: apply_init(self.fc_model, kaiming_normal)
        self.model = to_gpu(nn.Sequential(*(layers + fc_layers)))

    @property
    def name(self):
        return f'{self.f.__name__}_{self.xtra_cut}'

    def create_fc_layer(self, ni, nf, p, actn=None):
        res = [nn.BatchNorm1d(num_features=ni)]
        if p: res.append(nn.Dropout(p=p))
        res.append(nn.Linear(in_features=ni, out_features=nf))
        if actn: res.append(actn)
        return res

    def get_fc_layers(self):
        res = []
        ni = self.nf
        for i, nf in enumerate(self.xtra_fc):
            res += self.create_fc_layer(ni, nf, p=self.ps[i], actn=nn.ReLU())
            ni = nf
        final_actn = nn.Sigmoid() if self.is_multi else nn.LogSoftmax()
        if self.is_reg: final_actn = None
        res += self.create_fc_layer(ni, self.c, p=self.ps[-1], actn=final_actn)
        return res

    def get_layer_groups(self, do_fc=False):
        if do_fc:
            return [self.fc_model]
        idxs = [self.lr_cut]
        c = children(self.top_model)
        if len(c) == 3: c = children(c[0]) + c[1:]
        lgs = list(split_by_idxs(c, idxs))
        return lgs + [self.fc_model]


class ConvLearner(Learner):
    """
    Class used to train a chosen supported covnet model. Eg. ResNet-34, etc.
    Arguments:
        data: training data for model
        models: model architectures to base learner
        precompute: bool to reuse precomputed activations
        **kwargs: parameters from Learner() class
    """

    def __init__(self, data, models, precompute=False, **kwargs):
        self.precompute = False
        super().__init__(data, models, **kwargs)
        if hasattr(data, 'is_multi') and not data.is_reg and self.metrics is None:
            self.metrics = [accuracy_thresh(0.5)] if self.data.is_multi else [accuracy]
        if precompute: self.save_fc1()
        self.freeze()
        self.precompute = precompute

    def _get_crit(self, data):
        if not hasattr(data, 'is_multi'): return super()._get_crit(data)

        return F.l1_loss if data.is_reg else F.binary_cross_entropy if data.is_multi else F.nll_loss

    @classmethod
    def pretrained(cls, f, data, ps=None, xtra_fc=None, xtra_cut=0, custom_head=None, precompute=False,
                   pretrained=True, **kwargs):
        models = ConvnetBuilder(f, data.c, data.is_multi, data.is_reg,
                                ps=ps, xtra_fc=xtra_fc, xtra_cut=xtra_cut, custom_head=custom_head,
                                pretrained=pretrained)
        return cls(data, models, precompute, **kwargs)

    @classmethod
    def lsuv_learner(cls, f, data, ps=None, xtra_fc=None, xtra_cut=0, custom_head=None, precompute=False,
                     needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=False, **kwargs):
        models = ConvnetBuilder(f, data.c, data.is_multi, data.is_reg,
                                ps=ps, xtra_fc=xtra_fc, xtra_cut=xtra_cut, custom_head=custom_head, pretrained=False)
        convlearn = cls(data, models, precompute, **kwargs)
        convlearn.lsuv_init()
        return convlearn

    @property
    def model(self):
        return self.models.fc_model if self.precompute else self.models.model

    @property
    def data(self):
        return self.fc_data if self.precompute else self.data_

    def create_empty_bcolz(self, n, name):
        return bcolz.carray(np.zeros((0, n), np.float32), chunklen=1, mode='w', rootdir=name)

    def set_data(self, data, precompute=False):
        super().set_data(data)
        if precompute:
            self.unfreeze()
            self.save_fc1()
            self.freeze()
            self.precompute = True
        else:
            self.freeze()

    def get_layer_groups(self):
        return self.models.get_layer_groups(self.precompute)

    def summary(self):
        precompute = self.precompute
        self.precompute = False
        res = super().summary()
        self.precompute = precompute
        return res

    def get_activations(self, force=False):
        tmpl = f'_{self.models.name}_{self.data.sz}.bc'
        # TODO: Somehow check that directory names haven't changed (e.g. added test set)
        names = [os.path.join(self.tmp_path, p + tmpl) for p in ('x_act', 'x_act_val', 'x_act_test')]
        if os.path.exists(names[0]) and not force:
            self.activations = [bcolz.open(p) for p in names]
        else:
            self.activations = [self.create_empty_bcolz(self.models.nf, n) for n in names]

    def save_fc1(self):
        self.get_activations()
        act, val_act, test_act = self.activations
        m = self.models.top_model
        if len(self.activations[0]) != len(self.data.trn_ds):
            predict_to_bcolz(m, self.data.fix_dl, act)
        if len(self.activations[1]) != len(self.data.val_ds):
            predict_to_bcolz(m, self.data.val_dl, val_act)
        if self.data.test_dl and (len(self.activations[2]) != len(self.data.test_ds)):
            if self.data.test_dl: predict_to_bcolz(m, self.data.test_dl, test_act)

        self.fc_data = ImageClassifierData.from_arrays(self.data.path,
                                                       (act, self.data.trn_y), (val_act, self.data.val_y), self.data.bs,
                                                       classes=self.data.classes,
                                                       test=test_act if self.data.test_dl else None, num_workers=8)

    def freeze(self):
        """ Freeze all but the very last layer.

        Make all layers untrainable (i.e. frozen) except for the last layer.

        Returns:
            None
        """
        self.freeze_to(-1)

    def unfreeze(self):
        """ Unfreeze all layers.

        Make all layers trainable by unfreezing. This will also set the `precompute` to `False` since we can
        no longer pre-calculate the activation of frozen layers.

        Returns:
            None
        """
        self.freeze_to(0)
        self.precompute = False


class SaveBestModel(LossRecorder):
    """ Save weights of the best model based during training.
        If metrics are provided, the first metric in the list is used to
        find the best model.
        If no metrics are provided, the loss is used.

        Args:
            model: the fastai model
            lr: indicate to use test images; otherwise use validation images
            name: the name of filename of the weights without '.h5'

        Usage:
            Briefly, you have your model 'learn' variable and call fit.
            >>> learn.fit(lr, 2, cycle_len=2, cycle_mult=1, best_save_name='mybestmodel')
            ....
            >>> learn.load('mybestmodel')

            For more details see http://forums.fast.ai/t/a-code-snippet-to-save-the-best-model-during-training/12066

    """

    def __init__(self, model, layer_opt, metrics, name='best_model'):
        super().__init__(layer_opt)
        self.name = name
        self.model = model
        self.best_loss = None
        self.best_acc = None
        self.save_method = self.save_when_only_loss if metrics == None else self.save_when_acc

    def save_when_only_loss(self, metrics):
        loss = metrics[0]
        if self.best_loss == None or loss < self.best_loss:
            self.best_loss = loss
            self.model.save(f'{self.name}')

    def save_when_acc(self, metrics):
        loss, acc = metrics[0], metrics[1]
        if self.best_acc == None or acc > self.best_acc:
            self.best_acc = acc
            self.best_loss = loss
            self.model.save(f'{self.name}')
        elif acc == self.best_acc and loss < self.best_loss:
            self.best_loss = loss
            self.model.save(f'{self.name}')

    def on_epoch_end(self, metrics):
        super().on_epoch_end(metrics)
        self.save_method(metrics)


class CosAnneal(LR_Updater):
    """ Learning rate scheduler that inpelements a cosine annealation schedule. """

    def __init__(self, layer_opt, nb, on_cycle_end=None, cycle_mult=1):
        self.nb, self.on_cycle_end, self.cycle_mult = nb, on_cycle_end, cycle_mult
        super().__init__(layer_opt)

    def on_train_begin(self):
        self.cycle_iter, self.cycle_count = 0, 0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        if self.iteration < self.nb / 20:
            self.cycle_iter += 1
            return init_lrs / 100.

        cos_out = np.cos(np.pi * (self.cycle_iter) / self.nb) + 1
        self.cycle_iter += 1
        if self.cycle_iter == self.nb:
            self.cycle_iter = 0
            self.nb *= self.cycle_mult
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return init_lrs / 2 * cos_out


class LR_Finder(LR_Updater):
    """
    Helps you find an optimal learning rate for a model, as per suggetion of 2015 CLR paper.
    Learning rate is increased in linear or log scale, depending on user input, and the result of the loss funciton is retained and can be plotted later.
    """

    def __init__(self, layer_opt, nb, end_lr=10, linear=False, metrics=[]):
        self.linear, self.stop_dv = linear, True
        ratio = end_lr / layer_opt.lr
        self.lr_mult = (ratio / nb) if linear else ratio ** (1 / nb)
        super().__init__(layer_opt, metrics=metrics)

    def on_train_begin(self):
        super().on_train_begin()
        self.best = 1e9

    def calc_lr(self, init_lrs):
        mult = self.lr_mult * self.iteration if self.linear else self.lr_mult ** self.iteration
        return init_lrs * mult

    def on_batch_end(self, metrics):
        loss = metrics[0] if isinstance(metrics, list) else metrics
        if self.stop_dv and (math.isnan(loss) or loss > self.best * 4):
            return True
        if (loss < self.best and self.iteration > 10): self.best = loss
        return super().on_batch_end(metrics)

    def plot(self, n_skip=10, n_skip_end=5):
        """
        Plots the loss function with respect to learning rate, in log scale.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip:-(n_skip_end + 1)], self.losses[n_skip:-(n_skip_end + 1)])
        plt.xscale('log')


class LR_Finder2(LR_Finder):
    """
        A variant of lr_find() that helps find the best learning rate. It doesn't do
        an epoch but a fixed num of iterations (which may be more or less than an epoch
        depending on your data).
    """

    def __init__(self, layer_opt, nb, end_lr=10, linear=False, metrics=[], stop_dv=True):
        self.nb, self.metrics = nb, metrics
        super().__init__(layer_opt, nb, end_lr, linear, metrics)
        self.stop_dv = stop_dv

    def on_batch_end(self, loss):
        if self.iteration == self.nb:
            return True
        return super().on_batch_end(loss)

    def plot(self, n_skip=10, n_skip_end=5, smoothed=True):
        if self.metrics is None: self.metrics = []
        n_plots = len(self.metrics) + 2
        fig, axs = plt.subplots(n_plots, figsize=(6, 4 * n_plots))
        for i in range(0, n_plots): axs[i].set_xlabel('learning rate')
        axs[0].set_ylabel('training loss')
        axs[1].set_ylabel('validation loss')
        for i, m in enumerate(self.metrics):
            axs[i + 2].set_ylabel(m.__name__)
            if len(self.metrics) == 1:
                values = self.rec_metrics
            else:
                values = [rec[i] for rec in self.rec_metrics]
            if smoothed: values = smooth_curve(values, 0.98)
            axs[i + 2].plot(self.lrs[n_skip:-n_skip_end], values[n_skip:-n_skip_end])
        plt_val_l = smooth_curve(self.val_losses, 0.98) if smoothed else self.val_losses
        axs[0].plot(self.lrs[n_skip:-n_skip_end], self.losses[n_skip:-n_skip_end])
        axs[1].plot(self.lrs[n_skip:-n_skip_end], plt_val_l[n_skip:-n_skip_end])


class OptimScheduler(LossRecorder):
    """Learning rate Scheduler for training involving multiple phases."""

    def __init__(self, layer_opt, phases, nb_batches, stop_div=False):
        self.phases, self.nb_batches, self.stop_div = phases, nb_batches, stop_div
        super().__init__(layer_opt, record_mom=True)

    def on_train_begin(self):
        super().on_train_begin()
        self.phase, self.best = 0, 1e9

    def on_batch_begin(self):
        self.phases[self.phase].on_batch_begin()
        super().on_batch_begin()

    def on_batch_end(self, metrics):
        loss = metrics[0] if isinstance(metrics, list) else metrics
        if self.stop_div and (math.isnan(loss) or loss > self.best * 4):
            return True
        if (loss < self.best and self.iteration > 10): self.best = loss
        super().on_batch_end(metrics)
        self.phases[self.phase].update()

    def on_phase_begin(self):
        self.phases[self.phase].phase_begin(self.layer_opt, self.nb_batches)

    def on_phase_end(self):
        self.phase += 1

    def plot_lr(self, show_text=True, show_moms=True):
        """Plots the lr rate/momentum schedule"""
        phase_limits = [0]
        for phase in self.phases:
            phase_limits.append(phase_limits[-1] + self.nb_batches * phase.epochs)
        if not in_ipynb():
            plt.switch_backend('agg')
        np_plts = 2 if show_moms else 1
        fig, axs = plt.subplots(1, np_plts, figsize=(6 * np_plts, 4))
        if not show_moms: axs = [axs]
        for i in range(np_plts): axs[i].set_xlabel('iterations')
        axs[0].set_ylabel('learning rate')
        axs[0].plot(self.iterations, self.lrs)
        if show_moms:
            axs[1].set_ylabel('momentum')
            axs[1].plot(self.iterations, self.momentums)
        if show_text:
            for i, phase in enumerate(self.phases):
                text = phase.opt_fn.__name__
                if phase.wds is not None: text += '\nwds=' + str(phase.wds)
                if phase.beta is not None: text += '\nbeta=' + str(phase.beta)
                for k in range(np_plts):
                    if i < len(self.phases) - 1:
                        draw_line(axs[k], phase_limits[i + 1])
                    draw_text(axs[k], (phase_limits[i] + phase_limits[i + 1]) / 2, text)
        if not in_ipynb():
            plt.savefig(os.path.join(self.save_path, 'lr_plot.png'))

    def plot(self, n_skip=10, n_skip_end=5, linear=None):
        if linear is None: linear = self.phases[-1].lr_decay == DecayType.LINEAR
        plt.ylabel("loss")
        plt.plot(self.lrs[n_skip:-n_skip_end], self.losses[n_skip:-n_skip_end])
        if linear:
            plt.xlabel("learning rate")
        else:
            plt.xlabel("learning rate (log scale)")
            plt.xscale('log')


class FP16(nn.Module):
    def __init__(self, module):
        super(FP16, self).__init__()
        self.module = batchnorm_to_fp32(module.half())

    def forward(self, input):
        return self.module(input.half())

    def load_state_dict(self, *inputs, **kwargs):
        self.module.load_state_dict(*inputs, **kwargs)

    def state_dict(self, *inputs, **kwargs):
        return self.module.state_dict(*inputs, **kwargs)


class SWA(Callback):
    def __init__(self, model, swa_model, swa_start):
        super().__init__()
        self.model, self.swa_model, self.swa_start = model, swa_model, swa_start

    def on_train_begin(self):
        self.epoch = 0
        self.swa_n = 0

    def on_epoch_end(self, metrics):
        if (self.epoch + 1) >= self.swa_start:
            self.update_average_model()
            self.swa_n += 1

        self.epoch += 1

    def update_average_model(self):
        # update running average of parameters
        model_params = self.model.parameters()
        swa_params = self.swa_model.parameters()
        for model_param, swa_param in zip(model_params, swa_params):
            swa_param.data *= self.swa_n
            swa_param.data += model_param.data
            swa_param.data /= (self.swa_n + 1)


class LayerOptimizer():
    def __init__(self, opt_fn, layer_groups, lrs, wds=None):
        if not isinstance(layer_groups, (list, tuple)): layer_groups = [layer_groups]
        if not isinstance(lrs, Iterable): lrs = [lrs]
        if len(lrs) == 1: lrs = lrs * len(layer_groups)
        if wds is None: wds = 0.
        if not isinstance(wds, Iterable): wds = [wds]
        if len(wds) == 1: wds = wds * len(layer_groups)
        self.layer_groups, self.lrs, self.wds = layer_groups, lrs, wds
        self.opt = opt_fn(self.opt_params())

    def opt_params(self):
        assert (len(self.layer_groups) == len(self.lrs))
        assert (len(self.layer_groups) == len(self.wds))
        params = list(zip(self.layer_groups, self.lrs, self.wds))
        return [opt_params(*p) for p in params]

    @property
    def lr(self):
        return self.lrs[-1]

    @property
    def mom(self):
        if 'betas' in self.opt.param_groups[0]:
            return self.opt.param_groups[0]['betas'][0]
        else:
            return self.opt.param_groups[0]['momentum']

    def set_lrs(self, lrs):
        if not isinstance(lrs, Iterable): lrs = [lrs]
        if len(lrs) == 1: lrs = lrs * len(self.layer_groups)
        set_lrs(self.opt, lrs)
        self.lrs = lrs

    def set_wds(self, wds):
        if not isinstance(wds, Iterable): wds = [wds]
        if len(wds) == 1: wds = wds * len(self.layer_groups)
        set_wds(self.opt, wds)
        self.wds = wds

    def set_mom(self, momentum):
        if 'betas' in self.opt.param_groups[0]:
            for pg in self.opt.param_groups: pg['betas'] = (momentum, pg['betas'][1])
        else:
            for pg in self.opt.param_groups: pg['momentum'] = momentum

    def set_beta(self, beta):
        if 'betas' in self.opt.param_groups[0]:
            for pg in self.opt.param_groups: pg['betas'] = (pg['betas'][0], beta)
        elif 'alpha' in self.opt.param_groups[0]:
            for pg in self.opt.param_groups: pg['alpha'] = beta

    def set_opt_fn(self, opt_fn):
        if type(self.opt) != type(opt_fn(self.opt_params())):
            self.opt = opt_fn(self.opt_params())


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class Lambda(nn.Module):
    def __init__(self, f): super().__init__(); self.f = f

    def forward(self, x): return self.f(x)


class Flatten(nn.Module):
    def __init__(self): super().__init__()

    def forward(self, x): return x.view(x.size(0), -1)


class DecayType(IntEnum):
    ''' Data class, each decay type is assigned a number. '''
    NO = 1
    LINEAR = 2
    COSINE = 3
    EXPONENTIAL = 4
    POLYNOMIAL = 5


class Stepper():
    def __init__(self, m, opt, crit, clip=0, reg_fn=None, fp16=False, loss_scale=1):
        self.m, self.opt, self.crit, self.clip, self.reg_fn = m, opt, crit, clip, reg_fn
        self.fp16 = fp16
        self.reset(True)
        if self.fp16: self.fp32_params = copy_model_to_fp32(m, opt)
        self.loss_scale = loss_scale

    def reset(self, train=True):
        if train:
            apply_leaf(self.m, set_train_mode)
        else:
            self.m.eval()
        if hasattr(self.m, 'reset'):
            self.m.reset()
            if self.fp16: self.fp32_params = copy_model_to_fp32(self.m, self.opt)

    def step(self, xs, y, epoch):
        xtra = []
        output = self.m(*xs)
        if isinstance(output, tuple): output, *xtra = output
        if self.fp16:
            self.m.zero_grad()
        else:
            self.opt.zero_grad()
        loss = raw_loss = self.crit(output, y)
        if self.loss_scale != 1: assert (self.fp16); loss = loss * self.loss_scale
        if self.reg_fn: loss = self.reg_fn(output, xtra, raw_loss)
        loss.backward()
        if self.fp16: update_fp32_grads(self.fp32_params, self.m)
        if self.loss_scale != 1:
            for param in self.fp32_params: param.grad.data.div_(self.loss_scale)
        if self.clip:  # Gradient clipping
            if IS_TORCH_04:
                nn.utils.clip_grad_norm_(trainable_params_(self.m), self.clip)
            else:
                nn.utils.clip_grad_norm(trainable_params_(self.m), self.clip)
        self.opt.step()
        if self.fp16:
            copy_fp32_to_model(self.m, self.fp32_params)
            torch.cuda.synchronize()
        return torch_item(raw_loss.data)

    def evaluate(self, xs, y):
        preds = self.m(*xs)
        if isinstance(preds, tuple): preds = preds[0]
        return preds, self.crit(preds, y)


class IterBatch():
    def __init__(self, dl):
        self.idx = 0
        self.dl = dl
        self.iter = iter(dl)

    def __iter__(self): return self

    def next(self):
        res = next(self.iter)
        self.idx += 1
        if self.idx == len(self.dl):
            self.iter = iter(self.dl)
            self.idx = 0
        return res


def torch_item(x): return x.item() if hasattr(x, 'item') else x[0]


def set_train_mode(m):
    if (hasattr(m, 'running_mean') and (getattr(m, 'bn_freeze', False)
                                        or not getattr(m, 'trainable', False))):
        m.eval()
    elif (getattr(m, 'drop_freeze', False) and hasattr(m, 'p')
          and ('drop' in type(m).__name__.lower())):
        m.eval()
    else:
        m.train()


def opt_params(parm, lr, wd):
    return {'params': chain_params(parm), 'lr': lr, 'weight_decay': wd}


def set_lrs(opt, lrs):
    if not isinstance(lrs, Iterable): lrs = [lrs]
    if len(lrs) == 1: lrs = lrs * len(opt.param_groups)
    for pg, lr in zip_strict_(opt.param_groups, lrs): pg['lr'] = lr


def set_wds(opt, wds):
    if not isinstance(wds, Iterable): wds = [wds]
    if len(wds) == 1: wds = wds * len(opt.param_groups)
    assert (len(opt.param_groups) == len(wds))
    for pg, wd in zip_strict_(opt.param_groups, wds): pg['weight_decay'] = wd


def zip_strict_(l, r):
    assert (len(l) == len(r))
    return zip(l, r)


def chain_params(p):
    if is_listy(p):
        return list(chain(*[trainable_params_(o) for o in p]))
    return trainable_params_(p)


def trainable_params_(m):
    '''Returns a list of trainable parameters in the model m. (i.e., those that require gradients.)'''
    return [p for p in m.parameters() if p.requires_grad]


def fit(model, data, n_epochs, opt, crit, metrics=None, callbacks=None, stepper=Stepper,
        swa_model=None, swa_start=None, swa_eval_freq=None, **kwargs):
    """ Fits a model

    Arguments:
       model (model): any pytorch module
           net = to_gpu(net)
       data (ModelData): see ModelData class and subclasses (can be a list)
       opts: an optimizer. Example: optim.Adam.
       If n_epochs is a list, it needs to be the layer_optimizer to get the optimizer as it changes.
       n_epochs(int or list): number of epochs (or list of number of epochs)
       crit: loss function to optimize. Example: F.cross_entropy
    """

    all_val = kwargs.pop('all_val') if 'all_val' in kwargs else False
    get_ep_vals = kwargs.pop('get_ep_vals') if 'get_ep_vals' in kwargs else False
    metrics = metrics or []
    callbacks = callbacks or []
    avg_mom = 0.98
    batch_num, avg_loss = 0, 0.
    for cb in callbacks: cb.on_train_begin()
    names = ["epoch", "trn_loss", "val_loss"] + [f.__name__ for f in metrics]
    if swa_model is not None:
        swa_names = ['swa_loss'] + [f'swa_{f.__name__}' for f in metrics]
        names += swa_names
        # will use this to call evaluate later
        swa_stepper = stepper(swa_model, None, crit, **kwargs)

    layout = "{!s:10} " * len(names)
    if not isinstance(n_epochs, Iterable): n_epochs = [n_epochs]
    if not isinstance(data, Iterable): data = [data]
    if len(data) == 1: data = data * len(n_epochs)
    for cb in callbacks: cb.on_phase_begin()
    model_stepper = stepper(model, opt.opt if hasattr(opt, 'opt') else opt, crit, **kwargs)
    ep_vals = collections.OrderedDict()
    tot_epochs = int(np.ceil(np.array(n_epochs).sum()))
    cnt_phases = np.array([ep * len(dat.trn_dl) for (ep, dat) in zip(n_epochs, data)]).cumsum()
    phase = 0
    for epoch in tnrange(tot_epochs, desc='Epoch'):
        model_stepper.reset(True)
        cur_data = data[phase]
        if hasattr(cur_data, 'trn_sampler'): cur_data.trn_sampler.set_epoch(epoch)
        if hasattr(cur_data, 'val_sampler'): cur_data.val_sampler.set_epoch(epoch)
        num_batch = len(cur_data.trn_dl)
        t = tqdm(iter(cur_data.trn_dl), leave=False, total=num_batch)
        if all_val: val_iter = IterBatch(cur_data.val_dl)

        for (*x, y) in t:
            batch_num += 1
            for cb in callbacks: cb.on_batch_begin()
            loss = model_stepper.step(V(x), V(y), epoch)
            avg_loss = avg_loss * avg_mom + loss * (1 - avg_mom)
            debias_loss = avg_loss / (1 - avg_mom ** batch_num)
            t.set_postfix(loss=debias_loss)
            stop = False
            los = debias_loss if not all_val else [debias_loss] + validate_next(model_stepper, metrics, val_iter)
            for cb in callbacks: stop = stop or cb.on_batch_end(los)
            if stop: return
            if batch_num >= cnt_phases[phase]:
                for cb in callbacks: cb.on_phase_end()
                phase += 1
                if phase >= len(n_epochs):
                    t.close()
                    break
                for cb in callbacks: cb.on_phase_begin()
                if isinstance(opt, LayerOptimizer): model_stepper.opt = opt.opt
                if cur_data != data[phase]:
                    t.close()
                    break

        if not all_val:
            vals = validate(model_stepper, cur_data.val_dl, metrics)
            stop = False
            for cb in callbacks: stop = stop or cb.on_epoch_end(vals)
            if swa_model is not None:
                if (epoch + 1) >= swa_start and (
                        (epoch + 1 - swa_start) % swa_eval_freq == 0 or epoch == tot_epochs - 1):
                    fix_batchnorm(swa_model, cur_data.trn_dl)
                    swa_vals = validate(swa_stepper, cur_data.val_dl, metrics)
                    vals += swa_vals

            if epoch == 0: print(layout.format(*names))
            print_stats(epoch, [debias_loss] + vals)
            ep_vals = append_stats(ep_vals, epoch, [debias_loss] + vals)
        if stop: break
    for cb in callbacks: cb.on_train_end()
    if get_ep_vals:
        return vals, ep_vals
    else:
        return vals


def append_stats(ep_vals, epoch, values, decimals=6):
    ep_vals[epoch] = list(np.round(values, decimals))
    return ep_vals


def print_stats(epoch, values, decimals=6):
    layout = "{!s:^10}" + " {!s:10}" * len(values)
    values = [epoch] + list(np.round(values, decimals))
    print(layout.format(*values))


def validate(stepper, dl, metrics):
    batch_cnts, loss, res = [], [], []
    stepper.reset(False)
    with no_grad_context():
        for (*x, y) in iter(dl):
            preds, l = stepper.evaluate(VV(x), VV(y))
            if isinstance(x, list):
                batch_cnts.append(len(x[0]))
            else:
                batch_cnts.append(len(x))
            loss.append(to_np(l))
            res.append([f(preds.data, y) for f in metrics])
    return [np.average(loss, 0, weights=batch_cnts)] + list(np.average(np.stack(res), 0, weights=batch_cnts))


def validate_next(stepper, metrics, val_iter):
    """Computes the loss on the next minibatch of the validation set."""
    stepper.reset(False)
    with no_grad_context():
        (*x, y) = val_iter.next()
        preds, l = stepper.evaluate(VV(x), VV(y))
        res = [to_np(l)[0]]
        res += [f(preds.data, y) for f in metrics]
    stepper.reset(True)
    return res


def predict(m, dl):
    preda, _ = predict_with_targs_(m, dl)
    return to_np(torch.cat(preda))


def predict_batch(m, x):
    m.eval()
    if hasattr(m, 'reset'): m.reset()
    return m(VV(x))


def predict_to_bcolz(m, gen, arr, workers=4):
    arr.trim(len(arr))
    lock = threading.Lock()
    m.eval()
    for x, *_ in tqdm(gen):
        y = to_np(m(VV(x)).data)
        with lock:
            arr.append(y)
            arr.flush()


def predict_with_targs_(m, dl):
    m.eval()
    if hasattr(m, 'reset'): m.reset()
    res = []
    for *x, y in iter(dl): res.append([get_prediction(m(*VV(x))), y])
    return zip(*res)


def predict_with_targs(m, dl):
    preda, targa = predict_with_targs_(m, dl)
    return to_np(torch.cat(preda)), to_np(torch.cat(targa))


def get_prediction(x):
    if is_listy(x): x = x[0]
    return x.data


def no_grad_context(): return torch.no_grad() if IS_TORCH_04 else contextlib.suppress()


def collect_bn_modules(module, bn_modules):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        bn_modules.append(module)


def fix_batchnorm(swa_model, train_dl):
    """
    During training, batch norm layers keep track of a running mean and
    variance of the previous layer's activations. Because the parameters
    of the SWA model are computed as the average of other models' parameters,
    the SWA model never sees the training data itself, and therefore has no
    opportunity to compute the correct batch norm statistics. Before performing
    inference with the SWA model, we perform a single pass over the training data
    to calculate an accurate running mean and variance for each batch norm layer.
    """
    bn_modules = []
    swa_model.apply(lambda module: collect_bn_modules(module, bn_modules))

    if not bn_modules: return

    swa_model.train()

    for module in bn_modules:
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

    momenta = [m.momentum for m in bn_modules]

    inputs_seen = 0

    for (*x, y) in iter(train_dl):
        xs = V(x)
        batch_size = xs[0].size(0)

        momentum = batch_size / (inputs_seen + batch_size)
        for module in bn_modules:
            module.momentum = momentum

        res = swa_model(*xs)

        inputs_seen += batch_size

    for module, momentum in zip(bn_modules, momenta):
        module.momentum = momentum


def batchnorm_to_fp32(module):
    """
    BatchNorm layers to have parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    """
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_fp32(child)
    return module


def copy_model_to_fp32(m, optim):
    """  Creates a fp32 copy of model parameters and sets optimizer parameters
    """
    fp32_params = [m_param.clone().type(torch.cuda.FloatTensor).detach() for m_param in m.parameters()]
    optim_groups = [group['params'] for group in optim.param_groups]
    iter_fp32_params = iter(fp32_params)
    for group_params in optim_groups:
        for i in range(len(group_params)):
            fp32_param = next(iter_fp32_params)
            fp32_param.requires_grad = group_params[i].requires_grad
            group_params[i] = fp32_param
    return fp32_params


def copy_fp32_to_model(m, fp32_params):
    m_params = list(m.parameters())
    for fp32_param, m_param in zip(fp32_params, m_params):
        m_param.data.copy_(fp32_param.data)


def update_fp32_grads(fp32_params, m):
    m_params = list(m.parameters())
    for fp32_param, m_param in zip(fp32_params, m_params):
        if fp32_param.grad is None:
            fp32_param.grad = nn.Parameter(fp32_param.data.new().resize_(*fp32_param.data.size()))
        fp32_param.grad.data.copy_(m_param.grad.data)


def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module): f(m)
    if len(c) > 0:
        for l in c: apply_leaf(l, f)


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters(): p.requires_grad = b


def read_dirs(path, folder):
    """
    Fetches name of all files in path in long form, and labels associated by extrapolation of directory names.
    :param path:
    :param folder:
    :return:
    """
    lbls, fnames, all_lbls = [], [], []
    full_path = os.path.join(path, folder)
    for lbl in sorted(os.listdir(full_path)):
        if lbl not in ('.ipynb_checkpoints', '.DS_Store'):
            all_lbls.append(lbl)
            for fname in os.listdir(os.path.join(full_path, lbl)):
                fnames.append(os.path.join(folder, lbl, fname))
                lbls.append(lbl)
    return fnames, lbls, all_lbls


def folder_source(path, folder):
    """
    Returns the filenames and labels for a folder within a path

    Returns:
    -------
    fnames: a list of the filenames within `folder`
    all_lbls: a list of all of the labels in `folder`, where the # of labels is determined by the # of directories
        within `folder`
    lbl_arr: a numpy array of the label indices in `all_lbls`
    """
    fnames, lbls, all_lbls = read_dirs(path, folder)
    lbl2idx = {lbl: idx for idx, lbl in enumerate(all_lbls)}
    idxs = [lbl2idx[lbl] for lbl in lbls]
    lbl_arr = np.array(idxs, dtype=int)
    return fnames, lbl_arr, all_lbls


def read_dir(path, folder):
    """ Returns a list of relative file paths to `path` for all files within `folder` """
    full_path = os.path.join(path, folder)
    fnames = glob(f"{full_path}/*.*")
    if any(fnames):
        return [os.path.relpath(f, path) for f in fnames]
    else:
        raise FileNotFoundError("{} folder doesn't exist or is empty".format(folder))


def csv_source(folder, csv_file, skip_header=True, suffix='', continuous=False):
    fnames, csv_labels = parse_csv_labels(csv_file, skip_header)
    return dict_source(folder, fnames, csv_labels, suffix, continuous)


def dict_source(folder, fnames, csv_labels, suffix='', continuous=False):
    all_labels = sorted(list(set(p for o in csv_labels.values() for p in o)))
    full_names = [os.path.join(folder, str(fn) + suffix) for fn in fnames]
    if continuous:
        label_arr = np.array([np.array(csv_labels[i]).astype(np.float32) for i in fnames])
    else:
        label2idx = {v: k for k, v in enumerate(all_labels)}
        label_arr = nhot_labels(label2idx, csv_labels, fnames, len(all_labels))
        is_single = np.all(label_arr.sum(axis=1) == 1)
        if is_single:
            label_arr = np.argmax(label_arr, axis=1)
    return full_names, label_arr, all_labels


def nhot_labels(label2idx, csv_labels, fnames, c):
    all_idx = {k: n_hot([label2idx[o] for o in v], c)
               for k, v in csv_labels.items()}
    return np.stack([all_idx[o] for o in fnames])


def split_by_idx(idxs, *a):
    """
    Split each array passed as *a, to a pair of arrays like this (elements selected by idxs,  the remaining elements)
    This can be used to split multiple arrays containing training data to validation and training set.
    :param idxs:
    :param a:
    :return: list of tuples, each containing a split of corresponding array from *a.
            First element of each tuple is an array composed from elements selected by idxs,
            second element is an array of remaining elements.
    """
    mask = np.zeros(len(a[0]), dtype=bool)
    mask[np.array(idxs)] = True
    return [(o[mask], o[~mask]) for o in a]


def split_by_idxs(seq, idxs):
    '''A generator that returns sequence pieces, seperated by indexes specified in idxs. '''
    last = 0
    for idx in idxs:
        yield seq[last:idx]
        last = idx
    yield seq[last:]


def get_cv_idxs(n, cv_idx=0, val_pct=0.2, seed=42):
    """ Get a list of index values for Validation set from a dataset

    Arguments:
        n : int, Total number of elements in the data set.
        cv_idx : int, starting index [idx_start = cv_idx*int(val_pct*n)]
        val_pct : (int, float), validation set percentage
        seed : seed value for RandomState

    Returns:
        list of indexes
    """
    np.random.seed(seed)
    n_val = int(val_pct * n)
    idx_start = cv_idx * n_val
    idxs = np.random.permutation(n)
    return idxs[idx_start:idx_start + n_val]


def num_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def get_tensor(batch, pin, half=False):
    if isinstance(batch, (np.ndarray, np.generic)):
        batch = T(batch, half=half, cuda=False).contiguous()
        if pin:
            batch = batch.pin_memory()
        return to_gpu(batch)
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: get_tensor(sample, pin, half) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [get_tensor(sample, pin, half) for sample in batch]
    raise TypeError(f"batch must contain numbers, dicts or lists; found {type(batch)}")


def chunk_iter(iterable, chunk_size):
    """A generator that yields chunks of iterable, chunk_size at a time. """
    while True:
        chunk = []
        try:
            for _ in range(chunk_size):
                chunk.append(next(iterable))
            yield chunk
        except StopIteration:
            if chunk:
                yield chunk
            break


def open_image(fn):
    """ Opens an image using OpenCV given the file path.

    Arguments:
        fn: the file path of the image

    Returns:
        The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    flags = cv2.IMREAD_UNCHANGED + cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        try:
            im = cv2.imread(str(fn), flags).astype(np.float32) / 255
            if im is None:
                raise OSError(f'File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e


def resize_imgs(fnames, targ, path, new_path):
    """
    Enlarge or shrink a set of images in the same directory to scale, such that the smaller of the height or width
    dimension is equal to targ.
    Note:
    -- This function is multithreaded for efficiency.
    -- When destination file or folder already exist, function exists without raising an error.
    """
    if not os.path.exists(os.path.join(path, new_path, str(targ), fnames[0])):
        with ThreadPoolExecutor(8) as e:
            ims = e.map(lambda x: resize_img(x, targ, path, new_path), fnames)
            for _ in tqdm(ims, total=len(fnames), leave=False):
                pass
    return os.path.join(path, new_path, str(targ))


def resize_img(fname, targ, path, new_path):
    """
    Enlarge or shrink a single image to scale, such that the smaller of the height or width dimension is equal to targ.
    """
    dest = os.path.join(path, new_path, str(targ), fname)
    if os.path.exists(dest):
        return
    im = Image.open(os.path.join(path, fname)).convert('RGB')
    r, c = im.size
    ratio = targ / min(r, c)
    sz = (scale_to(r, ratio, targ), scale_to(c, ratio, targ))
    os.makedirs(os.path.split(dest)[0], exist_ok=True)
    im.resize(sz, Image.LINEAR).save(dest)


def to_np(v):
    """returns an np.array object given an input of np.array, list, tuple, torch variable or tensor."""
    if isinstance(v, (np.ndarray, np.generic)):
        return v
    if isinstance(v, (list, tuple)):
        return [to_np(o) for o in v]
    if isinstance(v, Variable):
        v = v.data
    if isinstance(v, torch.cuda.HalfTensor):
        v = v.float()
    return v.cpu().numpy()


def parse_csv_labels(fn, skip_header=True, cat_separator=' '):
    """
    Parse filenames and label sets from a CSV file.

    This method expects that the csv file at path :fn: has two columns. If it
    has a header, :skip_header: should be set to True. The labels in the
    label set are expected to be space separated.

    :param fn: Path to a CSV file.
    :param skip_header: A boolean flag indicating whether to skip the header.
    :param cat_separator: the separator for the categories column
    :return: a two-tuple of (
            image filenames,
            a dictionary of filenames and corresponding labels
        )
    """
    df = pd.read_csv(fn, index_col=0, header=0 if skip_header else None, dtype=str)
    fnames = df.index.values
    df.iloc[:, 0] = df.iloc[:, 0].str.split(cat_separator)
    return fnames, list(df.to_dict().values())[0]


def n_hot(ids, c):
    """one hot encoding by index. Returns array of length c, where all entries are 0, except for the indecies in ids"""
    res = np.zeros((c,), dtype=np.float32)
    res[ids] = 1
    return res


def T(a, half=False, cuda=True):
    """
    Convert numpy array into a pytorch tensor.
    if Cuda is available and USE_GPU=ture, store resulting tensor in GPU.
    """
    if not torch.is_tensor(a):
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            a = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            a = torch.cuda.HalfTensor(a) if half else torch.FloatTensor(a)
        else:
            raise NotImplementedError(a.dtype)
    if cuda:
        a = to_gpu(a, async=True)
    return a


def A(*a):
    """convert iterable object into numpy array"""
    return np.array(a[0]) if len(a) == 1 else [np.array(o) for o in a]


def V_(x, requires_grad=False, volatile=False):
    '''equivalent to create_variable, which creates a pytorch tensor'''
    return create_variable(x, volatile=volatile, requires_grad=requires_grad)


def V(x, requires_grad=False, volatile=False):
    '''creates a single or a list of pytorch tensors, depending on input x. '''
    return map_over(x, lambda o: V_(o, requires_grad, volatile))


def VV_(x):
    '''creates a volatile tensor, which does not require gradients. '''
    return create_variable(x, True)


def VV(x):
    '''creates a single or a list of pytorch tensors, depending on input x. '''
    return map_over(x, VV_)


def map_over(x, f): return [f(o) for o in x] if is_listy(x) else f(x)


def map_none(x, f): return None if x is None else f(x)


def create_variable(x, volatile, requires_grad=False):
    if type(x) != Variable:
        if IS_TORCH_04:
            x = Variable(T(x), requires_grad=requires_grad)
        else:
            x = Variable(T(x), requires_grad=requires_grad, volatile=volatile)
    return x


def to_gpu(x, *args, **kwargs):
    """puts pytorch variable to gpu, if cuda is avaialble and USE_GPU is set to true. """
    return x.cuda(*args, **kwargs) if USE_GPU else x


def scale_to(x, ratio, targ):
    """Calculate dimension of an image during scaling with aspect ratio"""
    return max(math.floor(x * ratio), targ)


# From https://github.com/ncullen93/torchsample
def model_summary(m, input_size):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            if is_listy(output):
                summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and module.bias is not None:
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if (not isinstance(module, nn.Sequential) and
                not isinstance(module, nn.ModuleList) and
                not (module == m)):
            hooks.append(module.register_forward_hook(hook))

    summary = OrderedDict()
    hooks = []
    m.apply(register_hook)

    if is_listy(input_size[0]):
        x = [to_gpu(Variable(torch.rand(3, *in_size))) for in_size in input_size]
    else:
        x = [to_gpu(Variable(torch.rand(3, *input_size)))]
    m(*x)

    for h in hooks: h.remove()
    return summary


gg = {}
gg['hook_position'] = 0
gg['total_fc_conv_layers'] = 0
gg['done_counter'] = -1
gg['hook'] = None
gg['act_dict'] = {}
gg['counter_to_apply_correction'] = 0
gg['correction_needed'] = False
gg['current_coef'] = 1.0


def count_conv_fc_layers(m):
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        gg['total_fc_conv_layers'] += 1
    return


def apply_lsuv_init(model, data, needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=True, cuda=True):
    model.eval();
    if cuda:
        model = model.cuda()
        data = data.cuda()
    else:
        model = model.cpu()
        data = data.cpu()

    model.apply(count_conv_fc_layers)
    if do_orthonorm:
        model.apply(orthogonal_weights_init)
        if cuda:
            model = model.cuda()
    for layer_idx in range(gg['total_fc_conv_layers']):
        model.apply(add_current_hook)
        out = model(data)
        current_std = gg['act_dict'].std()
        attempts = 0
        while (np.abs(current_std - needed_std) > std_tol):
            gg['current_coef'] = needed_std / (current_std + 1e-8);
            gg['correction_needed'] = True
            model.apply(apply_weights_correction)
            if cuda:
                model = model.cuda()
            out = model(data)
            current_std = gg['act_dict'].std()
            attempts += 1
            if attempts > max_attempts:
                print(f'Cannot converge in {max_attempts} iterations')
                break
        if gg['hook'] is not None:
            gg['hook'].remove()
        gg['done_counter'] += 1
        gg['counter_to_apply_correction'] = 0
        gg['hook_position'] = 0
        gg['hook'] = None
    if not cuda:
        model = model.cpu()
    return model


def add_current_hook(m):
    if gg['hook'] is not None:
        return
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        if gg['hook_position'] > gg['done_counter']:
            gg['hook'] = m.register_forward_hook(store_activations)
        else:
            gg['hook_position'] += 1
    return


def store_activations(self, input, output):
    gg['act_dict'] = output.data.cpu().numpy();
    return


def apply_weights_correction(m):
    if gg['hook'] is None:
        return
    if not gg['correction_needed']:
        return
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        if gg['counter_to_apply_correction'] < gg['hook_position']:
            gg['counter_to_apply_correction'] += 1
        else:
            if hasattr(m, 'weight_g'):
                m.weight_g.data *= float(gg['current_coef'])
                gg['correction_needed'] = False
            else:
                m.weight.data *= gg['current_coef']
                gg['correction_needed'] = False
            return
    return


def svd_orthonormal(w):
    shape = w.shape
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)#w;
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q.astype(np.float32)


def orthogonal_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if hasattr(m, 'weight_v'):
            w_ortho = svd_orthonormal(m.weight_v.data.cpu().numpy())
            m.weight_v.data = torch.from_numpy(w_ortho)
            try:
                nn.init.constant(m.bias, 0)
            except:
                pass
        else:
            w_ortho = svd_orthonormal(m.weight.data.cpu().numpy())
            m.weight.data = torch.from_numpy(w_ortho)
            try:
                nn.init.constant(m.bias, 0)
            except:
                pass
    return


def in_ipynb():
    try:
        cls = get_ipython().__class__.__name__
        return cls == 'ZMQInteractiveShell'
    except NameError:
        return False


def cut_model(m, cut):
    return list(m.children())[:cut] if cut else [m]


def num_features(m):
    c = children(m)
    if len(c) == 0: return None
    for l in reversed(c):
        if hasattr(l, 'num_features'): return l.num_features
        res = num_features(l)
        if res is not None: return res


def cond_init(m, init_fn):
    if not isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)):
        if hasattr(m, 'weight'): init_fn(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)

def apply_init(m, init_fn):
    m.apply(lambda x: cond_init(x, init_fn))


def draw_line(ax, x):
    xmin, xmax, ymin, ymax = ax.axis()
    ax.plot([x, x], [ymin, ymax], color='red', linestyle='dashed')


def draw_text(ax, x, text):
    xmin, xmax, ymin, ymax = ax.axis()
    ax.text(x, (ymin + ymax) / 2, text, horizontalalignment='center', verticalalignment='center', fontsize=14,
            alpha=0.5)


def smooth_curve(vals, beta):
    avg_val = 0
    smoothed = []
    for (i, v) in enumerate(vals):
        avg_val = beta * avg_val + (1 - beta) * v
        smoothed.append(avg_val / (1 - beta ** (i + 1)))
    return smoothed


class TfmType(IntEnum):
    """ Type of transformation.
    Parameters
        IntEnum: predefined types of transformations
            NO:    the default, y does not get transformed when x is transformed.
            PIXEL: x and y are images and should be transformed in the same way.
                   Example: image segmentation.
            COORD: y are coordinates (i.e bounding boxes)
            CLASS: y are class labels (same behaviour as PIXEL, except no normalization)
    """
    NO = 1
    PIXEL = 2
    COORD = 3
    CLASS = 4


class Denormalize(object):
    """ De-normalizes an image, returning it to original format.
    """

    def __init__(self, m, s):
        self.m = np.array(m, dtype=np.float32)
        self.s = np.array(s, dtype=np.float32)

    def __call__(self, x): return x * self.s + self.m


class Normalize(object):
    """ Normalizes an image to zero mean and unit standard deviation, given the mean m and std s of the original
    image """

    def __init__(self, m, s, tfm_y=TfmType.NO):
        self.m = np.array(m, dtype=np.float32)
        self.s = np.array(s, dtype=np.float32)
        self.tfm_y = tfm_y

    def __call__(self, x, y=None):
        x = (x - self.m) / self.s
        if self.tfm_y == TfmType.PIXEL and y is not None:
            y = (y - self.m) / self.s
        return x, y


class CropType(IntEnum):
    """ Type of image cropping.
    """
    RANDOM = 1
    CENTER = 2
    NO = 3
    GOOGLENET = 4


class Transform(object):
    """ A class that represents a transform.

    All other transforms should subclass it. All subclasses should override
    do_transform.

    Arguments
    ---------
        tfm_y : TfmType
            type of transform
    """

    def __init__(self, tfm_y=TfmType.NO):
        self.tfm_y = tfm_y
        self.store = threading.local()

    def set_state(self): pass

    def __call__(self, x, y):
        self.set_state()
        x, y = ((self.transform(x), y) if self.tfm_y == TfmType.NO
        else self.transform(x, y) if self.tfm_y in (TfmType.PIXEL, TfmType.CLASS)
        else self.transform_coord(x, y))
        return x, y

    def transform_coord(self, x, y): return self.transform(x), y

    def transform(self, x, y=None):
        x = self.do_transform(x, False)
        return (x, self.do_transform(y, True)) if y is not None else x

    @abstractmethod
    def do_transform(self, x, is_y): raise NotImplementedError


class CoordTransform(Transform):
    """ A coordinate transform.  """

    @staticmethod
    def make_square(y, x):
        r, c, *_ = x.shape
        y1 = np.zeros((r, c))
        y = y.astype(np.int)
        y1[y[0]:y[2], y[1]:y[3]] = 1.
        return y1

    def map_y(self, y0, x):
        y = CoordTransform.make_square(y0, x)
        y_tr = self.do_transform(y, True)
        return to_bb(y_tr)

    def transform_coord(self, x, ys):
        yp = partition(ys, 4)
        y2 = [self.map_y(y, x) for y in yp]
        x = self.do_transform(x, False)
        return x, np.concatenate(y2)


class RandomScale(CoordTransform):
    """ Scales an image so that the min size is a random number between [sz, sz*max_zoom]

    This transforms (optionally) scales x,y at with the same parameters.
    Arguments:
        sz: int
            target size
        max_zoom: float
            float >= 1.0
        p : float
            a probability for doing the random sizing
        tfm_y: TfmType
            type of y transform
    """

    def __init__(self, sz, max_zoom, p=0.75, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.sz, self.max_zoom, self.p, self.sz_y = sz, max_zoom, p, sz_y

    def set_state(self):
        min_z = 1.
        max_z = self.max_zoom
        if isinstance(self.max_zoom, collections.Iterable):
            min_z, max_z = self.max_zoom
        self.store.mult = random.uniform(min_z, max_z) if random.random() < self.p else 1
        self.store.new_sz = int(self.store.mult * self.sz)
        if self.sz_y is not None:
            self.store.new_sz_y = int(self.store.mult * self.sz_y)

    def do_transform(self, x, is_y):
        if is_y:
            return scale_min(x, self.store.new_sz_y,
                             cv2.INTER_AREA if self.tfm_y == TfmType.PIXEL else cv2.INTER_NEAREST)
        else:
            return scale_min(x, self.store.new_sz, cv2.INTER_AREA)


class Scale(CoordTransform):
    """ A transformation that scales the min size to sz.

    Arguments:
        sz: int
            target size to scale minimum size.
        tfm_y: TfmType
            type of y transformation.
    """

    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.sz, self.sz_y = sz, sz_y

    def do_transform(self, x, is_y):
        if is_y:
            return scale_min(x, self.sz_y, cv2.INTER_AREA if self.tfm_y == TfmType.PIXEL else cv2.INTER_NEAREST)
        else:
            return scale_min(x, self.sz, cv2.INTER_AREA)


class AddPadding(CoordTransform):
    """ A class that represents adding paddings to an image.

    The default padding is border_reflect
    Arguments
    ---------
        pad : int
            size of padding on top, bottom, left and right
        mode:
            type of cv2 padding modes. (e.g., constant, reflect, wrap, replicate. etc. )
    """

    def __init__(self, pad, mode=cv2.BORDER_REFLECT, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.pad, self.mode = pad, mode

    def do_transform(self, im, is_y):
        return cv2.copyMakeBorder(im, self.pad, self.pad, self.pad, self.pad, self.mode)


class Transforms(object):
    def __init__(self, sz, tfms, normalizer, denorm, crop_type=CropType.CENTER,
                 tfm_y=TfmType.NO, sz_y=None):
        if sz_y is None:
            sz_y = sz
        self.sz, self.denorm, self.norm, self.sz_y = sz, denorm, normalizer, sz_y
        crop_tfm = crop_fn_lu[crop_type](sz, tfm_y, sz_y)
        self.tfms = tfms
        self.tfms.append(crop_tfm)
        if normalizer is not None:
            self.tfms.append(normalizer)
        self.tfms.append(ChannelOrder(tfm_y))

    def __call__(self, im, y=None):
        return compose(im, y, self.tfms)

    def __repr__(self):
        return str(self.tfms)


class ChannelOrder(object):
    """
    changes image array shape from (h, w, 3) to (3, h, w).
    tfm_y decides the transformation done to the y element.
    """

    def __init__(self, tfm_y=TfmType.NO):
        self.tfm_y = tfm_y

    def __call__(self, x, y):
        x = np.rollaxis(x, 2)
        # if isinstance(y,np.ndarray) and (len(y.shape)==3):
        if self.tfm_y == TfmType.PIXEL:
            y = np.rollaxis(y, 2)
        elif self.tfm_y == TfmType.CLASS:
            y = y[..., 0]
        return x, y


class RandomCrop(CoordTransform):
    """ A class that represents a Random Crop transformation.

    This transforms (optionally) transforms x,y at with the same parameters.
    Arguments
    ---------
        targ: int
            target size of the crop.
        tfm_y: TfmType
            type of y transformation.
    """

    def __init__(self, targ_sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.targ_sz, self.sz_y = targ_sz, sz_y

    def set_state(self):
        self.store.rand_r = random.uniform(0, 1)
        self.store.rand_c = random.uniform(0, 1)

    def do_transform(self, x, is_y):
        r, c, *_ = x.shape
        sz = self.sz_y if is_y else self.targ_sz
        start_r = np.floor(self.store.rand_r * (r - sz)).astype(int)
        start_c = np.floor(self.store.rand_c * (c - sz)).astype(int)
        return crop(x, start_r, start_c, sz)


class CenterCrop(CoordTransform):
    """ A class that represents a Center Crop.

    This transforms (optionally) transforms x,y at with the same parameters.
    Arguments
    ---------
        sz: int
            size of the crop.
        tfm_y : TfmType
            type of y transformation.
    """

    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.min_sz, self.sz_y = sz, sz_y

    def do_transform(self, x, is_y):
        return center_crop(x, self.sz_y if is_y else self.min_sz)


class NoCrop(CoordTransform):
    """  A transformation that resize to a square image without cropping.

    This transforms (optionally) resizes x,y at with the same parameters.
    Arguments:
        targ: int
            target size of the crop.
        tfm_y (TfmType): type of y transformation.
    """

    def __init__(self, sz, tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.sz, self.sz_y = sz, sz_y

    def do_transform(self, x, is_y):
        if is_y:
            return no_crop(x, self.sz_y, cv2.INTER_AREA if self.tfm_y == TfmType.PIXEL else cv2.INTER_NEAREST)
        else:
            return no_crop(x, self.sz, cv2.INTER_AREA)


class GoogleNetResize(CoordTransform):
    """ Randomly crops an image with an aspect ratio and returns a squared resized image of size targ

    Arguments:
        targ_sz: int
            target size
        min_area_frac: float < 1.0
            minimum area of the original image for cropping
        min_aspect_ratio : float
            minimum aspect ratio
        max_aspect_ratio : float
            maximum aspect ratio
        flip_hw_p : float
            probability for flipping magnitudes of height and width
        tfm_y: TfmType
            type of y transform
    """

    def __init__(self, targ_sz,
                 min_area_frac=0.08, min_aspect_ratio=0.75, max_aspect_ratio=1.333, flip_hw_p=0.5,
                 tfm_y=TfmType.NO, sz_y=None):
        super().__init__(tfm_y)
        self.targ_sz, self.tfm_y, self.sz_y = targ_sz, tfm_y, sz_y
        self.min_area_frac, self.min_aspect_ratio, self.max_aspect_ratio, self.flip_hw_p = \
            min_area_frac, min_aspect_ratio, max_aspect_ratio, flip_hw_p

    def set_state(self):
        # if self.random_state: random.seed(self.random_state)
        self.store.fp = random.random() < self.flip_hw_p

    def do_transform(self, x, is_y):
        sz = self.sz_y if is_y else self.targ_sz
        if is_y:
            interpolation = cv2.INTER_NEAREST if self.tfm_y in (TfmType.COORD, TfmType.CLASS) else cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_AREA
        return googlenet_resize(x, sz, self.min_area_frac, self.min_aspect_ratio, self.max_aspect_ratio, self.store.fp,
                                interpolation=interpolation)


crop_fn_lu = {CropType.RANDOM: RandomCrop, CropType.CENTER: CenterCrop, CropType.NO: NoCrop,
              CropType.GOOGLENET: GoogleNetResize}


def accuracy_np(preds, targs):
    preds = np.argmax(preds, 1)
    return (preds == targs).mean()


def accuracy(preds, targs):
    preds = torch.max(preds, dim=1)[1]
    return (preds == targs).float().mean()


def accuracy_thresh(thresh):
    return lambda preds, targs: accuracy_multi(preds, targs, thresh)


def accuracy_multi(preds, targs, thresh):
    return ((preds > thresh).float() == targs).float().mean()


def accuracy_multi_np(preds, targs, thresh):
    return ((preds > thresh) == targs).mean()


def recall(preds, targs, thresh=0.5):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    return tpos.sum() / targs.sum()


def precision(preds, targs, thresh=0.5):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    return tpos.sum() / pred_pos.sum()


def fbeta(preds, targs, beta, thresh=0.5):
    """Calculates the F-beta score (the weighted harmonic mean of precision and recall).
    This is the micro averaged version where the true positives, false negatives and
    false positives are calculated globally (as opposed to on a per label basis).

    beta == 1 places equal weight on precision and recall, b < 1 emphasizes precision and
    beta > 1 favors recall.
    """
    assert beta > 0, 'beta needs to be greater than 0'
    beta2 = beta ** 2
    rec = recall(preds, targs, thresh)
    prec = precision(preds, targs, thresh)
    return (1 + beta2) * prec * rec / (beta2 * prec + rec)


def f1(preds, targs, thresh=0.5): return fbeta(preds, targs, 1, thresh)


def tfms_from_stats(stats, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=CropType.RANDOM,
                    tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT, norm_y=True, scale=None):
    """ Given the statistics of the training image sets, returns separate training and validation transform functions
    """
    if aug_tfms is None:
        aug_tfms = []
    tfm_norm = Normalize(*stats, tfm_y=tfm_y if norm_y else TfmType.NO) if stats is not None else None
    tfm_denorm = Denormalize(*stats) if stats is not None else None
    val_crop = CropType.CENTER if crop_type in (CropType.RANDOM, CropType.GOOGLENET) else crop_type
    val_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=pad, crop_type=val_crop,
                        tfm_y=tfm_y, sz_y=sz_y, scale=scale)
    trn_tfm = image_gen(tfm_norm, tfm_denorm, sz, pad=pad, crop_type=crop_type,
                        tfm_y=tfm_y, sz_y=sz_y, tfms=aug_tfms, max_zoom=max_zoom, pad_mode=pad_mode, scale=scale)
    return trn_tfm, val_tfm


def tfms_from_model(f_model, sz, aug_tfms=None, max_zoom=None, pad=0, crop_type=CropType.RANDOM,
                    tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT, norm_y=True, scale=None):
    """
    Returns separate transformers of images for training and validation.
    Transformers are constructed according to the image statistics given b y the model. (See tfms_from_stats)
    :param f_model: model, pretrained or not pretrained
    :param sz:
    :param aug_tfms:
    :param max_zoom:
    :param pad:
    :param crop_type:
    :param tfm_y:
    :param sz_y:
    :param pad_mode:
    :param norm_y:
    :param scale:
    :return:
    """
    stats = inception_stats if f_model in inception_models else imagenet_stats
    return tfms_from_stats(stats, sz, aug_tfms, max_zoom=max_zoom, pad=pad, crop_type=crop_type,
                           tfm_y=tfm_y, sz_y=sz_y, pad_mode=pad_mode, norm_y=norm_y, scale=scale)


def image_gen(normalizer, denorm, sz, tfms=None, max_zoom=None, pad=0, crop_type=None,
              tfm_y=None, sz_y=None, pad_mode=cv2.BORDER_REFLECT, scale=None):
    """
    Generate a standard set of transformations
    See Also:  Transforms: the transformer object returned by this function
    :param normalizer: image normalizing function
    :param denorm: image denormalizing function
    :param sz: size, sz_y = sz if not specified.
    :param tfms: iterable collection of transformation functions
    :param max_zoom: float, maximum zoom
    :param pad: int, padding on top, left, right and bottom
    :param crop_type: crop type
    :param tfm_y: y axis specific transformations
    :param sz_y: y size, height
    :param pad_mode:  cv2 padding style: repeat, reflect, etc.
    :param scale:
    :return: type : ``Transforms``
         transformer for specified image operations.
    """
    if tfm_y is None:
        tfm_y = TfmType.NO
    if tfms is None:
        tfms = []
    elif not isinstance(tfms, collections.Iterable):
        tfms = [tfms]
    if sz_y is None:
        sz_y = sz
    if scale is None:
        scale = [RandomScale(sz, max_zoom, tfm_y=tfm_y, sz_y=sz_y) if max_zoom is not None
                 else Scale(sz, tfm_y, sz_y=sz_y)]
    elif not is_listy(scale):
        scale = [scale]
    if pad:
        scale.append(AddPadding(pad, mode=pad_mode))
    if crop_type != CropType.GOOGLENET:
        tfms = scale + tfms
    return Transforms(sz, tfms, normalizer, denorm, crop_type,
                      tfm_y=tfm_y, sz_y=sz_y)


def to_bb(YY):
    """Convert mask YY to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(YY)
    if len(cols) == 0:
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)


def partition(a, sz):
    """splits iterables a in equal parts of size sz"""
    return [a[i:i + sz] for i in range(0, len(a), sz)]


def scale_min(im, targ, interpolation=cv2.INTER_AREA):
    """ Scales the image so that the smallest axis is of size targ.

    Arguments:
        im (array): image
        targ (int): target size
        interpolation:
    """
    r, c, *_ = im.shape
    ratio = targ / min(r, c)
    sz = (scale_to(c, ratio, targ), scale_to(r, ratio, targ))
    return cv2.resize(im, sz, interpolation=interpolation)


def compose(im, y, fns):
    """ apply a collection of transformation functions fns to images
    """
    for fn in fns:
        # pdb.set_trace()
        im, y = fn(im, y)
    return im if y is None else (im, y)


def crop(im, r, c, sz):
    """ crop image into a square of size sz, """
    return im[r:r + sz, c:c + sz]


def center_crop(im, min_sz=None):
    """ Returns a center crop of an image"""
    r, c, *_ = im.shape
    if min_sz is None:
        min_sz = min(r, c)
    start_r = math.ceil((r - min_sz) / 2)
    start_c = math.ceil((c - min_sz) / 2)
    return crop(im, start_r, start_c, min_sz)


def no_crop(im, min_sz=None, interpolation=cv2.INTER_AREA):
    """ Returns a squared resized image """
    r, c, *_ = im.shape
    if min_sz is None:
        min_sz = min(r, c)
    return cv2.resize(im, (min_sz, min_sz), interpolation=interpolation)


def googlenet_resize(im, targ, min_area_frac, min_aspect_ratio, max_aspect_ratio, flip_hw_p,
                     interpolation=cv2.INTER_AREA):
    """ Randomly crops an image with an aspect ratio and returns a squared resized image of size targ

    References:
    1. https://arxiv.org/pdf/1409.4842.pdf
    2. https://arxiv.org/pdf/1802.07888.pdf
    """
    h, w, *_ = im.shape
    area = h * w
    for _ in range(10):
        targetArea = random.uniform(min_area_frac, 1.0) * area
        aspectR = random.uniform(min_aspect_ratio, max_aspect_ratio)
        ww = int(np.sqrt(targetArea * aspectR) + 0.5)
        hh = int(np.sqrt(targetArea / aspectR) + 0.5)
        if flip_hw_p:
            ww, hh = hh, ww
        if hh <= h and ww <= w:
            x1 = 0 if w == ww else random.randint(0, w - ww)
            y1 = 0 if h == hh else random.randint(0, h - hh)
            out = im[y1:y1 + hh, x1:x1 + ww]
            out = cv2.resize(out, (targ, targ), interpolation=interpolation)
            return out
    out = scale_min(im, targ, interpolation=interpolation)
    out = center_crop(out)
    return out


def is_listy(x): return isinstance(x, (list, tuple))


def SGD_Momentum(momentum):
    return lambda *args, **kwargs: optim.SGD(*args, momentum=momentum, **kwargs)


def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def save_model(m, p):
    torch.save(m.state_dict(), p)


def load_model(m, p):
    m.load_state_dict(torch.load(p, map_location=lambda storage, loc: storage))


def load_pre(pre, f, fn):
    m = f()
    path = os.path.dirname(__file__)
    if pre:
        load_model(m, f'{path}/weights/{fn}.pth')
    return m


def _fastai_model(name, paper_title, paper_href):
    def add_docs_wrapper(f):
        f.__doc__ = f"""{name} model from
        `"{paper_title}" <{paper_href}>`_

        Args:
           pre (bool): If True, returns a model pre-trained on ImageNet
        """
        return f

    return add_docs_wrapper


@_fastai_model('Inception 4', 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning',
               'https://arxiv.org/pdf/1602.07261.pdf')
def inception_4(pre): return children(inceptionv4(pretrained=pre))[0]


@_fastai_model('Inception 4', 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning',
               'https://arxiv.org/pdf/1602.07261.pdf')
def inceptionresnet_2(pre): return load_pre(pre, InceptionResnetV2, 'inceptionresnetv2-d579a627')


@_fastai_model('ResNeXt 50', 'Aggregated Residual Transformations for Deep Neural Networks',
               'https://arxiv.org/abs/1611.05431')
def resnext50(pre): return load_pre(pre, resnext_50_32x4d, 'resnext_50_32x4d')


@_fastai_model('ResNeXt 101_32', 'Aggregated Residual Transformations for Deep Neural Networks',
               'https://arxiv.org/abs/1611.05431')
def resnext101(pre): return load_pre(pre, resnext_101_32x4d, 'resnext_101_32x4d')


@_fastai_model('ResNeXt 101_64', 'Aggregated Residual Transformations for Deep Neural Networks',
               'https://arxiv.org/abs/1611.05431')
def resnext101_64(pre): return load_pre(pre, resnext_101_64x4d, 'resnext_101_64x4d')


@_fastai_model('Wide Residual Networks', 'Wide Residual Networks',
               'https://arxiv.org/pdf/1605.07146.pdf')
def wrn(pre): return load_pre(pre, wrn_50_2f, 'wrn_50_2f')


@_fastai_model('Densenet-121', 'Densely Connected Convolutional Networks',
               'https://arxiv.org/pdf/1608.06993.pdf')
def dn121(pre): return children(densenet121(pre))[0]


@_fastai_model('Densenet-169', 'Densely Connected Convolutional Networks',
               'https://arxiv.org/pdf/1608.06993.pdf')
def dn161(pre): return children(densenet161(pre))[0]


@_fastai_model('Densenet-161', 'Densely Connected Convolutional Networks',
               'https://arxiv.org/pdf/1608.06993.pdf')
def dn169(pre): return children(densenet169(pre))[0]


@_fastai_model('Densenet-201', 'Densely Connected Convolutional Networks',
               'https://arxiv.org/pdf/1608.06993.pdf')
def dn201(pre): return children(densenet201(pre))[0]


@_fastai_model('Vgg-16 with batch norm added', 'Very Deep Convolutional Networks for Large-Scale Image Recognition',
               'https://arxiv.org/pdf/1409.1556.pdf')
def vgg16(pre): return children(vgg16_bn(pre))[0]


@_fastai_model('Vgg-19 with batch norm added', 'Very Deep Convolutional Networks for Large-Scale Image Recognition',
               'https://arxiv.org/pdf/1409.1556.pdf')
def vgg19(pre): return children(vgg19_bn(pre))[0]


imagenet_stats = A([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
"""Statistics pertaining to image data from image net. mean and std of the images of each color channel"""
inception_stats = A([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
inception_models = (inception_4, inceptionresnet_2)
model_meta = {
    resnet18: [8, 6], resnet34: [8, 6], resnet50: [8, 6], resnet101: [8, 6], resnet152: [8, 6],
    vgg16: [0, 22], vgg19: [0, 22],
    resnext50: [8, 6], resnext101: [8, 6], resnext101_64: [8, 6],
    wrn: [8, 6], inceptionresnet_2: [-2, 9], inception_4: [-1, 9],
    dn121: [0, 7], dn161: [0, 7], dn169: [0, 7], dn201: [0, 7],
}
model_features = {inception_4: 3072, dn121: 2048, dn161: 4416, }  # nasnetalarge: 4032*2}
