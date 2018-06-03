"""
Code lightly modified from the Fast AI library
https://github.com/fastai/fastai
"""

from src.data_sci.fastai.transforms import *
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from src.data_sci.fastai.core import *
import collections

string_classes = (str, bytes)


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


def read_dir(path, folder):
    """ Returns a list of relative file paths to `path` for all files within `folder` """
    full_path = os.path.join(path, folder)
    fnames = glob(f"{full_path}/*.*")
    if any(fnames):
        return [os.path.relpath(f, path) for f in fnames]
    else:
        raise FileNotFoundError("{} folder doesn't exist or is empty".format(folder))


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


def n_hot(ids, c):
    """
    one hot encoding by index. Returns array of length c, where all entries are 0, except for the indecies in ids
    :param ids:
    :param c:
    :return:
    """
    res = np.zeros((c,), dtype=np.float32)
    res[ids] = 1
    return res


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


def parse_csv_labels(fn, skip_header=True, cat_separator=' '):
    """
    Parse filenames and label sets from a CSV file.
    This method expects that the csv file at path :fn: has two columns. If it
    has a header, :skip_header: should be set to True. The labels in the
    label set are expected to be space separated.
    :param fn: Path to a CSV file.
    :param skip_header: A boolean flag indicating whether to skip the header.
    :param cat_separator: the separator for the categories column
    :return: a four-tuple of (
            sorted image filenames,
            a dictionary of filenames and corresponding labels,
            a sorted set of unique labels,
            a dictionary of labels to their corresponding index, which will
            be one-hot encoded.
        )
    """
    df = pd.read_csv(fn, index_col=0, header=0 if skip_header else None, dtype=str)
    fnames = df.index.values
    df.iloc[:, 0] = df.iloc[:, 0].str.split(cat_separator)
    return sorted(fnames), list(df.to_dict().values())[0]


def nhot_labels(label2idx, csv_labels, fnames, c):
    all_idx = {k: n_hot([label2idx[o] for o in v], c)
               for k, v in csv_labels.items()}
    return np.stack([all_idx[o] for o in fnames])


def csv_source(folder, csv_file, skip_header=True, suffix='', continuous=False):
    fnames, csv_labels = parse_csv_labels(csv_file, skip_header)
    return dict_source(folder, fnames, csv_labels, suffix, continuous)


def dict_source(folder, fnames, csv_labels, suffix='', continuous=False):
    all_labels = sorted(list(set(p for o in csv_labels.values() for p in o)))
    full_names = [os.path.join(folder, str(fn)+suffix) for fn in fnames]
    if continuous:
        label_arr = np.array([np.array(csv_labels[i]).astype(np.float32) for i in fnames])
    else:
        label2idx = {v: k for k, v in enumerate(all_labels)}
        label_arr = nhot_labels(label2idx, csv_labels, fnames, len(all_labels))
        is_single = np.all(label_arr.sum(axis=1) == 1)
        if is_single:
            label_arr = np.argmax(label_arr, axis=1)
    return full_names, label_arr, all_labels


def resize_img(fname, targ, path, new_path):
    """
    Enlarge or shrink a single image to scale, such that the smaller of the height or width dimension is equal to targ.
    """
    dest = os.path.join(path, new_path, str(targ), fname)
    if os.path.exists(dest):
        return
    im = Image.open(os.path.join(path, fname)).convert('RGB')
    r, c = im.size
    ratio = targ/min(r, c)
    sz = (scale_to(r, ratio, targ), scale_to(c, ratio, targ))
    os.makedirs(os.path.split(dest)[0], exist_ok=True)
    im.resize(sz, Image.LINEAR).save(dest)


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
            for x in tqdm(ims, total=len(fnames), leave=False): pass
    return os.path.join(path, new_path, str(targ))


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

    def __len__(self): return self.n

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


def open_image(fn):
    """ Opens an image using OpenCV given the file path.
    Arguments:
        fn: the file path of the image
    Returns:
        The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        # res = np.array(Image.open(fn), dtype=np.float32)/255
        # if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
        # return res
        try:
            im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None:
                raise OSError(f'File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e


class ArraysDataset(BaseDataset):
    def __init__(self, x, y, transform):
        self.x, self.y = x, y
        assert(len(x) == len(y))
        super().__init__(transform)

    def get_x(self, i):
        return self.x[i]

    def get_y(self, i):
        return self.y[i]

    def get_n(self):
        return len(self.y)

    def get_sz(self):
        return self.x.shape[1]


class ArraysIndexDataset(ArraysDataset):
    def get_c(self):
        return int(self.y.max())+1

    def get_y(self, i):
        return self.y[i]


class ArraysNhotDataset(ArraysDataset):
    def get_c(self):
        return self.y.shape[1]

    @property
    def is_multi(self):
        return True


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
        return DataLoader(ds, batch_size=self.bs, shuffle=shuffle, num_workers=self.num_workers, pin_memory=False)

    @property
    def sz(self): return self.trn_ds.sz

    @property
    def c(self): return self.trn_ds.c

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
            fn(val[0], val[1], tfms[0], **kwargs)   # aug
        ]
        if test is not None:
            if isinstance(test, tuple):
                test_lbls = test[1]
                test = test[0]
            else:
                test_lbls = np.zeros((len(test), 1))
            res += [
                fn(test, test_lbls, tfms[1], **kwargs),  # test
                fn(test, test_lbls, tfms[0], **kwargs)   # test_aug
            ]
        else:
            res += [None, None]
        return res


class FilesDataset(BaseDataset):
    def __init__(self, fnames, transform, path):
        self.path, self.fnames = path, fnames
        super().__init__(transform)

    def get_sz(self): return self.transform.sz

    def get_x(self, i): return open_image(os.path.join(self.path, self.fnames[i]))

    def get_n(self): return len(self.fnames)

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


class FilesArrayDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y = y
        assert(len(fnames) == len(y))
        super().__init__(fnames, transform, path)

    def get_y(self, i): return self.y[i]

    def get_c(self):
        return self.y.shape[1] if len(self.y.shape) > 1 else 0


class FilesIndexArrayDataset(FilesArrayDataset):
    def get_c(self): return int(self.y.max())+1


class FilesNhotArrayDataset(FilesArrayDataset):
    @property
    def is_multi(self): return True


class FilesIndexArrayRegressionDataset(FilesArrayDataset):
    def is_reg(self): return True


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
            test_with_labels:
            num_workers: number of workers
        Returns:
            ImageClassifierData
        """
        assert not(tfms[0] is None or tfms[1] is None), "please provide transformations for your train and " \
                                                        "validation sets"
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
            continuous: TODO
            skip_header: skip the first row of the CSV file.
            num_workers: number of workers
        Returns:
            ImageClassifierData
        """
        assert not (tfms[0] is None or tfms[1] is None), "please provide transformations for your train and" \
                                                         " validation sets"
        assert not (os.path.isabs(folder)), "folder needs to be a relative path"
        fnames, y, classes = csv_source(folder, csv_fname, skip_header, suffix, continuous=continuous)
        return cls.from_names_and_array(path, fnames, y, classes, val_idxs, test_name, num_workers=num_workers,
                                        suffix=suffix, tfms=tfms, bs=bs, continuous=continuous)

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
        datasets = cls.get_ds(f, (trn_fnames, trn_y), (val_fnames, val_y), tfms, path=path, test=test_fnames)
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

    def __len__(self): return len(self.batch_sampler)

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
                for c in chunk_iter(iter(self.batch_sampler), self.num_workers*10):
                    for batch in e.map(self.get_batch, c):
                        yield get_tensor(batch, self.pin_memory, self.half)


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
