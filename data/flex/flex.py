from fluomapper.utils.data import search_spectral_window, select

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
import numpy as np

import xarray as xr


def get_data_window(mask):
    data_rows = np.where(mask.min(axis=-1) == False)[0]
    top = data_rows[0]
    bottom = data_rows[-1]

    data_cols = np.where(mask.min(axis=-2) == False)[0]
    left = data_cols[0]
    right = data_cols[-1]

    return (left, bottom, right, top)


class FLEXReader(object):
    def __init__(self, path_tuple, out_mode, target_var, out_shape=None, meta_info=None, spectral_window_obs_wvl=None, 
            *args, **kwargs):
        super(FLEXReader, self).__init__()

        self.obs = xr.open_rasterio(path_tuple[0])
        self.params = xr.open_rasterio(path_tuple[1])

        self.mask = ~np.logical_and(np.all(self.obs.data > 0, axis=0), np.all(~np.isnan(self.params.data), axis=0))

        left, bottom, right, top = get_data_window(self.mask)
        self.data_window = (slice(None, None), slice(top, bottom), slice(left, right))

        self.params_array = self.params.data[self.data_window]
        self.param_names = self.params.descriptions

        self.obs_array = self.obs.data[self.data_window]
        self.wavelengths = np.asarray(self.obs.descriptions).astype(float)
        self.spectral_window_obs_wvl = spectral_window_obs_wvl

        if self.spectral_window_obs_wvl is not None:
            win = search_spectral_window(*self.spectral_window_obs_wvl, where=self.wavelengths)
            self.wavelengths = select(windows=win, signals=self.wavelengths)
            self.obs_array = select(windows=win, signals=self.obs_array, axis=0)

        self.meta_info = meta_info
        self.target_var = target_var

        self.out_mode = out_mode
        if self.out_mode == 'single_spectra':
            self.obs_array = torch.from_numpy(self.obs_array).flatten(start_dim=1).numpy().transpose()
            self.params_array = torch.from_numpy(self.params_array).flatten(start_dim=1).numpy().transpose()

        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        vars_ = dict(obs=self.obs_array[index])
        vars_.update(dict([(param_name, self.params_array[index, self.param_names.index(param_name)])
                           for param_name in self.meta_info]))

        y = self.params_array[index, self.param_names.index(self.target_var)]

        return vars_, y

    def __len__(self):
        if self.out_mode == 'single_spectra':
            return self.obs_array.shape[0]


class FLEXDset(Dataset):
    def __init__(self, path_tuples, indices=None, *args, **kwargs):
        super(FLEXDset, self).__init__()

        if type(path_tuples) is not list:
            path_tuples = [path_tuples]

        self.readers = [FLEXReader(path_tuple, *args, **kwargs) for path_tuple in path_tuples]
        self.len_per_reader = np.asarray([len(reader) for reader in self.readers])
        self.cumsum = np.r_[0, np.cumsum(self.len_per_reader)]

        self.indices = indices if indices is not None else range(len(self))

    def __getitem__(self, item, **kwargs):
        reader_ind = np.where(item < self.cumsum)[0][0] - 1
        ind = item - self.cumsum[reader_ind]
        return self.readers[reader_ind].__getitem__(ind, **kwargs)

    def __len__(self, ):
        return np.sum(self.len_per_reader)

    def setup(self):
        return

    @classmethod
    def add_argparse_args(self, parser, *args, **kwargs):
        parser.add_argument('--spectral_window_obs_wvl', type=float, default=None, nargs="+")
        parser.add_argument('--meta_info', type=str, default=None, nargs="+")

        parser.add_argument('--out_shape', type=int, default=17)
        parser.add_argument('--out_mode', type=str, default='single_spectra')

        return parser



class FLEXDmodule(pl.LightningDataModule):
    def __init__(self, val_frac, path_tuples, shuffle_dataset=False, batch_size=16, persistent_workers=True,
                 pin_memory=False, num_workers=1, random_seed=42, val_path_tuples=None, *args, **kwargs):
        super(FLEXDmodule, self).__init__()

        self.path_tuples = path_tuples
        self.val_path_tuples = val_path_tuples if val_path_tuples is not None else path_tuples

        if val_path_tuples is None:
            dset = self.get_dset(path_tuples=self.path_tuples, *args, **kwargs)
            indices = list(range(len(dset)))
            split = int(np.floor(val_frac * len(dset)))
            if shuffle_dataset:
                np.random.seed(random_seed)
                np.random.shuffle(indices)

            self.train_indices, self.val_indices = indices[split:], indices[:split]

        else:
            dset = self.get_dset(path_tuples=self.path_tuples, *args, **kwargs)
            self.train_indices = list(range(len(dset)))

            dset = self.get_dset(path_tuples=self.val_path_tuples, *args, **kwargs)
            self.val_indices = list(range(len(dset)))

        self.batch_size = batch_size
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        self.dset_kwargs = kwargs

    def get_dloader(self, shuffle=False, mode='train', **kwargs):
        if mode == 'train':
            dset = self.trn_dset

        elif mode == 'val':
            dset = self.val_dset

        else:
            raise NotImplementedError()

        sampler_kwargs = dict(batch_size=self.batch_size,
                              shuffle=shuffle)

        other_kwargs = dict(persistent_workers=self.persistent_workers and self.num_workers > 0,
                            pin_memory=self.pin_memory,
                            num_workers=self.num_workers,
                            collate_fn=self.collate_fn,)

        dl = DataLoader(dset, **sampler_kwargs, **other_kwargs)

        return dl

    def setup(self, stage='fit'):
        if stage in ("val", "fit") or stage is None:
            ## VALIDATION ##
            limit_patches = None
            self.val_dset = self.get_dset(limit_patches=limit_patches,
                                          **self.dset_kwargs, **self.val_data)

            self.val_dset.setup()

        if stage == "fit" or stage is None:
            ## TRAIN ##
            limit_patches = None

            self.trn_dset = self.get_dset(limit_patches=limit_patches,
                                          **self.dset_kwargs, **self.trn_data)

            self.trn_dset.setup()

        if stage == "test" or stage is None:
            raise NotImplementedError()

    def get_dset(self, *args, **kwargs):
        if type(self.path_tuples) is str:
            return FLEXDset(*args, **kwargs)
        else:
            return FLEXDset(*args, **kwargs)

    def prepare_data(self):
        self.trn_data, self.val_data, self.tst_data = self.assemble_data()

    def assemble_data(self, *args, **kwargs):
        return (dict(indices=self.train_indices, path_tuples=self.path_tuples),
                dict(indices=self.val_indices, path_tuples=self.val_path_tuples),
                dict([]))

    @staticmethod
    def collate_fn(*args, **kwargs):
        return default_collate(*args, **kwargs)

    def train_dataloader(self, shuffle=True, **kwargs):
        return self.get_dloader(mode='train', shuffle=True, **kwargs)

    def val_dataloader(self, **kwargs):
        return self.get_dloader(mode='val', shuffle=False, **kwargs)

    def test_dataloader(self, **kwargs):
        return self.get_dloader(mode='test', shuffle=False, **kwargs)

    @classmethod
    def add_argparse_args(cls, parser, *args, **kwargs):
        parser = FluomapDset.add_argparse_args(parser, *args, **kwargs)
        parser.add_argument('--database_path', type=str, nargs="*", default=None)
        parser.add_argument('--val_database_path', type=str, default=None)

        parser.add_argument('--num_workers', default=1, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--pin_memory', default=0, type=int)
        parser.add_argument('--persistent_workers', default=0, type=int)
        parser.add_argument('--val_frac', default=0.5, type=float)
        parser.add_argument('--shuffle_dataset', default=0, type=int)
        return parser
