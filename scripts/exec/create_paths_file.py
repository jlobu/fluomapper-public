# BASIC PATHS
user = '/home/user'
data = '/home/user/mount'
app = '/home/user/.../fluomapper'

data = data.rstrip('/')
user = user.rstrip('/')

import sys
sys.path.append(f'{user}/projects/')

import fluomapper
from fluomapper.utils.prediction_utils import list_to_txt

import glob

import os 
from os.path import join as pjoin


# BASIC SEARCH SETUP

SAVE_DIR = f'{app}/_paths'
SAVE_DIR = f'{user}/projects/data/HyPlant/path_files'

NAME = 'RAJ_all.txt'

BASE = f'{data}'
DO_SEARCH = True

sensor = 'FLUO'
height = ''
#campaign_ids = ['VER', 'WST', 'NRS', 'CKA', 'SEL', 'KRA', 'RAJ', 'HOE', 'JIM', 'AFO', 'BEC', 'BOR', 'KAL', 'INN', 'TR32']
campaign_ids = ['RAJ', ]


if DO_SEARCH:
    ls = []
    for line in campaign_ids:
        #ls += glob.glob(pjoin(BASE, f'processed/*2018/*/*/{sensor}/*{line}*{height}*_radiance.dat'))
        #ls += glob.glob(pjoin(BASE, f'processed/*2019/*/*/{sensor}/*{line}*{height}*_radiance.dat'))
        #ls += glob.glob(pjoin(BASE, f'processed/*2020/*/*/{sensor}/*{line}*{height}*_radiance.dat'))
        #ls += glob.glob(pjoin(BASE, f'processed/*2021/*/*/{sensor}/*{line}*{height}*_radiance.dat'))
        #ls += glob.glob(pjoin(BASE, f'processed/*2022/*/*/{sensor}/*{line}*{height}*_radiance.dat'))
        #ls += glob.glob(pjoin(BASE, f'processed/*2023/*/*/{sensor}/*{line}*{height}*_radiance.dat'))
        #ls += glob.glob(pjoin(BASE, f'processed/*2023/*/{sensor}/*/*{line}*{height}*_radiance.dat'))

        ls += glob.glob(pjoin(BASE, f'raw/*2023/{sensor}/*/*{line}*{height}*_radiance.dat'))

    save_ls = [p[len(BASE.strip().rstrip('/')) + 1:] for p in ls]
    val_files = ls

    # SAVE TO TEXT FILE
    os.makedirs(SAVE_DIR, exist_ok=True)
    paths_file = pjoin(SAVE_DIR, NAME)
    list_to_txt(paths_file, save_ls)

    print('Created ', paths_file)

else:
    paths_file = pjoin(SAVE_DIR, NAME)

    if paths_file.endswith('pkl'):
        with open(paths_file, 'rb') as f:
            val = np.asarray(pkl.load(f))

    elif paths_file.endswith('txt'):
        with open(file, "r") as f:
            paths = [line.strip() for line in f if line.strip()]

    else:
        raise NotImplementedError()

    val_files = [pjoin(BASE, p[0]) for p in val]
    print('Loaded ', paths_file)

