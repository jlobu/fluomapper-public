import numpy as np
from numba import njit
import pickle as pkl

import os, glob
from os.path import join as pjoin
from os.path import basename as pbasename
import time

import subprocess as sp

import rioxarray as xr
import rasterio
from rasterio.warp import transform, reproject, Resampling
from rasterio.transform import Affine

import geopandas as gpd
import pandas as pd

from shapely.geometry import box
import datetime, pytz
import pysolar as sol
import pyproj

from scipy.interpolate import griddata
from scipy.ndimage import label

from functools import partial
import multiprocessing


def execute_sequential(func, args_list, **kwargs):
    """Execute a function sequentially over a list of arguments."""
    results = []
    for args in args_list:
        result = func(*args, **kwargs)
        results.append(result)
    return results


def execute_parallel_multithreading(func, args_list, num_workers=4, wait=True, **kwargs):
    """Execute a function in parallel using ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(func, *args, **kwargs): args for args in args_list}

        if not wait:
            return futures, executor

        #results = [future.result() for future in concurrent.futures.as_completed(futures)]
        results = [future.result() for future in futures]

    gc.collect()

    return results, futures, executor

def execute_parallel_multiprocessing(func, args_list, num_workers=4, wait=True, **kwargs):
    # Partially apply the function with the kwargs
    func_with_kwargs = partial(func, **kwargs)

    with multiprocessing.Pool(processes=num_workers) as pool:
        # Map each set of arguments in args_list to the target function
        results = pool.starmap(func_with_kwargs, args_list)  # Use starmap to pass multiple arguments

    return results, pool


def execute_function(func, args_list, use_parallel=True, num_workers=4, wait=True, **kwargs):
    """Generalized function execution (either sequentially or in parallel)."""
    if use_parallel:
        #return execute_parallel_multithreading(func, args_list, num_workers, wait=wait, **kwargs)
        return execute_parallel_multiprocessing(func, args_list, num_workers, wait=wait, **kwargs)
    else:
        return execute_sequential(func, args_list, **kwargs)


# Get DEM in sensor geometry

def get_in_glt_format(fil, path, glt_path):
    glt = xr.open_rasterio(glt_path)

    ret = crop_with_resampling(fil, (0, 0, glt.shape[2], glt.shape[1]), glt.rio.crs.to_wkt(), glt.rio.transform())
    ret = ret.astype(float)

    new_shape = xr.open_rasterio(path).shape[1:]
    sensor_arr = to_sensor_geometry(new_shape, glt=glt, data=ret)

    return sensor_arr, glt


@njit
def _iterate(data, coords, new_shape):
    out = np.ones(new_shape) * np.nan
    norm = np.zeros(new_shape)

    for c, a in zip(coords, data):
        if np.logical_or(c[0] - 1 > new_shape[1],
                         c[1] - 1 > new_shape[0]):
            continue

        if np.isnan(c[0]) or np.isnan(c[1]):
            continue

        if c[0] == 0 and c[1] == 0:
            continue

        np_index = c - np.array([1, 1])
        np_index = np_index[::-1]

        if np.isnan(out[np_index[0], np_index[1]]):
            out[np_index[0], np_index[1]] = 0

        out[np_index[0], np_index[1]] += a  # here add arr value and normalize
        norm[np_index[0], np_index[1]] += 1

        # print(c, np_index, new_shape, a)
    out = out / (norm + 1e-12)
    return out


def to_sensor_geometry(new_shape, glt, data, do_norm=True, do_interpolate=True):
    glt_data = glt.data
    glt_data = np.abs(glt_data)
    all_coords = np.stack((glt_data[0].flatten(), glt_data[1].flatten()), axis=-1)
    data = data.flatten()

    out = _iterate(data, all_coords, new_shape)

    if do_interpolate:
        out = out[None]

        masked_out = np.ma.masked_where(np.isnan(out), out)
        out = interpolate(masked_out, method='linear')

        out = out[0]

    return out


def cut(fil, shp, shape, tmp, write=True):
    os.makedirs(tmp, exist_ok=True)

    out_file = 'cut_' + '%.0f' % time.time() + fil[-4:]
    out_file = pjoin(tmp, out_file)
    out_tmp = out_file[:-4] + '_tmp' + out_file[-4:]

    cmd = f'gdalwarp -overwrite -of GTiff -cutline "{shp}" -crop_to_cutline "{fil}" "{out_tmp}" '
    sp.run(cmd, shell=True)

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    fil = xr.open_rasterio(fil)
    fil_shape = ' '.join(np.array(list(map(str, shape))))
    cmd = f'gdal_translate -of GTiff -outsize {fil_shape} -r bilinear "{out_tmp}" "{out_file}"'
    sp.run(cmd, shell=True)
    os.remove(out_tmp)

    ret = xr.open_rasterio(out_file).data
    if not write:
        ret = ret.compute()
        os.remove(out_file)

        if os.path.exists(out_tmp):
            os.remove(out_tmp)

    return ret


def get_extent_as_shp(file, out_path):
    if type(file) is str:
        fil = xr.open_rasterio(file)

    else:
        fil = file

    minx, miny = min(fil.coords['x']), min(fil.coords['y'])
    maxx, maxy = max(fil.coords['x']), max(fil.coords['y'])

    in_crs = fil.rio.crs
    df = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=in_crs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_file(out_path)

    return df, (minx, miny, maxx, maxy)


def find_enclosed_nan(nan_mask):
    # Label the connected components of valid values
    valid_mask = ~nan_mask
    labeled_array, num_features = label(valid_mask)

    # Prepare a set to collect indices of enclosed NaNs
    enclosed_nan = np.zeros_like(nan_mask)

    # Iterate through each NaN in the original array
    for (i, j) in np.argwhere(nan_mask):
        # Get the 1-pixel boundary around the NaN
        surrounding_labels = labeled_array[i - 1:i + 2, j - 1:j + 2]

        # Check if all surrounding values belong to the same valid region
        unique_labels = set(surrounding_labels.flatten())
        unique_labels.discard(0)  # Remove background label

        if len(unique_labels) == 1:
            # If only one unique label exists (meaning enclosed)
            enclosed_nan[i, j] = True

        else:
            enclosed_nan[i, j] = False

    return enclosed_nan


def interpolate(masked_arr, method='linear', only_enclosed_nan=False):
    # Prepare the output array
    interpolated_arr = np.empty(masked_arr.shape)
    if np.sum(masked_arr.mask) == 0:
        return masked_arr.data

    if only_enclosed_nan:
        enclosed_nan = find_enclosed_nan(masked_arr.mask[0])

    else:
        enclosed_nan = None

    # Get the first band to compute valid points
    first_band = masked_arr[0]
    xx, yy = np.meshgrid(np.arange(first_band.shape[1]), np.arange(first_band.shape[0]))

    # Get valid and invalid points from the first band
    xvalid = xx[~first_band.mask]
    yvalid = yy[~first_band.mask]
    arrvalid = first_band[~first_band.mask]

    for band in range(masked_arr.shape[0]):
        band_data = masked_arr[band]

        tbi_mask = band_data.mask
        if enclosed_nan is not None:
            mask = np.logical_and(tbi_mask, enclosed_nan)

        xinvalid = xx[tbi_mask]
        yinvalid = yy[tbi_mask]

        # Create a copy of the band data
        interpolated_band = band_data.data.copy()

        # Interpolate using valid points from the first band
        interpolated_band[band_data.mask] = griddata(
            (xvalid, yvalid),
            arrvalid.flatten(),
            (xinvalid, yinvalid),
            method=method
        )

        # Store the interpolated band
        interpolated_arr[band] = interpolated_band

    return interpolated_arr


def interpolate_(masked_arr, method='linear'):
    xx, yy = np.meshgrid(np.arange(masked_arr.shape[1]), np.arange(masked_arr.shape[0]))
    # get only the valid values
    xvalid = xx[~masked_arr.mask]
    yvalid = yy[~masked_arr.mask]
    arrvalid = masked_arr[~masked_arr.mask]

    xinvalid = xx[masked_arr.mask]
    yinvalid = yy[masked_arr.mask]

    arr = masked_arr.data.copy()
    arr[masked_arr.mask] = griddata((xvalid, yvalid),
                                    arrvalid.flatten(),
                                    (xinvalid, yinvalid),
                                    method=method)

    return arr


def rotation_matrix(pitch, yaw, roll, deg=False, **kwargs):
    """
    Compute the rotation matrix representing the orientation of the sensor
    given the pitch, yaw, and roll angles.

    Arguments:
    pitch : float
        Pitch angle in radians.
    yaw : float
        Yaw angle in radians.
    roll : float
        Roll angle in radians.

    Returns:
    R : numpy array
        Rotation matrix representing the orientation of the sensor.
    """

    if deg:
        pitch, yaw, roll = np.array([pitch, yaw, roll]) * np.pi / 180

    # Calculate rotation matrices for yaw, pitch, and roll
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])

    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

    # Combine rotation matrices to get the overall rotation matrix
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))

    return R


def correct_(inp, corrs):
    out = dict()
    for k in inp.keys():
        if k in corrs:
            out[k] = inp[k] + corrs[k]
        else:
            out[k] = inp[k]

    return out


def compute_geometry(path, dem_arr, sensor):
    if sensor == 'FLUO':
        f = 16
        tilt = [(np.abs(i) / 192 * np.pi / 180) for i in range(-192, 192)]
        sensor_dirs = np.stack([np.array([0, -np.sign(i) * np.tan(f * np.abs(i) / 192 * np.pi / 180), -1])
                                for i in range(-192, 192)])

    elif sensor == 'DUAL':
        f = 16
        tilt = [(np.abs(i) / 192 * np.pi / 180) for i in range(-192, 192)]
        sensor_dirs = np.stack([np.array([0, -np.sign(i) * np.tan(f * np.abs(i) / 192 * np.pi / 180), -1])
                                for i in range(-192, 192)])

    # Get nav path
    try:
        nav_path = glob.glob(path[:-len('_radiance.dat')] + '_syncedNav.txt')[0]

    except IndexError as e:
        print('Could not find nav_path', path[:-len('_radiance.dat')] + '_syncedNav.txt')
        return None

    # get geomtrey corrections
    try:
        opt_file = pd.read_csv(nav_path[:-len('_syncedNav.txt')] + '.opt', skiprows=[1], sep='\s')

        corrs = dict(roll=float(opt_file.query('RAW_FILE == "ROLL_CORRECTION"').iloc[:, 3].item()),
                     pitch=float(opt_file.query('RAW_FILE == "PITCH_CORRECTION"').iloc[:, 3].item()),
                     yaw=float(opt_file.query('RAW_FILE == "YAW_CORRECTION"').iloc[:, 3].item()))

    except Exception as e:
        print(f'Could not find opt file for {os.path.basename(path)}')
        print(e)

        corrs = dict(roll=0,
                     pitch=0,
                     yaw=0)

    # Compute geometry
    nav_file = pd.read_csv(nav_path, skiprows=[0], header=None, sep='\s+')
    nav_file.columns = ['line', 'time', 'Latitude', 'Longitude', 'altitude', 'yaw', 'roll', 'pitch']

    h2alt = dem_arr
    h1alt = nav_file['altitude'].values[:, None]

    ############ Compute rotation matrices for each line
    Rs = [rotation_matrix(**correct_(row.to_dict(), corrs), deg=True) for i, row in nav_file.iterrows()]
    world_dirs = np.einsum('lij, kj -> lik', Rs, sensor_dirs)

    ############ Compute optical path length
    t = []
    for i, wdir in enumerate(world_dirs):
        h2alt_line = h2alt[i]
        h1alt_line = h1alt[i]
        d = h1alt_line - h2alt_line
        full_vecs = np.einsum('ik, k -> ik', wdir, d / np.abs(wdir[2]))
        t.append(np.sqrt((full_vecs ** 2).sum(0)))

    optical_path_lengths = np.stack(t)

    ############ Compute incidence angle
    invnorms = np.sqrt(1 / (world_dirs ** 2).sum(1))
    sc_prod = np.einsum('lik, i -> lk', world_dirs, np.array([0, 0, -1]))
    incidence_angle = np.degrees(np.arccos(np.einsum('li, li -> li', sc_prod, invnorms)))

    ############ Compute scene_azimuth
    invnorms = np.sqrt(1 / (world_dirs[:, :2] ** 2).sum(1))
    sc_prod = np.einsum('lik, i -> lk', world_dirs[:, :2], np.array([1, 0]))
    azimuth = np.degrees(np.arccos(np.einsum('li, li -> li', sc_prod, invnorms)))

    sc_prod = np.einsum('lik, i -> lk', world_dirs[:, :2], np.array([0, 1]))
    easting = np.sign(sc_prod)
    azimuth[easting < 0] = 360 - azimuth[easting < 0]

    ############ Compute solar azimuth
    lat, lon = nav_file.Latitude.values, nav_file.Longitude.values

    dt = '-'.join(np.array(os.path.basename(nav_path).split('-'))[[0, 2]])
    filter_ = '%Y%m%d-%H%M'

    dt = datetime.datetime.strptime(dt, filter_) - datetime.timedelta(hours=2)
    tz = pytz.timezone('UTC')
    dt = tz.localize(dt)

    solar_azimuth = sol.solar.get_azimuth(lat, lon, dt)[:, None].repeat(optical_path_lengths.shape[1], axis=1)
    solar_azimuth_inv = (solar_azimuth + 180) % 360

    solar_zenith = (90 - sol.solar.get_altitude(lat, lon, dt))[:, None].repeat(optical_path_lengths.shape[1], axis=1)

    ############ Get relative azimuth
    rel_azimuth = np.abs(solar_azimuth_inv - azimuth)
    rel_azimuth[rel_azimuth > 180] = 360 - rel_azimuth[rel_azimuth > 180]
    # rel_azimuth = 180 - rel_azimuth # forward scattering = 180 deg

    arrs = [dem_arr, optical_path_lengths, incidence_angle, solar_zenith, azimuth, solar_azimuth, rel_azimuth]
    geometry = np.stack(arrs)

    return geometry


@staticmethod
def georectify(arr, glt, do_interpolate=False):
    sensor_x, sensor_y = glt[1], glt[0]

    # --- Flatten GLT File & Filter Out Invalid Pixels ---
    # valid_mask = (sensor_x != 0) | (sensor_y != 0)  # Ignore (0,0) pixels
    # valid_idx = np.where(valid_mask)

    # Extract only valid (sensor_x, sensor_y) pairs
    # valid_x = sensor_x[valid_idx].flatten()
    # valid_y = sensor_y[valid_idx].flatten()
    # out_shape = sensor_x.shape

    # args_list = [[arr[..., b], valid_x, valid_y, valid_idx, out_shape] for b in range(arr.shape[-1])]
    # data, _ = execute_function(process_band, args_list, use_parallel=True)

    # Convert iterator to list
    # georegistered_array = np.stack(list(data), axis=-1)

    # Build Output Dataset
    fill_value = np.nan
    GLT_NODATA_VALUE = 0
    glt_array = np.nan_to_num(np.stack([sensor_x, sensor_y], axis=-1), nan=GLT_NODATA_VALUE).astype(int)

    out_ds = np.zeros((glt_array.shape[0], glt_array.shape[1], arr.shape[-1]), dtype=np.float32) + fill_value
    valid_glt = np.all(glt_array > GLT_NODATA_VALUE, axis=-1)

    # Adjust for One based Index
    glt_array[valid_glt] -= 1

    # Use indexing/broadcasting to populate array cells with 0 values
    out_ds[valid_glt, :] = arr[glt_array[valid_glt, 0], glt_array[valid_glt, 1], :]
    georegistered_array = out_ds

    if do_interpolate:
        arr = georegistered_array.transpose(2, 0, 1)
        masked_out = np.ma.masked_where(np.isnan(arr), arr)
        georegistered_array = interpolate(masked_out, method='linear', only_enclosed_nan=False).transpose(1, 2, 0)

    return georegistered_array


def write_tif(out_path, arr, **kwargs):
    metadata = {
        'driver': 'GTiff',
        'count': arr.shape[0],
        'height': arr.shape[1],
        'width': arr.shape[2],
        'dtype': arr.dtype,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    metadata.update(**kwargs)
    with rasterio.open(out_path, 'w', **metadata) as dst:
        for band_nr, band in enumerate(arr, start=1):
            dst.write(band, band_nr)


def write_tif_fluomapper_style(out_dir, arr, orig_path, data_path):
    name_dict = {0: '_dem_sensor.dat', 2: '_scene_incidence_sensor.dat', 6: '_parm1_sensor.dat'}
    out_dir = pjoin(out_dir, os.path.dirname(orig_path[len(f'{data_path}/'):]))
    print(f'Saving files for {os.path.basename(orig_path)[:-4]} to {out_dir}')
    for ind, name in name_dict.items():
        out_path = pjoin(out_dir, os.path.basename(orig_path)[:-len('.tif')] + name)
        write_tif(out_path, arr[[ind]])


def process_path(path, dem_path, out_dir, sensor, data_path=None, 
                 overwrite=False, process_style='spectral_foundation'):
    glt_path = path.replace('_radiance.dat', '_GLT.dat')
    dem, glt = get_in_glt_format(dem_path, path, glt_path)

    os.makedirs(out_dir, exist_ok=True)
    name = pbasename(path)[:-4] + '_sensorgeometry.tif'

    if process_style == 'spectral_foundation':
        if not os.path.exists(pjoin(out_dir, 'sensor_geom', name)) or overwrite:
            geom = compute_geometry(path, dem, sensor)
            write_tif(pjoin(out_dir, 'sensor_geom', name), geom)

            georect_geom = georectify(geom.transpose(1, 2, 0), glt).transpose(2, 0, 1)
            georect_geom = np.concatenate([georect_geom, glt], axis=0)

            name = pbasename(path)[:-4] + '_geometry.tif'
            crs = glt.rio.crs.to_wkt()
            transform = glt.rio.transform()
            write_tif(pjoin(out_dir, 'georect', name), georect_geom, crs=crs, transform=transform)

    elif process_style == 'fluomapper':
        geom = compute_geometry(path, dem, sensor)
        write_tif_fluomapper_style(out_dir, geom, path, data_path)

    else:
        raise NotImplementedError()


def crop_with_resampling(in_path, target_bbox, target_crs_string, target_transform, band=1):
    """
    Crop the DEM to the desired bounding box, output shape, and target resolution efficiently.

    Parameters:
    - in_path: Path to the DEM file.
    - target_bbox: (xmin, ymin, xmax, ymax) in pixel coordinates of the input DEM.
    - target_crs_string: The target CRS for reprojecting, as a WKT string.
    - target_transform: The desired Affine transform for the output crop.
    - band: The band number to read (default is 1).

    Returns:
    - reprojected_patch: The reprojected DEM patch that **aligns exactly** with target_bbox.
    """
    # Convert CRS from WKT string
    target_crs = pyproj.CRS.from_wkt(target_crs_string)

    # Ensure target_transform is Affine
    if not isinstance(target_transform, Affine):
        target_transform = Affine(*target_transform)

    # Open the input raster
    with rasterio.open(in_path) as infile:
        crs_in = infile.crs
        transform_in = infile.transform

        # Convert target_bbox (pixel coordinates in target) to world coordinates
        min_px, min_py, max_px, max_py = map(int, target_bbox)
        min_px = max(0, min_px)
        min_py = max(0, min_py)
        max_px = min_px + (max_px - min_px)
        max_py = min_py + (max_py - min_py)

        # Define the shape of the output patch
        patch_width = max_px - min_px
        patch_height = max_py - min_py
        output_shape = (patch_height, patch_width)

        if type(band) is int:
            band = [band]

        ret = []
        for b in band:
            # Create an empty array for the output patch
            reprojected_patch = np.empty(output_shape, dtype=infile.dtypes[0])

            target_window = rasterio.windows.Window.from_slices((min_px, max_px), (min_py, max_py))
            target_transform_win = rasterio.windows.transform(target_window, target_transform)

            reproject(
                source=rasterio.band(infile, b),
                destination=reprojected_patch,
                src_transform=transform_in,
                dst_transform=target_transform_win,
                src_crs=crs_in,
                dst_crs=target_crs,
                resampling=Resampling.cubic
            )

            ret.append(reprojected_patch)

    ret = np.stack(ret)
    return ret


def create_geometry_files(paths, dem_file, out_dir, data_path, sensor='FLUO',
                          saving_style='fluomapper', overwrite=True,
                          use_parallel=True, num_workers=20):

    if data_path is not None:
        data_path = data_path.rstrip('/')

    if type(paths[0]) is str:
        paths = [pjoin(f'{data_path}', p) for p in paths]
    else:
        paths = [pjoin(f'{data_path}', p) for p in paths]

    args_list = [[p, dem_file, out_dir, sensor, data_path, overwrite, saving_style] for p in paths]
    execute_function(process_path, args_list, use_parallel=use_parallel, num_workers=num_workers)
