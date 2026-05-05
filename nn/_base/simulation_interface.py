#!/usr/bin/env python

"""Fast forward simulator

This script provides the necessary functions to load and apply the learned
fast version of the forward simulator F: L_sen = F(atm,geo,sen,tar).

Available fast forward simulators:
--------------------------------------------------------------------------------
Sensor	Database		Model	Input
--------------------------------------------------------------------------------
DESIS	DB1 (mid-lat summer)	PRR:2:0	tar: rho(740 nm), drho/dl, LF(737 nm)
DESIS	DB1 (tropical)		PRR:2:0	tar: rho(740 nm), drho/dl, LF(737 nm)
DESIS	DB2 (mid-lat summer)	PRR:2:0	tar: rho(740 nm), drho/dl, LF(737 nm)
					atm: -AOT 550 nm, H2O
DESIS	DB2 (tropical)		PRR:2:0	tar: rho(740 nm), drho/dl, LF(737 nm)
					atm: -AOT 550 nm, H2O
DESIS	DB3 (mid-lat summer)	PRR:4:0	tar: rho(740 nm), drho/dl, LF(737 nm)
					atm: -AOT 550 nm, H2O
					geo: TA, SZA, RA, h_gnd
DESIS	DB3 (tropical)		PRR:4:0	tar: rho(740 nm), drho/dl, LF(737 nm)
					atm: -AOT 550 nm, H2O
					geo: TA, SZA, RA, h_gnd
DESIS	DB4 (mid-lat summer)	PRR:4:0	tar: rho(740 nm), drho/dl, LF(737 nm)
					atm: -AOT 550 nm, H2O
					geo: TA, SZA, RA, h_gnd
					sen: CW shift, FWHM shift
DESIS	DB4 (tropical)		PRR:4:0	tar: rho(740 nm), drho/dl, LF(737 nm)
					atm: -AOT 550 nm, H2O
					geo: TA, SZA, RA, h_gnd
					sen: CW shift, FWHM shift
DESIS	DB5 (mid-lat summer)	PRR:4:0	tar: rho(740 nm), drho/dl, e, LF(737 nm)
					atm: -AOT 550 nm, H2O
					geo: TA, SZA, RA, h_gnd
					sen: CW shift, FWHM shift
HyPlant	DB1 (675 m agl)		PRR:2:0	tar: rho(740 nm), drho/dl, LF(737 nm)
HyPlant	DB1 (1559 m agl)	PRR:2:0	tar: rho(740 nm), drho/dl, LF(737 nm)
HyPlant	DB2 (675 m agl)		PRR:4:0	tar: rho(740 nm), drho/dl, LF(737 nm)
					atm: g, -AOT 550 nm, H2O
HyPlant	DB2 (1559 m agl)	PRR:4:0	tar: rho(740 nm), drho/dl, LF(737 nm)
					atm: g, -AOT 550 nm, H2O
HyPlant	DB3 (675 m agl)		PRR:4:0	tar: rho(740 nm), drho/dl, LF(737 nm)
					atm: g, -AOT 550 nm, H2O
					geo: TA, SZA, RA, h_gnd, h_sen
HyPlant	DB3 (1559 m agl)	PRR:4:0	tar: rho(740 nm), drho/dl, LF(737 nm)
					atm: g, -AOT 550 nm, H2O
					geo: TA, SZA, RA, h_gnd, h_sen
HyPlant	DB4 (675 m agl)		PRR:4:0	tar: rho(740 nm), drho/dl, LF(737 nm)
					atm: g, -AOT 550 nm, H2O
					geo: TA, SZA, RA, h_gnd, h_sen
					sen: CW shift, FWHM shift
HyPlant	DB4 (1559 m agl)	PRR:4:0	tar: rho(740 nm), drho/dl, LF(737 nm)
					atm: g, -AOT 550 nm, H2O
					geo: TA, SZA, RA, h_gnd, h_sen
					sen: CW shift, FWHM shift
HyPlant	DB5 (675 m agl)		PRR:4:0	tar: rho(740 nm), drho/dl, e, LF(737 nm)
					atm: -AOT 550 nm, H2O
					geo: TA, SZA, RA, h_gnd, h_sen
					sen: CW shift, FWHM shift

Input parameters:
--------------------------------------------------------------------------------
Parameter	Unit		Databases	Description
--------------------------------------------------------------------------------
TA 		deg		DESIS   DB3-5 	Tilt angle at the sensor.
				HyPlant DB3-5
SZA		deg		DESIS   DB3-5	Solar zenith angle at the
				HyPlant DB3-5	 target.
RAA		deg		DESIS   DB3-5	Relative azimuth angle between
				HyPlant DB3-5	Sun and LOS at the target.
h_gnd		km		DESIS   DB3-5	Ground altitude.
				HyPlant DB3-5
h_sen		km		HyPlant DB3-5	Sensor altitude. Sensor altitude
						wrt ground is h_sen-h_gnd.
g		-		HyPlant DB2-4	Aerosol asymmetry parameter.
-AOT 550 nm	-		DESIS   DB2-5	Negative aerosol optical
				HyPlant DB2-5	thickness at 550 nm.
H2O		cm		DESIS   DB2-5	Water column density.
				HyPlant DB2-5
CW shift	nm		DESIS   DB4-5	Sensor central wavelength shift.
				HyPlant DB4-5
FWHM shift	nm		DESIS   DB4-5	Sensor full width half maximum
				HyPlant DB4-5	shift.
rho(740 nm)	-		DESIS   DB1-5	Ground reflectance at 740 nm.
				HyPlant DB1-5
drho/dl		1/nm		DESIS   DB1-5	Ground reflectance slope.
				HyPlant DB1-5
e		-		DESIS   DB5	Ratio of reflectance slopes.
				HyPlant DB5
LF(737 nm)	mW/m2/sr/nm	DESIS   DB1-5	Ground fluorescence at 737 nm.
				HyPlant DB1-5

Output:
--------------------------------------------------------------------------------
Sensor	Quantity 		Units			Bands	
--------------------------------------------------------------------------------
DESIS	at-sensor radiance	mW/cm2/sr/microm	13
HyPlant	at-sensor radiance	mW/cm2/sr/microm	349

Notes:
1. The input parameter ranges for each database are printed out when using this
script. The evaluation outside these ranges is not tested nor supported.
2. For further details, please refer to the slides of the DLR+FZJ meetings
on 11.03.2022 12.08.2022 and 28.10.2022.


Usage 1: Import in own code (requires python3, numpy, pickle)
         Example: Evaluate fast forward simulator for HyPlant DB4 (1559 m agl)

                  import numpy as np
                  import fast_fwd_sim

                  # Load fast forward simulator
                  f = '<...>/HyPlant_SENSOR_DB4_1559_PRR:4:0_FHR_model.pkl'
                  ffs = fast_fwd_sim.load(f)

                  # List of input points (here just one point)
                  ip = np.array([[
                    0.0, 25., 90., 0., 1.559, # TA, SZA, RAA, h_gnd, h_sen
                    0.5, -0.1, 2.0,           # g, -AOT 550 nm, H2O
                    0.0, 0.0,                 # CW shift, FWHM shift
                    0.3,0.0,4.0]])            # rho(740 nm), drho/dl, LF(737 nm)

                  # Evaluate model in input points
                  lsen = fast_fwd_sim.evaluate(ffs, ip)
                  print(lsen.shape) # shape: [#points]x[#bands]

                  # Evaluate model first derivative in input points
                  lsenp = fast_fwd_sim.evaluate_deriv(ffs, ip)
                  print(lsenp.shape) # shape: [#points]x[#bands]x[#input_dim]


Usage 2: Run script for testing (internal use only)
         python simulation_interface.py [output_dir]

"""

import sys
import os
import numpy as np
import pickle



# Global variables
supported_models = ['PRR']
supported_sensors = ['DESIS', 'HyPlant']
supported_dbs = ['DB1', 'DB2', 'DB3', 'DB4', 'DB5', 'DB51', 'DB6', 'DB7']
supported_db_extra = {'DESIS': ['ATM_MIDLAT_SUMMER', 'ATM_TROPICAL'],\
                      'HyPlant': ['675', '1559', '']}

# Database output dimensions
db_output = {'DESIS': 13, 'HyPlant': 449}

# Database input ranges
db_input = dict()
db_input['DESIS_DB1'] =   [['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.0008)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]
db_input['DESIS_DB2'] =   [['-AOT 550 nm []',           (-0.30, -0.05)],\
                           ['H2O [cm]',                 (0.3, 5.0)],\
                           ['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.0008)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]
db_input['DESIS_DB3'] =   [['TA [deg]',                 (0.0, 25.0)],\
                           ['SZA [deg]',                (0.0, 55.0)],\
                           ['RAA [deg]',                (0.0, 180.0)],\
                           ['h_gnd [km]',               (0.0, 0.6)],\
                           ['-AOT 550 nm []',           (-0.30, -0.05)],\
                           ['H2O [cm]',                 (0.3, 5.0)],\
                           ['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.0008)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]
db_input['DESIS_DB4'] =   [['TA [deg]',                 (0.0, 25.0)],\
                           ['SZA [deg]',                (0.0, 55.0)],\
                           ['RAA [deg]',                (0.0, 180.0)],\
                           ['h_gnd [km]',               (0.0, 0.6)],\
                           ['-AOT 550 nm []',           (-0.30, -0.05)],\
                           ['H2O [cm]',                 (0.3, 5.0)],\
                           ['CW shift [nm]',            (-1.25, 0.75)],\
                           ['FWHM shift [nm]',          (-0.21, 0.15)],\
                           ['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.0008)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]
db_input['DESIS_DB5'] =   [['TA [deg]',                 (0.0, 25.0)],\
                           ['SZA [deg]',                (0.0, 55.0)],\
                           ['RAA [deg]',                (0.0, 180.0)],\
                           ['h_gnd [km]',               (0.0, 0.6)],\
                           ['-AOT 550 nm []',           (-0.30, -0.02)],\
                           ['H2O [cm]',                 (0.3, 5.0)],\
                           ['CW shift [nm]',            (-1.25, 0.75)],\
                           ['FWHM shift [nm]',          (-0.21, 0.15)],\
                           ['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.012)],\
                           ['e',                        (0.0, 1.0)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]
db_input['DESIS_DB51'] =  [['TA [deg]',                 (0.0, 25.0)],\
                           ['SZA [deg]',                (0.0, 55.0)],\
                           ['RAA [deg]',                (0.0, 180.0)],\
                           ['h_gnd [km]',               (0.0, 0.6)],\
                           ['-AOT 550 nm []',           (-0.30, -0.02)],\
                           ['H2O [cm]',                 (0.3, 5.0)],\
                           ['CW shift [nm]',            (-1.75, 1.25)],\
                           ['FWHM shift [nm]',          (-0.30, 0.30)],\
                           ['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.012)],\
                           ['e',                        (0.0, 1.0)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]
db_input['HyPlant_DB1'] = [['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.0008)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]
db_input['HyPlant_DB2'] = [['g []',                     (-1, 1)],\
                           ['-AOT 550 nm []',           (-0.40, -0.05)],\
                           ['H2O [cm]',                 (0.3, 3.0)],\
                           ['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.0008)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]
db_input['HyPlant_DB3_675'] = [\
                           ['TA [deg]',                 (0.0, 20.0)],\
                           ['SZA [deg]',                (20.0, 55.0)],\
                           ['RAA [deg]',                (0.0, 180.0)],\
                           ['h_gnd [km]',               (0.0, 0.3)],\
                           ['h_sen [km]',               (0.659, 0.991)],\
                           ['g []',                     (-0.99, 1)],\
                           ['-AOT 550 nm []',           (-0.40, -0.05)],\
                           ['H2O [cm]',                 (0.3, 3.0)],\
                           ['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.0008)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]
db_input['HyPlant_DB3_1559'] = [\
                           ['TA [deg]',                 (0.0, 20.0)],\
                           ['SZA [deg]',                (20.0, 55.0)],\
                           ['RAA [deg]',                (0.0, 180.0)],\
                           ['h_gnd [km]',               (0.0, 0.3)],\
                           ['h_sen [km]',               (1.543, 1.898)],\
                           ['g []',                     (-0.99, 1)],\
                           ['-AOT 550 nm []',           (-0.40, -0.05)],\
                           ['H2O [cm]',                 (0.3, 3.0)],\
                           ['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.0008)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]
db_input['HyPlant_DB4_675'] = [\
                           ['TA [deg]',                 (0.0, 20.0)],\
                           ['SZA [deg]',                (20.0, 55.0)],\
                           ['RAA [deg]',                (0.0, 180.0)],\
                           ['h_gnd [km]',               (0.0, 0.3)],\
                           ['h_sen [km]',               (0.659, 0.991)],\
                           ['g []',                     (-0.99, 1)],\
                           ['-AOT 550 nm []',           (-0.40, -0.05)],\
                           ['H2O [cm]',                 (0.3, 3.0)],\
                           ['CW shift [nm]',            (-0.080, 0.023)],\
                           ['FWHM shift [nm]',          (-0.04, 0.04)],\
                           ['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.0008)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]
db_input['HyPlant_DB4_1559'] = [\
                           ['TA [deg]',                 (0.0, 20.0)],\
                           ['SZA [deg]',                (20.0, 55.0)],\
                           ['RAA [deg]',                (0.0, 180.0)],\
                           ['h_gnd [km]',               (0.0, 0.3)],\
                           ['h_sen [km]',               (1.543, 1.898)],\
                           ['g []',                     (-0.99, 1)],\
                           ['-AOT 550 nm []',           (-0.40, -0.05)],\
                           ['H2O [cm]',                 (0.3, 3.0)],\
                           ['CW shift [nm]',            (-0.080, 0.023)],\
                           ['FWHM shift [nm]',          (-0.04, 0.04)],\
                           ['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.0008)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]
db_input['HyPlant_DB5_675'] = [\
                           ['TA [deg]',                 (0.0, 20.0)],\
                           ['SZA [deg]',                (20.0, 55.0)],\
                           ['RAA [deg]',                (0.0, 180.0)],\
                           ['h_gnd [km]',               (0.0, 0.3)],\
                           ['h_sen [km]',               (0.659, 0.991)],\
                           ['-AOT 550 nm []',           (-0.30, -0.02)],\
                           ['H2O [cm]',                 (0.3, 3.0)],\
                           ['CW shift [nm]',            (-0.08, 0.08)],\
                           ['FWHM shift [nm]',          (-0.04, 0.04)],\
                           ['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.012)],\
                           ['e',                        (0.0, 1.0)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]
db_input['HyPlant_DB51_675'] = db_input['HyPlant_DB5_675']

db_input['HyPlant_DB6'] = [\
                           ['TA [deg]',                 (0.0, 25.0)],\
                           ['SZA [deg]',                (20.0, 55.0)],\
                           ['RAA [deg]',                (0.0, 180.0)],\
                           ['h_gnd [km]',               (0.0, 0.730)],\
                           ['h_sen [km]',               (0.400, 1.580)],\
                           ['-AOT 550 nm []',           (-0.30, -0.02)],\
                           ['H2O [cm]',                 (0.3, 3.0)],\
                           ['CW shift [nm]',            (-0.08, 0.08)],\
                           ['FWHM shift [nm]',          (-0.04, 0.04)],\
                           ['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.012)],\
                           ['e',                        (0.0, 1.0)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]


db_input['HyPlant_DB7'] = [\
                           ['TA [deg]',                 (0.0, 25.0)],\
                           ['SZA [deg]',                (20.0, 55.0)],\
                           ['RAA [deg]',                (0.0, 180.0)],\
                           ['h_gnd [km]',               (0.0, 0.760)],\
                           ['h_sen [km]',               (0.200, 2.860)],\
                           ['-AOT 550 nm []',           (-0.30, -0.02)],\
                           ['H2O [cm]',                 (0.3, 3.0)],\
                           ['CW shift [nm]',            (-0.08, 0.08)],\
                           ['FWHM shift [nm]',          (-0.04, 0.04)],\
                           ['rho(740 nm) []',           (0.05, 0.60)],\
                           ['drho/dl [1/nm]',           (0.0, 0.012)],\
                           ['e',                        (0.0, 1.0)],\
                           ['LF(737 nm) [mW/m2/sr/nm]', (0.0, 8.0)]]

# Database baseline input point
db_baseline = dict()
db_baseline['DESIS_DB1'] =   np.array([[0.3, 0.0, 4.0]])
db_baseline['DESIS_DB2'] =   np.array([[-0.1, 2.0, 0.3, 0.0, 4.0]])
db_baseline['DESIS_DB3'] =   np.array([[0.0, 22.5, 90, 0.0, -0.175, 2.65,\
                                        0.3, 0.0, 4.0]])
db_baseline['DESIS_DB4'] =   np.array([[0.0, 22.5, 90, 0.0, -0.175, 2.65,\
                                        0.0, 0.0, 0.3, 0.0, 4.0]])
db_baseline['DESIS_DB5'] =   np.array([[0.0, 22.5, 90, 0.0, -0.175, 2.65,\
                                        0.0, 0.0, 0.3, 0.0, 1.0, 4.0]])
db_baseline['DESIS_DB51'] =   np.array([[0.0, 22.5, 90, 0.0, -0.175, 2.65,\
                                        0.0, 0.0, 0.3, 0.0, 1.0, 4.0]])

db_baseline['HyPlant_DB1'] = np.array([[0.3, 0.0, 4.0]])
db_baseline['HyPlant_DB2'] = np.array([[0.5, -0.1, 2.0, 0.3, 0.0, 4.0]])
db_baseline['HyPlant_DB3_675'] = \
                             np.array([[0.0, 37.5, 90, 0.0, 0.675,\
                                        1.0, -0.225, 1.65,\
                                        0.3, 0.0, 4.0]])
db_baseline['HyPlant_DB3_1559'] = \
                             np.array([[0.0, 37.5, 90, 0.0, 1.5705,\
                                        1.0, -0.225, 1.65,\
                                        0.3, 0.0, 4.0]])
db_baseline['HyPlant_DB4_675'] = \
                             np.array([[0.0, 37.5, 90, 0.0, 0.675,\
                                        1.0, -0.225, 1.65,\
                                        0.0, 0.0, 0.3, 0.0, 4.0]])
db_baseline['HyPlant_DB4_1559'] = \
                             np.array([[0.0, 37.5, 90, 0.0, 1.5705,\
                                        1.0, -0.225, 1.65,\
                                        0.0, 0.0, 0.3, 0.0, 4.0]])
db_baseline['HyPlant_DB5_675'] = \
                             np.array([[0.0, 37.5, 90, 0.0, 0.675,\
                                        -0.225, 1.65,\
                                        0.0, 0.0, 0.3, 0.0, 1.0, 4.0]])

db_baseline['HyPlant_DB51_675'] = db_baseline['HyPlant_DB5_675']
db_baseline['HyPlant_DB6'] = db_baseline['HyPlant_DB5_675']
db_baseline['HyPlant_DB7'] = db_baseline['HyPlant_DB5_675']

#######################################
# Load functions
#######################################


def load(f):
    """Load fast forward simulator.

    :param f: file with fast forward simulator

    :returns: fast forward simulator model, input ranges and auxiliary data
    """

    print('')
    print('-----------------------------------')
    print('Loading fast forward simulator')
    print('-----------------------------------')
    print(' file: ', f)

    # Auxiliary data
    fb = os.path.basename(f)
    sensor = fb.split('_')[0]                # DESIS, HyPlant
    db = fb.split('_')[2]                    # DB1, DB2, DB3, DB4
    db_extra = '_'.join(fb.split('_')[3:-3]) # e.g. ATM_TROPICAL, 675
    mod = fb.split('_')[-3]                  # e.g. PRR:4:0
    # To avoid problems with :, models can be renamed with - (e.g. PRR-4-0)
    mod = mod.replace('-', ':')#.replace('_', ':')

    td = fb.split('_')[-2]                   # e.g. FHR
    bands = db_output[sensor]                # e.g. 13
    print(' sensor:\t', sensor)
    print(' database:\t', db + ' ' + db_extra)
    print(' model:\t\t', mod)
    print(' training data:\t', td)
    ffsaux = {'sensor': sensor, 'db': db, 'db_extra': db_extra, 'model': mod,\
              'bands': bands}

    # Load learned model
    print(' loading learned model...')
    ffsmod = load_model(f, mod)

    # Input ranges
    print(f' loading input ranges for db {db}...')
    ffsir = load_input_ranges(sensor, db, db_extra)

    return (ffsmod, ffsir, ffsaux)


def load_model(f, mod):
    """Load fast forward simulator model.

    :param f: file with fast forward simulator
    :param mod: fast forward simulator model (e.g. PRR:4:0)

    :returns: fast forward simulator model
    """

    # Model type
    modt = mod.split(':')[0]

    # Supported models
    if modt not in supported_models:
        print('Error: Model not supported: ', modt)
        sys.exit(1)

    # Load model
    if modt not in ['kerasNN']:
        fmod = open(f, 'rb')
        ffsmod = pickle.load(fmod)
        fmod.close()

    return ffsmod


def load_input_ranges(sensor, db, db_extra):
    """Load input ranges for given sensor database.

    :param sensor: sensor name (DESIS, HyPlant)
    :param db: database identifier (DB1, DB2, DB3, DB4)
    :param db_extra database extra identifier (e.g. ATM_TROPICAL, 675)

    :returns: input ranges for database
    """
    # Supported sensors and databases
    if sensor not in supported_sensors:
        print('Error: Sensor not supported: ', sensor)
        sys.exit(1)
    if db not in supported_dbs:
        print('Error: Database not supported: ', db)
        sys.exit(1)
    if db_extra not in supported_db_extra[sensor]:
        print('Error: Database extra not supported: ', db_extra)
        sys.exit(1)

    # Load ranges
    ident = sensor + '_' + db
    if sensor == 'HyPlant' and db in ['DB3', 'DB4', 'DB5']:
        ident = ident + '_' + db_extra 
    ffsir = db_input[ident]

    # Print summary
    print(' input ranges:')
    for ir in ffsir:
        var, m, M = ir[0], str(ir[1][0]), str(ir[1][1])
        ts = '\t\t\t\t'
        if len(var) > 5:
            ts = '\t\t\t'
        if len(var) > 10:
            ts = '\t\t'
        if len(var) > 20:
            ts = '\t'

        s = '  ' + var + ':' + ts + m + '\t' + M
        print(s)

    return ffsir



#######################################
# Evaluate functions
#######################################


def evaluate(ffs, ip):
    """Evaluate fast forward simulator on set of input points.

    :param ffs: fast forward simulator model, input ranges and auxiliary data
    :param ip: input points [#samples]x[#input_dim] (numpy array)

    :returns: at-sensor radiances for each input point [#samples]x[#bands]
    """

    print('')
    print('-----------------------------------')
    print('Evaluating fast forward simulator')
    print('-----------------------------------')

    # Unpack fast forward simulator
    ffsmod, ffsir, ffsaux = ffs

    # Normalise inputs
    print(' normalising input points...')
    ipn = normalise_inputs(ffsir, ffsaux, ip)

    # Evaluate fast forward simulator
    print(' evaluating...')
    lsen = ffsmod.predict(ipn)

    return lsen


def normalise_inputs(ffsir, ffsaux, ip):
    """Normalise input points to database ranges.

    :param ffsir: fast forward simulator input ranges
    :param ffsaux: fast forward simulator auxiliary data
    :param ip: input points [#samples]x[#input_dim] (numpy array)

    :returns: normalised input points [#samples]x[#input_dim] (numpy array)
    """

    # Check dimensions of input points
    if len(ffsir) != ip.shape[1]:
        print('Error: Wrong input dimensions: ', ip.shape)
        sys.exit(1)

    # Normalise input points to [0,1]
    hh = dict()
    ipn = np.zeros_like(ip)
    for ii, ir in enumerate(ffsir):
        var = ir[0]
        m, M = ir[1]
        ipn[:, ii] = (ip[:,ii]-m) / (M-m)

        # Check outliers
        if np.sum(ipn[:, ii] < 0) or np.sum(ipn[:, ii] > 1):
            print('Warning: There are values outside the range of ' +\
                   var + ': ', m, ' - ', M, '. The output of the fast ' +\
                  'forward simulator for those values is extrapolated and ' +\
                  'should not be trusted.')

        # Save h_gnd, h_sen
        if var in ['h_gnd [km]', 'h_sen [km]']:
            hh[var] = ip[:,ii]


    # Note that for HyPlant DB3-4, the valid range of sensor altitudes is
    # defined wrt the ground.
    if ffsaux['sensor'] == 'HyPlant' and ffsaux['db'] in ['DB3', 'DB4']:
        hsengnd = hh['h_sen [km]'] - hh['h_gnd [km]']
        if ffsaux['db_extra'] == '675': m, M = 0.659, 0.691
        elif ffsaux['db_extra'] == '1559': m, M = 1.543, 1.598
        if np.sum(hsengnd < m)>0 or np.sum(hsengnd > M)>0:
            print('Warning: There are values outside the range of sensor ' +\
                  'altitude wrt ground:', m, ' - ', M, '. The output of the ' +\
                  'fast forward simulator for those values is extrapolated ' +\
                  'and should not be trusted.')

    return ipn



#######################################
# Derivative functions
#######################################


def load_deriv_slow(ffs):
    """Compute first derivative of fast forward simulator on the inputs. This
       uses sympy which is exact, but slow for large dimensions.

    :param ffs: fast forward simulator model, input ranges and auxiliary data

    :returns: first derivative of fast forward simulator model (polynomial list
              of dimensions [#bands]x[#input_dim])
              
    """

    print('')
    print('-----------------------------------')
    print('Computing derivative of fast forward simulator (slow)')
    print('-----------------------------------')

    # Unpack fast forward simulator
    ffsmod, ffsir, ffsaux = ffs

    # Supported models
    modt = ffsaux['model'].split(':')[0]
    if modt not in supported_models:
        print('Error: Model not supported: ', modt)
        sys.exit(1)

    # Polynomial ridge regression
    if modt == 'PRR':

        # Import sympy for polynomial differentiation
        import sympy as sp

        # Powers of polynomial features
        # 0, x1, x2, x3, ..., x1*x2, ...
        powers = ffsmod.named_steps['poly'].powers_
        p, m = powers.shape # #polynomial_features, #input_dim
        #print( ' polynomial features powers: ', powers)
        # shape: [#polynomial_features]x[#input_dim]

        # Coefficients fitted in regression
        coefs = ffsmod.named_steps['ridge'].coef_
        b = coefs.shape[0] # #bands
        #print( ' coefficients: ', coefs)
        # shape: [#bands]x[#polynomial_features]


        # Form polynom y = c1*x1 + ... + c12 * x1*x2 + ... for each band
        # Note that the intercept is ignored because it gives a null derivative.

        # List of symbols x1, ..., xm
        x = dict()
        for im in range(m): x[im] = sp.Symbol('x' + str(im))

        # Loop over bands
        poly_deriv = list()
        for ib in range(b):
            #print('band ', ib)

            # Polynom string
            polystr = ''

            # Loop over polynomial features
            for ip in range(p):
                cf = coefs[ib, ip]
                if cf == 0.0: continue
                coefstr = '(' + str(cf) + ')*'
                polystr = polystr + coefstr

                # Loop over input dimensions
                for im in range(m):
                    pw = powers[ip,im]
                    if pw == 0: continue
                    elif pw == 1:
                        powstr = 'x' + str(im) + '*'
                    else:
                        powstr = 'x' + str(im) + '**' + str(pw) + '*'
                    polystr = polystr + powstr

                # Remove last *
                polystr = polystr[:-1]

                polystr = polystr + '+'

            # Remove last +
            polystr = polystr[:-1]
            #print(polystr)

            # Define polynomial (slow)
            poly = sp.Poly(polystr)
            #print(poly)

            # Compute derivative on each input dimension
            poly_deriv_aux = list()
            for im in range(m):
                poly_deriv_aux.append(poly.diff(im))
            poly_deriv.append(poly_deriv_aux)

        ## Print derivative polynomials
        #for il in range(len(poly_deriv)):
        #    print('---- band ' + str(il) + ' ----')
        #    for ill in range(len(poly_deriv[il])):
        #        print(poly_deriv[il][ill])
        ## shape: [#bands]x[#input_dim]

    return poly_deriv


def evaluate_deriv_slow(ffs, ffs_deriv, ip):
    """Evaluate first derivative of fast forward simulator on set of input
       points. This uses sympy which is exactly, but slow for large dimensions.

    :param ffs: fast forward simulator model, input ranges and auxiliary data
    :param ffs_deriv: fast forward simulator model derivative
    :param ip: input points [#samples]x[#input_dim] (numpy array)

    :returns: evaluated first derivative of fast forward simulator model
              [#samples]x[#bands]x[#input_dim]
    """

    print('')
    print('-----------------------------------')
    print('Evaluating derivative of fast forward simulator (slow)')
    print('-----------------------------------')

    # Unpack fast forward simulator
    ffsmod, ffsir, ffsaux = ffs

    # Normalise inputs
    print(' normalising input points...')
    ipn = normalise_inputs(ffsir, ffsaux, ip)

    # Dimensions
    s, m = ip.shape
    b = len(ffs_deriv)

    # Evaluate fast forward simulator derivative
    print(' evaluating derivative...')
    deriv = np.empty((s,b,m), dtype='float')
    for iis in range(s):
        point = ipn[iis,:]
        for ib in range(b):
            for im in range(m):
                deriv[iis,ib,im] = ffs_deriv[ib][im](*point)

    return deriv


def evaluate_deriv(ffs, ip):
    """Evaluate first derivative of fast forward simulator on set of input
       points. This is a fast one-step alternative to using sympy to load
       and evaluate the derivative.

    :param ffs: fast forward simulator model, input ranges and auxiliary data
    :param ip: input points [#samples]x[#input_dim] (numpy array)

    :returns: evaluated first derivative of fast forward simulator model
              [#samples]x[#bands]x[#input_dim]
    """

    print('')
    print('-----------------------------------')
    print('Evaluating derivative of fast forward simulator (fast)')
    print('-----------------------------------')

    # Unpack fast forward simulator
    ffsmod, ffsir, ffsaux = ffs

    # Dimensions
    s, m = ip.shape
    b = ffsaux['bands']

    # Supported models
    modt = ffsaux['model'].split(':')[0]
    if modt not in supported_models:
        print('Error: Model not supported: ', modt)
        sys.exit(1)

    # Normalise inputs
    print(' normalising input points...')
    ipn = normalise_inputs(ffsir, ffsaux, ip)


    # Compute derivative
    print(' computing derivative...')

    # Polynomial ridge regression
    if modt == 'PRR':

        # Powers of polynomial features
        # 0, x1, x2, x3, ..., x1*x2, ...
        powers = ffsmod.named_steps['poly'].powers_
        p, m = powers.shape # #polynomial_features, #input_dim
        #print( ' polynomial features powers: ', powers)
        # shape: [#polynomial_features]x[#input_dim]

        # Coefficients fitted in regression
        coefs = ffsmod.named_steps['ridge'].coef_
        b = coefs.shape[0] # #bands
        #print( ' coefficients: ', coefs)
        # shape: [#bands]x[#polynomial_features]

        # Model for given band ib
        # c: coefs, p: powers
        # y_ib = sum_ip( c_(ib,ip) * x_0^(p_(ip,0))
        #                          * x_1^(p_(ip,1)) * (...) )

        # Model derivative for band ib and input dimension im
        # dy_ib/dx_im = sum_ip( c(ib,ip) * p_(ip,im)
        #                                * x_0^(p_(ip,0)) * (...)
        #                                * x_im^(p_(ip,im)-1) * (...) )

        # TODO: This derivative computation can probably be made faster.

        # Derivative
        deriv = np.zeros((s,b,m), dtype='float')

        # Loop over input points
        for iis in range(s):
            x = ipn[iis,:] # x0, x1, ..

            # Loop over bands
            for ib in range(b):

                # Loop over input dimensions
                for im in range(m):

                    # Evaluate derivative of band ib on im feature

                    # Loop over polynomial features
                    for ip in range(p):
                        cf = coefs[ib,ip]
                        pw = powers[ip,im]
                        if cf == 0 or pw == 0: continue

                        # Loop over input dimensions
                        prodx = 1
                        for im2 in range(m):
                            if im2 == im: continue
                            prodx *= x[im2] ** powers[ip,im2]

                        deriv[iis,ib,im] += cf * pw * x[im]**(pw-1) * prodx

    return deriv



#######################################
# Cross-check with DB
#######################################


def cross_check(od):
    """Cross-check of fast forward simulator and databases.

    :param od: output directory
    """

    # Imports here so that not required to run other functions
    import glob
    import matplotlib.pyplot as plt
    import matplotlib

    # Folder with fast forward simulator
    ddffs = '/data/local/forward_learning/learned_models/'

    # Loop over sensors and databases
    for sensor in supported_sensors:
        for db in supported_dbs:
            for db_extra in supported_db_extra[sensor]:
                s = sensor + '_SENSOR_' + db + '_' + db_extra
                print(s)

                # DB extra string
                if db_extra == 'ATM_MIDLAT_SUMMER':
                    db_extra_name = 'mid-lat summer'
                elif db_extra == 'ATM_TROPICAL':
                    db_extra_name = 'tropical'
                elif db_extra == '675':
                    db_extra_name = '675 m agl'
                elif db_extra == '1559':
                    db_extra_name = '1559 m agl'

                # Baseline input point and input names
                ident = sensor + '_' + db
                if sensor == 'HyPlant' and db in ['DB3', 'DB4']:
                    ident = ident + '_' + db_extra
                ip = db_baseline[ident]
                print('Baseline point: ', ip)

                # Names of inputs
                ip_names = list()
                for ijk in range(len(db_input[ident])):
                     nm = db_input[ident][ijk][0].split('[')[0][:-1]
                     ip_names.append(nm)
                print('input names: ', ip_names)

                # Evaluate fast forward simulator
                f = glob.glob(ddffs + '/' + s + '*_model.pkl')[0]
                mod = os.path.basename(f).split('_')[-3]
                ffs = load(f)
                lsen_ffs = evaluate(ffs, ip)[0]
                #print(lsen_ffs.shape, lsen_ffs)
                lsen_deriv = evaluate_deriv(ffs, ip)[0] # derivative
                #print(lsen_deriv.shape, lsen_deriv)

                # Retrieve from database
                lsen_db = get_from_db(sensor, db, db_extra, ip)[0]
                #print(lsen_db.shape, lsen_db)

                # Plot comparison DB vs fast forward model
                nbands = lsen_db.shape[0]
                fig, ax = plt.subplots(figsize=(8.5, 7.5))
                gs = matplotlib.gridspec.GridSpec(2, 1)
                fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1,\
                                    top=0.9, hspace=0.05)
                ax00 = plt.subplot(gs[0,0])
                ax00.set(xticklabels=[])
                #plt.xlabel('band')
                plt.ylabel(r'$L_{\mathrm{sen}}$ [mW/cm$^2$/sr/$\mu$m]')
                plt.xlim(-0.25, nbands-1+0.25)
                plt.ylim(0, 25)
                ts = sensor + ' ' + db + ' (' + db_extra_name + ')'
                plt.text(-0.25+0.01*(nbands-1), 23.7, ts, fontsize=10)
                ts = mod
                plt.text(-0.25+0.01*(nbands-1), 22.4, ts, fontsize=10)
                plt.plot(lsen_db, 'k', label='true')
                plt.plot(lsen_ffs, 'r', label='predicted')
                plt.legend(bbox_to_anchor=(0.98, 0.98), loc=1,\
                           borderaxespad=0., fontsize=8)
                ax10 = plt.subplot(gs[1,0])
                plt.xlabel('band')
                plt.ylabel(r'$\Delta L_{\mathrm{sen}}$ [mW/cm$^2$/sr/$\mu$m]')
                plt.xlim(-0.25, nbands-1+0.25)
                #plt.ylim(0, 2e-4)
                plt.ylim(1e-7,1e-0)
                plt.yscale('log')
                plt.plot(np.abs(lsen_ffs-lsen_db), 'k')
                fig.savefig(od + '/' + s + '_comp.png', bbox_inches='tight')


                # Plot first derivatives
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.xlabel('band')
                plt.ylabel(r'$\partial L_{\mathrm{sen}} / \partial x$ ' +\
                            '[mW/cm$^2$/sr/$\mu$m]')
                plt.xlim(-0.25, nbands-1+0.25)
                plt.ylim(-5, 25)
                ts = sensor + ' ' + db + ' (' + db_extra_name + ')'
                plt.text(-0.25+0.01*(nbands-1), 24, ts, fontsize=10)
                ts = mod
                plt.text(-0.25+0.01*(nbands-1), 23.05, ts, fontsize=10)
                cols = ['k', 'r', 'g', 'b', 'm', 'c', 'orange',\
                        'k--', 'r--', 'g--', 'b--', 'm--', 'c--']
                cols = cols[:len(ip_names)][::-1]
                for ift, inm in enumerate(ip_names):
                    plt.plot(lsen_deriv[:,ift], cols[ift], label=inm)
                plt.legend(bbox_to_anchor=(0.98, 0.98), loc=1,\
                           borderaxespad=0., fontsize=8)
                fig.savefig(od + '/' + s + '_deriv.png', bbox_inches='tight')

                #plt.show()
                plt.close()


def get_from_db(sensor, db, db_extra, ip):
    """Retrieve spectrum from database.

    :param sensor: sensor name (DESIS, HyPlant)
    :param db: database identifier (DB1, DB2, DB3, DB4)
    :param db_extra database extra identifier (e.g. ATM_TROPICAL, 675)
    :param ip: input points [#samples]x[#input_dim] (numpy array)

    :returns: at-sensor radiance spectrum [#samples]x[#bands]
    """

    # Database features
    if sensor == 'DESIS':
        if db == 'DB1':
            features = ['rho:0', 'rho:1', 'lfluo:0']
        elif db == 'DB2':
            features = ['vis', 'h2ostr', 'rho:0', 'rho:1', 'lfluo:0']
        elif db == 'DB3':
            features = ['tilt', 'parm2', 'parm1', 'h2alt',\
                        'vis', 'h2ostr', 'rho:0', 'rho:1', 'lfluo:0']
        elif db == 'DB4':
            features = ['tilt', 'parm2', 'parm1', 'h2alt',\
                        'vis', 'h2ostr', 'cw_shift', 'fwhm_shift',\
                        'rho:0', 'rho:1', 'lfluo:0']
    elif sensor == 'HyPlant':
        if db == 'DB1':
            features = ['rho:0', 'rho:1', 'lfluo:0']
        elif db == 'DB2':
            features = ['hgpf', 'vis', 'h2ostr', 'rho:0', 'rho:1', 'lfluo:0']
        elif db == 'DB3':
            features = ['tilt', 'parm2', 'parm1', 'h2alt', 'h1alt',\
                        'hgpf', 'vis', 'h2ostr', 'rho:0', 'rho:1', 'lfluo:0']
        elif db == 'DB4':
            features = ['tilt', 'parm2', 'parm1', 'h2alt', 'h1alt',\
                        'hgpf', 'vis', 'h2ostr', 'cw_shift', 'fwhm_shift',\
                        'rho:0', 'rho:1', 'lfluo:0']

    # Load database
    od = load_db(sensor, db, db_extra, features)
    #print(od['feat'])

    # Renormalise H2O
    if sensor == 'DESIS':
        if db_extra == 'ATM_MIDLAT_SUMMER':
            h2o_conv = 3635.9*0.000804
        elif db_extra == 'ATM_TROPICAL':
            h2o_conv = 5119.4*0.000804
    elif sensor == 'HyPlant':
        h2o_conv = 3635.9*0.000804
    if db in ['DB2', 'DB3', 'DB4']:
        ih2ostr = np.where(np.array(features) == 'h2ostr')[0][0]
        od['feat'][:,ih2ostr] *= h2o_conv

    # Renormalise AOT -> Not needed because baseline point is -AOT already
    #if db in ['DB2', 'DB3', 'DB4']:
    #    ivis = np.where(np.array(features) == 'vis')[0][0]
    #    od['feat'][:,ivis] *= -1.

    # Loop over baseline points
    lsen = list()
    for p in ip:
        #print(p)

        # Search for point in feature space
        mask = np.ones(od['feat'].shape[0], dtype='bool')
        for dimf in range(len(p)):
            mask = mask * (od['feat'][:,dimf] == p[dimf])
        i = np.where(mask)[0][0]
        print('point: ', p, od['feat'][i,:])

        # Extract target
        lsen.append(od['targ'][i,:])

    lsen = np.array(lsen)

    return lsen


# Adapted from WP2 concept_study.py
def load_db(sensor, db, db_extra, features):
    """Load DB from disk.

    :param sensor: sensor (DESIS, HyPlant)
    :param db: database (DB1, DB2, DB3, DB4)
    :param db_extra: sub-database (e.g. ATM_MIDLAT_SUMMER/TROPICAL for DESIS)
    :param features: features to be collected

    :returns: dictionary with database
    """

    # Imports here so that not required to run other functions
    import h5py

    # Output dictionary
    output_dict = dict()
    output_dict['sensor'] = sensor
    output_dict['db'] = db
    output_dict['db_extra'] = db_extra

    # Identify database hdf5 file basename
    if sensor == 'DESIS':
        d = '/data/local/db_dump/DESIS/20220510/'
        bse = sensor + '_SENSOR_' + db + '_'
        bse_out = bse + db_extra
        db_extra_name = 'model'
    elif sensor == 'HyPlant':
        d = '/data/local/db_dump/HYPLANT/'
        bse = sensor + '_SENSOR_' + db + '_' + db_extra + '_'
        bse_out = bse[:-1]
        db_extra_name = 'h1alt'

    print('')

    # Loop over sampling type (FIXED, HALTON, RANDOM)
    ft_mat_list, tg_mat_list = list(), list()
    #for st in ['FIXED', 'HALTON', 'RANDOM']:
    for st in ['FIXED']:
        name = bse + st
        f = d + name + '.hdf5'
        print(st, f)
        output_dict[st + '_file'] = f

        # Open database file
        fh5 = h5py.File(f, 'r')
        print('keys: ', fh5[name].keys())

        # Filter per DB extra parameter (model for DESIS, h1alt for HyPlant)
        if sensor == 'DESIS':
            vec = np.array(fh5[name][db_extra_name]).astype(str)
            inds = np.where(vec==db_extra)[0]
        elif sensor == 'HyPlant':
            #vec = np.array(fh5[name][db_extra_name])
            #inds = np.where(vec==int(db_extra)/1000.)[0]
            # Note that db_extra indicates the altitude above the ground, i.e.
            # h1alt-h2alt. Note as well that h1alt and h2alt are varied in
            # DB3,4.
            h1vec = np.array(fh5[name]['h1alt'])
            h2vec = np.array(fh5[name]['h2alt'])
            vec = h1vec - h2vec
            if db_extra == '675': # 659-691 m
                inds = np.where((vec>=0.659)*(vec<=0.691))[0]
            if db_extra == '1559': # 1543-1598 m
                inds = np.where((vec>=1.543)*(vec<=1.598))[0]
        print(vec, inds, fh5[name][db_extra_name])
        print('selected ', len(inds), ' out of ', len(vec))

        # Extract features
        ft_mat = np.zeros((len(inds),len(features)), dtype='float32')
        ft_names = list()
        for ift, ft in enumerate(features):
            #print(ift, ft)

            # Features inside vectors
            if ':' in ft:
                ftn, fti = ft.split(':')
                ft_vec = np.array(fh5[name][ftn])[inds][:,int(fti)]
                ft_names.append(ft)
                #print(ft, ftn, fti, ft_vec)

            # Simple features
            else:
                ft_vec = np.array(fh5[name][ft])[inds]
                ft_names.append(ft)
                #print(ft, ft_vec)

            # Save in feature matrix
            #print('saving')
            ft_mat[:, ift] = ft_vec
        ft_names = np.array(ft_names)
        ft_mat_list.append(ft_mat)
        print('feature matrix:', ft_mat.shape, ft_mat.dtype)
        print('feature names:', ft_names, ft_names.shape, ft_names.dtype)
        output_dict['feat_names'] = ft_names

        # Extract target (at-sensor radiance)
        tg_mat = np.array(fh5[name]['lsensor'])[inds,:]
        tg_mat_list.append(tg_mat)
        print(fh5[name]['lsensor'][:].shape)
        print('target matrix:', tg_mat.shape, tg_mat.dtype)

        # Extract sensor id
        sid = np.array(fh5[name]['sensor_id'])[inds]
        print('sensor id:', sid.shape, sid.dtype)

        # Check zero-valued targets
        zero_targ = tuple(np.zeros(tg_mat.shape[1]))
        zt = np.where((tg_mat==zero_targ).all(axis=1))[0]
        print('  number of zero targets: ', len(zt), zt)
        print('  sensor ids: ', sid[zt])
        for zti in zt[:10]:
            print(ft_mat[zti,:], tg_mat[zti,:], sid[zti])

        # Extract central wavelengths
        cw = fh5[name]['srf_cw'][0,:]
        #print('cw: ', cw)

        # Close database file
        fh5.close()

        # Save memory
        del ft_mat, tg_mat, sid


    # Merge sampling types
    output_dict['feat'] = np.concatenate(ft_mat_list, axis=0)
    output_dict['targ'] = np.concatenate(tg_mat_list, axis=0)
    print('feature matrix:', output_dict['feat'].shape,\
                             output_dict['feat'].dtype)
    print('target matrix:', output_dict['targ'].shape,\
                            output_dict['targ'].dtype)
    output_dict['cw'] = cw

    return output_dict



#######################################
# Estimate LF uncertainty
#######################################


def estimate_LF_uncertainty(od):
    """Estimate LF uncertainty.

    :param od: output directory
    """

    # DESIS wavelengths
    cw_d = np.array([745, 747.5, 750, 752.5, 755, 757.5, 760, 762.5, 765, \
                     767.5, 770, 772.5, 775])

    # HyPlant wavelengths
    cw_h = np.array([740.76, 740.87, 740.98, 741.09, 741.20, 741.31, 741.42, 741.53, 741.64, 741.75, 741.86, 741.97, 742.08, 742.19, 742.30, 742.42, 742.53, 742.64, 742.75, 742.86, 742.97, 743.08, 743.19, 743.30, 743.41, 743.52, 743.63, 743.74, 743.85, 743.96, 744.07, 744.18, 744.29, 744.40, 744.51, 744.62, 744.73, 744.84, 744.95, 745.06, 745.17, 745.28, 745.39, 745.50, 745.61, 745.73, 745.84, 745.95, 746.06, 746.17, 746.28, 746.39, 746.50, 746.61, 746.72, 746.83, 746.94, 747.05, 747.16, 747.27, 747.38, 747.49, 747.60, 747.71, 747.82, 747.93, 748.04, 748.15, 748.26, 748.37, 748.48, 748.60, 748.71, 748.82, 748.93, 749.04, 749.15, 749.26, 749.37, 749.48, 749.59, 749.70, 749.81, 749.92, 750.03, 750.14, 750.25, 750.36, 750.47, 750.58, 750.69, 750.80, 750.91, 751.03, 751.14, 751.25, 751.36, 751.47, 751.58, 751.69, 751.80, 751.91, 752.02, 752.13, 752.24, 752.35, 752.46, 752.57, 752.68, 752.79, 752.90, 753.01, 753.12, 753.23, 753.35, 753.46, 753.57, 753.68, 753.79, 753.90, 754.01, 754.12, 754.23, 754.34, 754.45, 754.56, 754.67, 754.78, 754.89, 755.00, 755.11, 755.22, 755.33, 755.45, 755.56, 755.67, 755.78, 755.89, 756.00, 756.11, 756.22, 756.33, 756.44, 756.55, 756.66, 756.77, 756.88, 756.99, 757.10, 757.21, 757.33, 757.44, 757.55, 757.66, 757.77, 757.88, 757.99, 758.10, 758.21, 758.32, 758.43, 758.54, 758.65, 758.76, 758.87, 758.98, 759.09, 759.21, 759.32, 759.43, 759.54, 759.65, 759.76, 759.87, 759.98, 760.09, 760.20, 760.31, 760.42, 760.53, 760.64, 760.75, 760.87, 760.98, 761.09, 761.20, 761.31, 761.42, 761.53, 761.64, 761.75, 761.86, 761.97, 762.08, 762.19, 762.30, 762.41, 762.53, 762.64, 762.75, 762.86, 762.97, 763.08, 763.19, 763.30, 763.41, 763.52, 763.63, 763.74, 763.85, 763.96, 764.08, 764.19, 764.30, 764.41, 764.52, 764.63, 764.74, 764.85, 764.96, 765.07, 765.18, 765.29, 765.40, 765.51, 765.63, 765.74, 765.85, 765.96, 766.07, 766.18, 766.29, 766.40, 766.51, 766.62, 766.73, 766.84, 766.95, 767.07, 767.18, 767.29, 767.40, 767.51, 767.62, 767.73, 767.84, 767.95, 768.06, 768.17, 768.28, 768.39, 768.51, 768.62, 768.73, 768.84, 768.95, 769.06, 769.17, 769.28, 769.39, 769.50, 769.61, 769.72, 769.83, 769.95, 770.06, 770.17, 770.28, 770.39, 770.50, 770.61, 770.72, 770.83, 770.94, 771.05, 771.16, 771.28, 771.39, 771.50, 771.61, 771.72, 771.83, 771.94, 772.05, 772.16, 772.27, 772.38, 772.50, 772.61, 772.72, 772.83, 772.94, 773.05, 773.16, 773.27, 773.38, 773.49, 773.60, 773.71, 773.83, 773.94, 774.05, 774.16, 774.27, 774.38, 774.49, 774.60, 774.71, 774.82, 774.93, 775.05, 775.16, 775.27, 775.38, 775.49, 775.60, 775.71, 775.82, 775.93, 776.04, 776.15, 776.27, 776.38, 776.49, 776.60, 776.71, 776.82, 776.93, 777.04, 777.15, 777.26, 777.38, 777.49, 777.60, 777.71, 777.82, 777.93, 778.04, 778.15, 778.26, 778.37, 778.48, 778.60, 778.71, 778.82, 778.93, 779.04, 779.15, 779.26])

    # Imports here so that not required to run other functions
    import glob
    import matplotlib.pyplot as plt
    import matplotlib

    # Folder with fast forward simulator
    ddffs = '/data/local/forward_learning/learned_models/'

    # Select sensor and database
    sd_list = [['DESIS', 'DB4', 'ATM_MIDLAT_SUMMER', 'mid-lat summer'],\
               ['HyPlant', 'DB4', '675', '675 m agl']]

    # Loop over sensors/databases
    for sd in sd_list:
        sensor, db, db_extra, db_extra_name = sd
        s = sensor + '_SENSOR_' + db + '_' + db_extra
        print(s)

        if sensor == 'DESIS': cw_s = cw_d
        else: cw_s = cw_h

        # Baseline input point and input names
        ident = sensor + '_' + db
        if sensor == 'HyPlant' and db in ['DB3', 'DB4']:
            ident = ident + '_' + db_extra 
        ip = db_baseline[ident]
        print('Baseline point: ', ip.shape, ip)

        # Names of inputs
        ip_names = list()
        for ijk in range(len(db_input[ident])):
            nm = db_input[ident][ijk][0].split('[')[0][:-1]
            ip_names.append(nm)
        print('input names: ', ip_names)

        # Create list of points with several LFs in [0,8] mW/m2/sr/nm
        ilf = np.where(np.array(ip_names) == 'LF(737 nm)')[0][0]
        lfs = np.arange(1., 8.1, 1.0) #np.array([1., 4., 8.])
        ips = list()
        for lf in lfs:
            aux = np.array(ip[0,:])
            aux[ilf] = lf
            ips.append(aux)
        ips = np.array(ips)
        print('points: ', ips.shape, ips)

        # Evaluate fast forward simulator (inlc. derivative)
        f = glob.glob(ddffs + '/' + s + '*_model.pkl')[0]
        mod = os.path.basename(f).split('_')[-3]
        ffs = load(f)
        lsen_ffs = evaluate(ffs, ips)
        #print(lsen_ffs.shape, lsen_ffs)
        lsen_deriv = evaluate_deriv(ffs, ips) # derivative
        #print(lsen_deriv.shape, lsen_deriv)


        # LF uncertainty: delta_LF = delta_Lsen * |dLsen/dLF|^-1
        dLsendLF = np.abs(lsen_deriv[:,:,-1]) # [#samples]x[#bands]
        # The derivative is wrt the normalised LF, so it has units of Lsen, i.e.
        # mW/cm2/sr/microm. We re-normalise the input LF to [0,8] mW/m2/sr/nm =
        # [0,0.8] mW/cm2/sr/microm so that the the derivative is unitless.
        dLsendLF /= 0.8 # unitless: mW/cm2/sr/microm/(mW/cm2/sr/microm)=1

        # Plot delta_LF/delta_Lsen(lambda)=|dLsen/dLF|^-1
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.xlabel('wavelength [nm]')
        plt.ylabel(r'$\Delta L_F / \Delta L_{\mathrm{sen}}$')
        x0, x1, y0, y1 = 740., 780., 0, 10.
        plt.xlim(x0, x1)
        plt.ylim(y0, y1)
        #plt.yscale('log')
        ts = sensor + ' ' + db + ' (' + db_extra_name + ')'
        plt.text(x0+(x1-x0)*0.01, y1-(y1-y0)*0.025, ts, fontsize=10)
        ts = mod
        plt.text(x0+(x1-x0)*0.01, y1-2*(y1-y0)*0.025, ts, fontsize=10)
        lfs_plot = [1., 4., 8.]
        cols = ['r', 'g', 'b']
        for ii, lf in enumerate(lfs_plot):
            ilf = np.where(lfs==lf)[0][0]
            laux = r'$L_F=' + str(np.around(lf,1)) + r'$ mW/m$^2$/sr/nm'
            plt.plot(cw_s, 1./dLsendLF[ilf,:], cols[ii], label=laux)
        plt.legend(bbox_to_anchor=(0.99, 0.01), loc='lower right',\
                   borderaxespad=0., fontsize=8)
        fig.savefig(od + '/' + s + '_LFunc1.png', bbox_inches='tight')


        # Plot delta_LF/LF(LF)=(delta_Lsen/Lsen)*Lsen*|dLsen/dLF|^-1
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.xlabel(r'$L_F$ [mW/m$^2$/sr/nm]')
        plt.ylabel(r'$\Delta L_F / L_F$')
        x0, x1, y0, y1 = 0., 8., 0, 1.0
        plt.xlim(x0, x1)
        plt.ylim(y0, y1)
        #plt.yscale('log')
        ts = sensor + ' ' + db + ' (' + db_extra_name + ')'
        plt.text(x0+(x1-x0)*0.01, y1-(y1-y0)*0.025, ts, fontsize=10)
        ts = mod
        plt.text(x0+(x1-x0)*0.01, y1-2*(y1-y0)*0.025, ts, fontsize=10)
        dlsen_plot = [0.01, 0.05]
        w_plot = [755, 760, 762]
        cols = ['r', 'g', 'b', 'r--', 'g--', 'b--']
        ii = 0
        for dlsen in dlsen_plot:
            for w in w_plot:
                iw2 = np.argmin(np.abs(cw_s-w)) #np.where(np.abs(cw_s-w)<tol)[0][0]
                print('w: ', iw2, cw_s[iw2])
                laux = r'$\Delta L_{\mathrm{sen}}/L_{\mathrm{sen}}=' +\
                       str(np.around(dlsen,2)) + r'$, ' +\
                       r'$\lambda=' + str(np.around(cw_s[iw2],2)) + r'$ nm, '
                # delta_LF/LF(LF)=(delta_Lsen/Lsen)*Lsen*|dLsen/dLF|^-1/LF
                dLFLF = dlsen * lsen_ffs[:,iw2] / dLsendLF[:,iw2] / lfs
                plt.plot(lfs, dLFLF, cols[ii], label=laux)
                ii += 1
        plt.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right',\
                   borderaxespad=0., fontsize=8)
        fig.savefig(od + '/' + s + '_LFunc2.png', bbox_inches='tight')


        plt.show()
        plt.close()



#######################################
# Main script
#######################################

if __name__ == '__main__':

    # Check script usage
    if len(sys.argv) < 2:
        print('Usage: python simulation_interface.py [output_dir]')
        sys.exit(1)

    # Results directory
    out_dir = os.path.abspath(sys.argv[1])
    os.mkdir(out_dir)
    print(' output directory: ' + out_dir)
    print('')


    ## Test case: evaluate model
    #dd = '/data/local/forward_learning/learned_models/'
    #f = dd + '/HyPlant_SENSOR_DB4_1559_PRR:4:0_FHR_model.pkl'
    #ffs = load(f)
    #ip = np.array([[0.0, 25., 90., 0., 1.559, # TA, SZA, RAA, h_gnd, h_sen
    #                0.5, -0.1, 2.0,           # g, -AOT 550 nm, H2O
    #                0.0, 0.0,                 # CW shift, FWHM shift
    #                0.3,0.0,4.0]])            # rho(740 nm), drho/dl, LF(737 nm)
    #lsen = evaluate(ffs, ip)
    #print(lsen.shape)

    ## Test case: evaluate derivative (slow and fast)
    #dd = '/data/local/forward_learning/learned_models/'
    #f = dd + '/HyPlant_SENSOR_DB1_1559_PRR:2:0_FHR_model.pkl'
    #ffs = load(f)
    #ffs_deriv = load_deriv_slow(ffs)
    #ip = np.array([[0.3,0.0,4.0]])            # rho(740 nm), drho/dl, LF(737 nm)
    #lsenp = evaluate_deriv_slow(ffs, ffs_deriv, ip)
    #print(lsenp)
    #lsenp2 = evaluate_deriv(ffs, ip)
    #print(lsenp2)
    #np.testing.assert_array_almost_equal(lsenp, lsenp2, decimal=6)

    ## Test case: evaluate derivative (fast only)
    #dd = '/data/local/forward_learning/learned_models/'
    #f = dd + '/HyPlant_SENSOR_DB4_1559_PRR:4:0_FHR_model.pkl'
    #ffs = load(f)
    #ip = np.array([[0.0, 25., 90., 0., 1.559, # TA, SZA, RAA, h_gnd, h_sen
    #                0.5, -0.1, 2.0,           # g, -AOT 550 nm, H2O
    #                0.0, 0.0,                 # CW shift, FWHM shift
    #                0.3,0.0,4.0]])            # rho(740 nm), drho/dl, LF(737 nm)
    #lsenp = evaluate_deriv(ffs, ip)
    #print(lsenp.shape)

    # Test case: evaluate model and derivative for new DESIS DB5
    dd = '/data/local/forward_learning/learned_models/'
    f = dd + '/DESIS_SENSOR_DB5_ATM_MIDLAT_SUMMER_PRR:4:0_H_model.pkl'
    ffs = load(f)
    ip = np.array([[0.0, 22.5, 90, 0.0,   # TA, SZA, RAA, h_gnd
                    -0.175, 2.65,         # -AOT 550 nm, H2O
                    0.0, 0.0,             # CW shift, FWHM shift
                    0.3, 0.0, 1.0, 4.0]]) # rho(740 nm), drho/dl, e, LF(737 nm)
    lsen = evaluate(ffs, ip)
    print(lsen.shape)
    lsenp = evaluate_deriv(ffs, ip)
    print(lsenp.shape)

    # Test case: evaluate model and derivative for new HyPlant DB5
    dd = '/data/local/forward_learning/learned_models/'
    f = dd + '/HyPlant_SENSOR_DB5_675_PRR:4:0_H_model.pkl'
    ffs = load(f)
    ip = np.array([[0.0, 25., 90., 0., 0.675, # TA, SZA, RAA, h_gnd, h_sen
                    -0.1, 2.0,                # -AOT 550 nm, H2O
                    0.0, 0.0,                 # CW shift, FWHM shift
                    0.3,0.0, 1.0, 4.0]])            # rho(740 nm), drho/dl, LF(737 nm)
    lsen = evaluate(ffs, ip)
    print(lsen.shape)
    lsenp = evaluate_deriv(ffs, ip)
    print(lsenp.shape)


    # Cross-check with database
    #cross_check(out_dir)

    # Estimate LF uncertainty
    #estimate_LF_uncertainty(out_dir)

