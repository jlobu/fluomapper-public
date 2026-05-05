import os, sys

this_dir = os.path.dirname(os.path.realpath(__file__))
FLUOMAP_DIR = os.path.dirname(this_dir)
PARENT_FLUOMAP_DIR = os.path.dirname(FLUOMAP_DIR)
NN_RESOURCES = os.path.join(PARENT_FLUOMAP_DIR, 'nn_resources')

sys.path.insert(0, NN_RESOURCES)
sys.path.insert(0, FLUOMAP_DIR)
sys.path.insert(0, PARENT_FLUOMAP_DIR)
