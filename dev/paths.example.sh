# Placeholders used by the scripts in dev/. Copy to paths.local.sh (git-ignored), fill in, then
#   source dev/paths.local.sh
# Python scripts resolve ${VAR} with os.path.expandvars; shell scripts use them directly.

export EM_PIPE_ROOT=/path/to/em-pipe            # a clone of this repository
export EM_PIPE_ENV=/path/to/conda-envs/em-pipe   # the conda env (its bin/python)
export CLUSTER_LOGIN=login.example.org           # cluster login host (scripts that ran over ssh)

export DEV_OUT=/path/to/dev_out                  # where redirected outputs go (never a dataset folder)
export DEV_WORK=/path/to/dev_work                # scratch: intermediate files, copied tables

# Dataset output folders (output_dir of each dataset's config) and raw data folders
export DEV_35H=/path/to/35h_stitched
export DEV_37I=/path/to/37i_stitched
export DEV_RAW_35H=/path/to/raw/35h
export DEV_RAW_37I=/path/to/raw/37i
export DEV_DATA_ROOT=/path/to/electron_microscopy   # parent of the dataset folders

# Volume names start with the dataset's project prefix, e.g. "${DEV_PREFIX}37i_8nm.ome.zarr"
export DEV_PREFIX=PROJ_SAMPLE_
export DEV_PROJECT=PROJ

# Areas of the group's storage that test runs used
export DEV_USER_AREA=/path/to/your/area           # personal test outputs (two-day test, norm tests)
export DEV_SHARED_AREA=/path/to/shared/area
export DEV_LAB_AREA=/path/to/group/area
export DEV_CLUSTER_PATH=/path/on/cluster
