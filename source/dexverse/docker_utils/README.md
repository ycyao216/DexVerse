# Mount DexVerse into the Isaac Lab docker container

`docker-compose.dexverse.patch.yaml` bind-mounts this repository and a host `datastorage/`
directory into the Isaac Lab container. Pass it to Isaac Lab's `docker/container.py` via
`--files`:

```bash
cd <workspace>/IsaacLab
./docker/container.py start \
    --files ../../DexVerse/source/dexverse/docker_utils/docker-compose.dexverse.patch.yaml
```

For CloudXR teleoperation, combine it with Isaac Lab's CloudXR patch (see the main README's
"Teleoperation and Data Collection" section):

```bash
./docker/container.py start \
    --files docker-compose.cloudxr-runtime.patch.yaml \
            ../../DexVerse/source/dexverse/docker_utils/docker-compose.dexverse.patch.yaml \
    --env-files .env.cloudxr-runtime
```

## What gets mounted

| Host path (under DexVerse repo) | Container path |
| --- | --- |
| `source/dexverse/` | `/workspace/dexverse/source/dexverse` |
| `scripts/` | `/workspace/dexverse/scripts` |
| `datastorage/` | `/workspace/dexverse/datastorage` |

The patch sets `DEXVERSE_DATA_DIR=/workspace/dexverse/datastorage`. When this variable is
present, `scripts/record_demos.py` redirects all output to the host-mounted `datastorage/`
folder.

Create the host directory once if needed:

```bash
mkdir -p /path/to/DexVerse/datastorage
```

Run DexVerse scripts from inside the container:

```bash
cd /workspace/dexverse
python -m pip install -e source/dexverse   # once per container
./isaaclab.sh -p scripts/record_demos.py \
    --task Dexverse-PickUpStick-v0 \
    --dataset_dir grasping \
    --teleop_device handtracking \
    --enable_pinocchio
```

## Notes

- Relative paths in `--files` and bind sources are resolved against `IsaacLab/docker`. The
  defaults assume DexVerse is cloned next to Isaac Lab. If it lives elsewhere, export
  `DEXVERSE_PATH=/abs/path/to/DexVerse` before running `container.py`.
