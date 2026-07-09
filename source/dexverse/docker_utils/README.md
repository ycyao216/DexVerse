# Mount DexVerse into the Isaac Lab docker container

`docker-compose.dexverse.patch.yaml` is a Docker Compose patch that bind-mounts this repository
into the Isaac Lab container at `/workspace/dexverse`. Instead of copying or editing Isaac Lab's
own `docker-compose.yaml`, pass the patch to Isaac Lab's `docker/container.py` via `--files`:

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

Notes:

- Relative paths — both the `--files` arguments and the bind sources inside the patch — are
  resolved against `IsaacLab/docker`, so the defaults assume DexVerse is cloned next to Isaac Lab.
  If DexVerse lives elsewhere, export `DEXVERSE_PATH=/abs/path/to/DexVerse` before running
  `container.py`.
- Inside the container, install DexVerse once per container:

  ```bash
  ./isaaclab.sh -p -m pip install -e /workspace/dexverse/source/dexverse
  ```
