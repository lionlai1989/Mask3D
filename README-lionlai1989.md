# Mask3D

## Required Hardware and CUDA Version

Training has been successfully run on a GCP virtual machine (`a2-highgpu-1g`)
equipped with an `A100 40GB` GPU.

Inferencing has been successfully run on a GCP virtual machine (`g2-standard-8`)
equipped with an `NVIDIA L4` GPU.

## Running Inference in Docker Container

Currently, we only support the Docker container for inferencing, not training.

### Building the Docker Image

Use the Dockerfile `./deployment/Dockerfile` to build the image.

```
export DOCKER_BUILDKIT=1 && docker build -t mask3d:main -f "deployment/Dockerfile" .
```

**References:**
- https://stackoverflow.com/a/66301568/2641038
- https://pythonspeed.com/articles/docker-build-secrets/
- https://stackoverflow.com/questions/57187324/trying-to-pip-install-a-private-repo-in-a-dockerfile
- https://stackoverflow.com/questions/55929417/how-to-securely-git-clone-pip-install-a-private-repository-into-my-docker-image/59455653#59455653


### Running the Docker Image as a Container

There are two steps to perform inference: starting the container and executing
the Python script inside the container.

- 1. Start the container:

  ```
  docker run -d --rm \
    --gpus all \
    --runtime nvidia \
    --user 1004:1005 \
    mask3d:main \
    bash -c 'while true; do sleep 30; done'
  ```

- 2. Executing the inference script in the container:

  ```
  docker exec -it \
    -u 1004:1005 \
    <CONTAINER ID> \
    python inference.py \
      --input-file /Mask3D/data/<input_file_name> \
      --output-dir /Mask3D/output/<output_directory_name> \
      --checkpoint /Mask3D/<model_path> \
      --model-type s3dis \
      --confidence-threshold 0.8 \
      --visualize True
  ```

If everything goes well, you should see your folder `--output-dir` in `output/`.

## Running Inference Locally

If you don't want to run inference in a Docker container, you can follow this
section to run inference locally without using a Docker container. 

### Installing Prerequisites

This section describes how to install Mask3D on the virtual machine.

<!-- - Comment out the Python packages `pycocotools==2.0.4` and `pyyaml==5.4.1` in
  `environment.yml`. this may not be necessary. -->

- Use `conda` to create and install a new Conda virtual environment:

  ```shell
  conda env create -f environment.yml
  ```

- List all Conda virtual environments and activate `mask3d_cuda113`:

  ```shell
  conda env list
  conda activate mask3d_cuda113
  ```

<!-- - Manually install `pycocotools==2.0.4` and `pyyaml==5.4.1`: this may not be necessary.

  ```shell
  pip install "cython<3.0.0" "numpy==1.24.2" \
  && pip install --no-build-isolation "pycocotools==2.0.4" \
  && pip install --no-build-isolation "pyyaml==5.4.1"
  ``` -->

- Install everything else in `environment.yml`:

  ```shell
  conda env update --file environment.yml
  ```

- If you encounter any errors related to the compiler GCC version, you can
  install a specific GCC version by following this section. For instance, you
  can install `gcc-9` and `g++-9` on your system and set it as default: 

  ```
  sudo apt install -y gcc-9 g++-9
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 \
  --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
  --slave /usr/bin/gcov gcov /usr/bin/gcov-9
  ```

- Follow `README.md` to install `pytorch` related software.

- Follow `README.md` to install `MinkowskiEngine`. Please install
  `libopenblas-dev` beforehand to [avoid the compilation error](https://github.com/JonasSchult/Mask3D/issues/115).
  
  ```shell
  sudo apt install libopenblas-dev
  ```

- Continue following the instructions in `README.md` to install `ScanNet` and `pointnet2`.

- When installing `pytorch-lightning`, we fixate the version
  `pytorch-lightning==1.7.2`. But it doesn't fixate the version of the package
  `torchmetrics`, which is required by `pytorch-lightning`. Thus, we need to
  reinstall `torchmetrics==0.11.4`.

  ```shell
  pip install torchmetrics==0.11.4
  ```


### Inferencing on USDZ Files

After installation, we can run inferences on USDZ files by using the
script `inference.py`. Here are a few example commands:

```shell
# Use model trained on S3DIS from scratch
python inference.py \
--input-file /home/lionlai/HomeeAI/Mask3D/data/phone_booth_7f.usdz \
--output-dir /home/lionlai/HomeeAI/Mask3D/output/phone_booth_7f_s3dis_from_scratch \
--checkpoint /home/lionlai/HomeeAI/Mask3D/checkpoints/s3dis/from_scratch/area1_from_scratch.ckpt \
--model-type s3dis \
--confidence-threshold 0.8 \
--visualize True

# Use model pretrained on ScanNetV2 and finetuned on S3DIS
python inference.py \
--input-file /home/lionlai/HomeeAI/03_HomeeAI_Mask3D/data/phone_booth_7f.usdz \
--output-dir /home/lionlai/HomeeAI/03_HomeeAI_Mask3D/output/phone_booth_7f_s3dis_finetune_before_PR \
--checkpoint /home/lionlai/HomeeAI/03_HomeeAI_Mask3D/checkpoints/s3dis/scannet_pretrained/area1_scannet_pretrained.ckpt \
--model-type s3dis \
--confidence-threshold 0.8 \
--visualize True

# Use model trained on ScanNet200
python inference.py \
--input-file /home/lionlai/HomeeAI/Mask3D/data/phone_booth_7f.usdz \
--output-dir /home/lionlai/HomeeAI/Mask3D/output/phone_booth_7f_scannet200 \
--checkpoint /home/lionlai/HomeeAI/Mask3D/checkpoints/scannet200/scannet200_benchmark.ckpt \
--model-type scannet200 \
--confidence-threshold 0.8 \
--visualize True
```

The output folder contains two subfolders, `structural` and `non_structural`.

- `structural`: Contains structural instances such as walls, floors, ceilings,
  doors, etc.
- `non_structural`: Contains all non-structural instances such as chairs,
  tables, and clutter. Clutter includes all kind of things that don't belong to
  a well-defined class.

Note that if the `--model-type` is `scannet200`, it won't create instances for
`floor` and `wall`. The `scannet200` models from the officiai Mask3D repository
does not incorporate the classes, "wall" and "floor" because [the official
benchmark of ScanNet200 doesn't treat `floor` and `wall` as
instances](https://github.com/JonasSchult/Mask3D/issues/42). We will have to
redesign the target classes and retrain models if we want to include `wall` and `floor`.

### Running Mask3D's Inference Script

Remember to config the following setting:

```
data.batch_size=1
data.voxel_size=0.02 or 0.04
general.checkpoint="checkpoints/s3dis/scannet_pretrained/area${CURR_AREA}_scannet_pretrained.ckpt" \
```

## Training

Not supported.

### Preprocessing Dataset

#### S3DIS Preprocessing

For preprocessing S3DIS, follow the instructions in this [issue](https://github.com/JonasSchult/Mask3D/issues/8).

You may encounter the following error. (I haven't figured out what this error means.)

```
2024-05-21 09:48:17.809 | INFO     | datasets.preprocessing.base_preprocessing:preprocess:47 - Tasks for Area_5: 68
[Parallel(n_jobs=8)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=8)]: Done   2 tasks      | elapsed:    2.9s
[Parallel(n_jobs=8)]: Done   9 tasks      | elapsed:    9.0s
FILE SIZE DOES NOT MATCH FOR input_data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/Area_5/lobby_1/lobby_1.txt
(1162766 vs. 1223236)
```

#### Prepare ScanNet data

```shell
python -m datasets.preprocessing.scannet_preprocessing preprocess \
--data_dir="data/raw/ScanNet" \
--save_dir="data/processed/scannet" \
--git_repo="/home/lionlai/HomeeAI/Mask3D/ScanNet" \
--scannet200=false

python -m datasets.preprocessing.scannet_preprocessing preprocess \
--data_dir="/mnt/homeeai/lionlai/3d-segmentation-dataset/ScanNet/raw" \
--save_dir="/mnt/homeeai/lionlai/3d-segmentation-dataset/ScanNet/processed" \
--git_repo="/home/lionlai/HomeeAI/Mask3D/ScanNet" \
--scannet200=false 
```

### Prerequisites Before Training

- Disable `wandb`:

```
export WANDB_MODE=offline && export WANDB_MODE=disabled
```

Refer to this [issue](https://github.com/JonasSchult/Mask3D/issues/160)

- Reduce voxel size:

```
data.voxel_size=0.05
```

Refer to this [issue](https://github.com/JonasSchult/Mask3D/issues/59)

it should be done in the beginning or enlarge the gpu size.
add


- Reduce batch size:

Refer to this [issue](https://github.com/JonasSchult/Mask3D/issues/98)

## Miscellaneous Notes

- Removing a Conda environment.
  ```shell
  conda remove --name mask3d_cuda113 --all
  ```

Use docker:
```
docker run -it --rm xhm126/mask3d:v1.0 /bin/bash

docker run -ti --rm \
    --mount type=bind,source="$(pwd)/input_data",target=/workspace/project/Mask3D/input_data \
    --mount type=bind,source="$(pwd)/data",target=/workspace/project/Mask3D/data \
    xhm126/mask3d:v1.0 /bin/bash

docker run -ti --rm \
    --mount type=bind,source="$(pwd)",target=/workspace/project/Mask3D/ \
    xhm126/mask3d:v1.0 /bin/bash

python -m datasets.preprocessing.s3dis_preprocessing preprocess \
--data_dir="input_data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/" \
--save_dir="data/processed/s3dis"
```



- run on `cuda113` machine
    - preprocess "Aligned_Version"
```
python -m datasets.preprocessing.s3dis_preprocessing preprocess \
--data_dir="input_data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/" \
--save_dir="data/processed/s3dis"

python -m datasets.preprocessing.s3dis_preprocessing preprocess \
--data_dir="input_data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/" \
--save_dir="data/processed/s3dis"
```

   - preprocess non "Aligned_Version"
```
python -m datasets.preprocessing.s3dis_preprocessing preprocess \
--data_dir="input_data/s3dis/Stanford3dDataset_v1.2/" \
--save_dir="data/processed/s3dis"
```

- finetune on s3dis data with the pretrained model (Scannet)
```
./scripts/s3dis/s3dis_pretrained.sh
```

- download `scannet_pretrained.ckpt` and put into
  `checkpoints/s3dis/scannet_pretrained/`.
```
wget https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/scannet_pretrained/scannet_pretrained.ckpt
```
