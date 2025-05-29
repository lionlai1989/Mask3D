import argparse
import random
from pathlib import Path

import albumentations as A
import hydra
import MinkowskiEngine as ME
import numpy as np
import torch
import trimesh

from datasets.scannet200.scannet200_constants import (
    CLASS_LABELS_200,
    SCANNET_COLOR_MAP_200,
    VALID_CLASS_IDS_200,
    # Constants of ScanNetV2
    # CLASS_LABELS_20,
    # SCANNET_COLOR_MAP_20
)
from trainer.trainer import InstanceSegmentation
from utils.utils import (
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
    load_checkpoint_with_missing_or_exsessive_keys,
)

# From `benchmark/evaluate_semantic_instance.py`, line 510 and 525
CLASS_LABELS_S3DIS = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
]
VALID_CLASS_IDS_S3DIS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
# From `datasets/semseg.py` line 117
S3DIS_COLOR_MAP = [
    [0, 255, 0],  # ceiling
    [0, 0, 255],  # floor
    [0, 255, 255],  # wall
    [255, 255, 0],  # beam
    [255, 0, 255],  # column
    [100, 100, 255],  # window
    [200, 200, 100],  # door
    [170, 120, 200],  # table
    [255, 0, 0],  # chair
    [200, 100, 100],  # sofa
    [10, 200, 100],  # bookcase
    [200, 200, 200],  # board
    [50, 50, 50],  # clutter
]

# Selected from `CLASS_LABELS_200` and `CLASS_LABELS_S3DIS`.
STRUCTURAL_CLASSES = {
    "wall",
    "floor",
    "ceiling",
    "doorframe",
    "door",
    "beam",
    "column",
    "window",
}


def generate_random_rgb():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return np.array([r, g, b], dtype=int)


def colorize_semantic_pointcloud(labels_mapped, colormap="scannet200"):
    assert labels_mapped.ndim == 2
    num_points = labels_mapped.shape[0]
    colors = np.zeros((num_points, 3), dtype=np.uint8)

    unique_labels, counts = np.unique(labels_mapped, return_counts=True)
    for li in unique_labels:
        if colormap == "scannet200":
            # v_li = VALID_CLASS_IDS_200[int(li)]
            # print("v_li: ", v_li, CLASS_LABELS_200[v_li], CLASS_LABELS_200[li])
            # print("SCANNET_COLOR_MAP_200[v_li]: ", SCANNET_COLOR_MAP_200[v_li])
            colors[(labels_mapped == li)[:, 0], :] = SCANNET_COLOR_MAP_200[li]
        elif colormap == "s3dis":
            colors[(labels_mapped == li)[:, 0], :] = S3DIS_COLOR_MAP[li]
        else:
            raise ValueError("Unknown colormap - not supported")

    return colors


def colorize_instance_pointcloud(instance_binary_mask):
    assert instance_binary_mask.ndim == 2
    num_points = instance_binary_mask.shape[0]
    colors = np.zeros((num_points, 3), dtype=int)

    for i in range(instance_binary_mask.shape[1]):
        unique, counts = np.unique(instance_binary_mask[:, i], return_counts=True)
        # Each binary mask shall have only two values, 0 and 1.
        assert unique[0] == 0 and unique[1] == 1
        # Sum of counts shall equal the number of points.
        assert sum(counts) == num_points

        # colors[] = random_color
        colors[instance_binary_mask[:, i] == 1] = generate_random_rgb()

    return colors


def postprocess_output(logit, mask):
    p_labels = torch.softmax(logit, dim=-1)
    p_masks = torch.sigmoid(mask)

    la = torch.argmax(p_labels, dim=-1)
    c_label = torch.max(p_labels)
    m = p_masks > 0.5  # (M, ), binary mask
    c_m = p_masks[m].sum() / (m.sum() + 1e-8)  # scalar
    c = c_label * c_m
    return c, la, m


def map_output_to_pointcloud(
    num_vertex,
    outputs,
    inverse_map,
    label_space: str = "scannet200",
    confidence_threshold: float = 0.9,
):
    # Parse predictions
    logits = outputs["pred_logits"][0].detach().cpu()  # (num_queries, num_targets)
    masks = (
        outputs["pred_masks"][0].detach().cpu()
    )  # (num_points in sparse_tensor, num_queries)
    print("logits: ", logits.shape)
    print("masks: ", masks.shape)

    labels = []
    confidences = []
    masks_binary = []

    for i in range(len(logits)):  # 0 ~ num_queries-1
        confidence, label, m = postprocess_output(logits[i, :], masks[:, i])

        if label_space == "scannet200":
            if (
                label < 200
                and confidence > confidence_threshold
                and torch.any(m[inverse_map])
            ):
                labels.append(label.item())
                confidences.append(confidence.item())
                masks_binary.append(
                    m[inverse_map]
                )  # mapping the mask back to the original point cloud
        elif label_space == "s3dis":
            if (
                label < 13
                and confidence > confidence_threshold
                and torch.any(m[inverse_map])
            ):
                labels.append(label.item())
                confidences.append(confidence.item())
                masks_binary.append(
                    m[inverse_map]
                )  # mapping the mask back to the original point cloud

    labels_mapped = np.zeros((num_vertex, 1), dtype=int)

    instances_mapped = np.zeros((num_vertex, len(labels)), dtype=int)
    empty_room_mask = np.zeros((num_vertex, 1), dtype=bool)

    instance_names = []
    for i, (l, c, m) in enumerate(
        sorted(zip(labels, confidences, masks_binary), reverse=False)
    ):
        if label_space == "scannet200":
            label_offset = 2
            if l == 0:
                l = -1 + label_offset
            else:
                l = int(l) + label_offset
            labels_mapped[m == 1] = l
            # Create instance binary mask
            instances_mapped[m == 1, i] = 1
            instance_names.append(CLASS_LABELS_200[l])

        elif label_space == "s3dis":
            labels_mapped[m == 1] = l
            # Create instance binary mask
            instances_mapped[m == 1, i] = 1
            instance_names.append(CLASS_LABELS_S3DIS[l])

    # Create the instance of empty room
    if label_space == "scannet200":
        # Empty room composes of the following labels.
        # NOTE: "wall" contains "floor" and everything else not labelled because
        # labels_mapped is initialized as zeros.
        empty_room_idx = (
            (labels_mapped == CLASS_LABELS_200.index("wall"))
            | (labels_mapped == CLASS_LABELS_200.index("door"))
            | (labels_mapped == CLASS_LABELS_200.index("doorframe"))
            | (labels_mapped == CLASS_LABELS_200.index("ceiling"))
        )
    elif label_space == "s3dis":
        # Empty room composes of the following labels.
        empty_room_idx = (
            (labels_mapped == CLASS_LABELS_S3DIS.index("ceiling"))
            | (labels_mapped == CLASS_LABELS_S3DIS.index("floor"))
            | (labels_mapped == CLASS_LABELS_S3DIS.index("wall"))
            | (labels_mapped == CLASS_LABELS_S3DIS.index("beam"))
            | (labels_mapped == CLASS_LABELS_S3DIS.index("column"))
            | (labels_mapped == CLASS_LABELS_S3DIS.index("window"))
            | (labels_mapped == CLASS_LABELS_S3DIS.index("door"))
        )
    empty_room_mask[empty_room_idx.reshape(-1), 0] = 1

    return labels_mapped, instances_mapped, instance_names, empty_room_mask


def prepare_data(pointcloud_points, pointcloud_color, voxel_size=0.02, device="cuda"):
    """Prepare data as input of the Mask3D model. The unit of voxel_size is meter."""
    # Normalize point cloud features. From SemanticSegmentationDataset
    color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
    color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
    # Normalization should not be "min_max" because it's not used in training
    normalize_color = A.Normalize(mean=color_mean, std=color_std)

    points = pointcloud_points
    colors = pointcloud_color
    # colors = colors * 255.

    pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
    colors = np.squeeze(normalize_color(image=pseudo_image)["image"])

    coords = np.floor(points / voxel_size)
    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=coords,
        features=colors,
        return_index=True,
        return_inverse=True,
    )

    sample_coordinates = coords[unique_map]
    coordinates = [torch.from_numpy(sample_coordinates).int()]
    sample_features = colors[unique_map]
    features = [torch.from_numpy(sample_features).float()]

    coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features)
    features = torch.cat(features, dim=0)
    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features,
        device=device,
    )
    return data, points, colors, features, unique_map, inverse_map


class BaseDatasetConfig:
    def __init__(self, checkpoint):
        with hydra.experimental.initialize(config_path="conf"):
            # Compose a configuration
            self.cfg = hydra.experimental.compose(
                config_name="config_base_instance_segmentation.yaml"
            )

        self.cfg.general.checkpoint = str(checkpoint)
        self.cfg.general.train_mode = False
        self.cfg.general.eval_on_segments = True
        self.cfg.general.topk_per_image = 300
        self.cfg.general.use_dbscan = True
        self.cfg.general.dbscan_eps = 0.95
        self.cfg.general.export_threshold = 0.001
        self.cfg.data.test_mode = "test"
        self.cfg.model.num_queries = 150  # this is the dimension of the output!!!


# class Scannetv2Config(BaseDatasetConfig):
#     def __init__(self, checkpoint):
#         super().__init__(checkpoint)
#         # Specific configurations for ScanNetV2
#         self.cfg.general.num_targets = 21
#         self.cfg.data.num_labels = 20


class Scannet200Config(BaseDatasetConfig):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        # Specific configurations for ScanNet200
        self.cfg.general.num_targets = 201
        self.cfg.data.num_labels = 200


class S3disConfig(BaseDatasetConfig):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        # Specific configurations for S3DIS
        self.cfg.general.num_targets = 14
        self.cfg.data.num_labels = 13


def get_config(model_type, checkpoint):
    if model_type == "scannet200":
        return Scannet200Config(checkpoint)
    elif model_type == "s3dis":
        return S3disConfig(checkpoint)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main(
    input_file: Path,
    output_dir: Path,
    checkpoint: Path,
    model_type: str,
    up_axis: str,
    voxel_size: float,
    pointcloud_resampling_factor: float,
    confidence_threshold: float,
    visualize: bool,
):
    config = get_config(model_type, checkpoint)

    model = InstanceSegmentation(config.cfg)

    if config.cfg.general.backbone_checkpoint is not None:
        print("Load backbone checkpoint")
        config.cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            config.cfg, model
        )
    if config.cfg.general.checkpoint is not None:
        print("Load checkpoint")
        config.cfg, model = load_checkpoint_with_missing_or_exsessive_keys(
            config.cfg, model
        )

    model.eval()
    model.to("cuda")

    meshes, stage = read_usdz(input_file)
    assert len(meshes) == 1
    mesh = meshes[0]

    # Resample mesh
    num_resample_points = int(pointcloud_resampling_factor * mesh.vertices.shape[0])
    samples, face_index, colors = trimesh.sample.sample_surface(
        mesh, count=num_resample_points, sample_color=True
    )
    colors = colors.astype(np.uint8)

    # Rotate pointclouds
    if up_axis == "x":
        raise NotImplementedError("Please implement rotation matrix here.")
    elif up_axis == "y":
        # Rotate +90 degrees about X-axis
        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    elif up_axis == "z":
        # Identity matrix
        rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else:
        raise ValueError(f"Not recognized value for up_axis. Got {up_axis}")
    samples = np.matmul(samples, rotation_matrix)
    upsample_pc = trimesh.PointCloud(vertices=samples, colors=colors)
    # trimesh.Scene([upsample_pc]).show()

    data, points, colors, features, unique_map, inverse_map = prepare_data(
        upsample_pc.vertices, upsample_pc.colors[:, :3], voxel_size, "cuda"
    )

    # Run inference
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)

    labels, instances_mapped, instance_names, empty_room_mask = map_output_to_pointcloud(
        num_vertex=num_resample_points,
        outputs=outputs,
        inverse_map=inverse_map,
        confidence_threshold=confidence_threshold,
        label_space=model_type,
    )
    assert len(instance_names) == instances_mapped.shape[1]
    assert labels.shape[0] == instances_mapped.shape[0]
    assert labels.shape[1] == 1
    

    num_instances = instances_mapped.shape[1]

    instances_counter = {}
    for i in range(num_instances):
        instance_xyz = points[instances_mapped[:, i] == 1]

        # label_id = labels[instances_mapped[:, i] == 1]
        instance_label = instance_names[i]

        # Reversely rotate instance_xyz.
        instance_xyz = np.matmul(instance_xyz, rotation_matrix.T)

        instance_mesh = filter_mesh_by_proximity(
            mesh=mesh, points=instance_xyz, threshold_distance=0.02
        )

        # Store instances and their numbers of occurrences in dict
        if instance_label in instances_counter:
            instances_counter[instance_label] += 1
        else:
            instances_counter[instance_label] = 1
        instance_index = instances_counter[instance_label]

        # Write instances as USDZ
        if instance_label in STRUCTURAL_CLASSES:
            instance_label = instance_label.replace(" ", "_")
            output_file = (
                output_dir / f"structural/{instance_label}_{instance_index}.usdz"
            )

        else:
            instance_label = instance_label.replace(" ", "_")
            output_file = (
                output_dir / f"non_structural/{instance_label}_{instance_index}.usdz"
            )

        write_usdz(
            usdz_file=output_file,
            meshes=[instance_mesh],
            names=[instance_label],
        )

    # Write empty_room as USDZ
    instance_xyz = points[empty_room_mask[:, 0] == 1]
    instance_xyz = np.matmul(instance_xyz, rotation_matrix.T)
    instance_mesh = filter_mesh_by_proximity(
        mesh=mesh, points=instance_xyz, threshold_distance=0.02
    )
    write_usdz(
        usdz_file=output_dir / f"empty_room.usdz",
        meshes=[instance_mesh],
        names=["empty_room"],
    )

    if visualize:
        colors = colorize_instance_pointcloud(instances_mapped)
        trimesh.PointCloud(vertices=points, colors=colors).export(
            output_dir / "instances.ply", file_type="ply"
        )

        colors = colorize_semantic_pointcloud(labels, colormap=model_type)
        trimesh.PointCloud(vertices=points, colors=colors).export(
            output_dir / "semantics.ply", file_type="ply"
        )


if __name__ == "__main__":
    """
    This script performs 3D scene segmentation using the finetuned model from the Mask3D repository.
    The input is a USDZ mesh file and the output is a folder including segmented
    instances and semantic labels. The code in this script is adapted from the
    forked Mask3D repository: https://github.com/cvg/Mask3D
    
    NOTE: There are some issues with ScanNetV2 so ScanNetV2 is not suppported.
    ScanNetV2 doens't have the class `ceiling`. The inference code doesn't work.
    https://github.com/JonasSchult/Mask3D/issues/42

    NOTE: Currently, ScanNet200 doesn't support wall and floor instances.

    Arguments:
        --input-file (Path): Path to the input USDZ file.
        --output-dir (Path): Directory where the output files will be saved.
        --checkpoint (Path): Path to the segmentation model checkpoint.
        --model-type (str): Type of dataset used to train the segmentation model (default: 'scannet200').
        --pointcloud-resampling-factor (float): Factor for resampling the point cloud. Determines how many points to sample
                                                 from the mesh. If set to 10, it will sample 10 times the number of vertices.
                                                 Default is 10.
        --voxel-size (float): Size of the voxels in centimeters. Smaller values require more GPU memory. Default is 0.02.
        --up-axis (str): Axis that represents the UP direction in the input 3D model. Mask3D assumes +Z-axis is UP.
                         If the input model has a different UP axis (e.g., Y-axis), it needs to be rotated. Default is 'y'.
                         Choices are ['x', 'y', 'z'].
        --confidence-threshold (float): Confidence threshold for filtering predictions. Default is 0.85.
        --visualize (bool): If True, generates instance/semantic point cloud (.ply) files in the output directory.
                            Default is False.

    Note:
        This script currently only supports the ScanNet200 model for segmentation. Most of the code is derived
        from the forked Mask3D repository: https://github.com/cvg/Mask3D.
    """
    parser = argparse.ArgumentParser(description="3D Scene Segmentation")
    parser.add_argument(
        "--input-file", type=Path, help="Path to the input USDZ file", required=True
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory path", required=True
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Path to the segmentation model", required=True
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="s3dis",
        choices=["scannet200", "s3dis"],
        help="Type of dataset used to train the segmentation model (default: 'scannet200')",
    )
    parser.add_argument(
        "--pointcloud-resampling-factor",
        type=float,
        default=10,
        help="Factor for resampling the point cloud from the mesh. Determines how many points to sample from the mesh. "
        "If set to 10, it will sample 10 times the number of vertices in the mesh. Default is 10.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.02,
        help="Size of the voxels in centimeters. Smaller values require more GPU memory. Default is 0.02.",
    )
    parser.add_argument(
        "--up-axis",
        type=str,
        help="Axis that represents the UP direction in the input 3D model. If the input model has Y-axis as UP, "
        "it needs to be rotated because Mask3D assumes the input 3D model has +Z-axis as UP. Default is 'y'.",
        default="y",
        choices=["x", "y", "z"],
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Confidence threshold for filtering predictions. Default is 0.85.",
        default=0.85,
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        help="If True, generates instance/semantic point cloud (.ply) files in the output directory. Default is False.",
        default=False,
    )
    args = parser.parse_args()

    main(
        input_file=args.input_file,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        up_axis=args.up_axis,
        voxel_size=args.voxel_size,
        pointcloud_resampling_factor=args.pointcloud_resampling_factor,
        confidence_threshold=args.confidence_threshold,
        visualize=args.visualize,
    )
