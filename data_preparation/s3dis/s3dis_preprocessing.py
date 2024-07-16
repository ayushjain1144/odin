import os
import re
import ipdb

import numpy as np
from fire import Fire
from loguru import logger
from natsort import natsorted

from data_preparation.base_preprocessing import BasePreprocessing

st = ipdb.set_trace


class S3DISPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/raw/s3dis",
        save_dir: str = "./data/processed/s3dis",
        modes: tuple = (
            "Area_1",
            "Area_2",
            "Area_3",
            "Area_4",
            "Area_5",
            "Area_6",
        ),
        n_jobs: int = -1,
    ):
        # n_jobs = 1
        super().__init__(data_dir, save_dir, modes, n_jobs)

        self.class_map = {
            "ceiling": 0,
            "floor": 1,
            "wall": 2,
            "beam": 3,
            "column": 4,
            "window": 5,
            "door": 6,
            "table": 7,
            "chair": 8,
            "sofa": 9,
            "bookcase": 10,
            "board": 11,
            "clutter": 12,
            "stairs": 12,  # stairs are also mapped to clutter
        }

        self.color_map = [
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
            [50, 50, 50],
        ]  # clutter

        for mode in self.modes:
            filepaths = []
            for scene_path in [
                f.path for f in os.scandir(self.data_dir / mode) if f.is_dir()
            ]:
                filepaths.append(scene_path)
            self.files[mode] = natsorted(filepaths)

    def _buf_count_newlines_gen(self, fname):
        def _make_gen(reader):
            while True:
                b = reader(2**16)
                if not b:
                    break
                yield b

        with open(fname, "rb") as f:
            count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
        return count

    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        filebase = {
            "filepath": filepath,
            "scene": filepath.split("/")[-1],
            "area": mode,
            "raw_filepath": str(filepath),
            "file_len": -1,
        }

        scene_name = filepath.split("/")[-1]
        instance_counter = 0
        scene_points = []
        for instance in [
            f
            for f in os.scandir(
                self.data_dir / mode / scene_name / "Annotations"
            )
            if f.name.endswith(".txt")
        ]:
            instance_class = self.class_map[instance.name.split("_")[0]]
            instance_points = np.loadtxt(instance.path)

            instance_normals = np.ones((instance_points.shape[0], 3))
            instance_class = np.array(instance_class).repeat(
                instance_points.shape[0]
            )[..., None]
            instance_id = np.array(instance_counter).repeat(
                instance_points.shape[0]
            )[..., None]

            instance_points = np.hstack(
                (
                    instance_points,
                    instance_normals,
                    instance_class,
                    instance_id,
                )
            )

            scene_points.append(instance_points)
            instance_counter += 1

        points = np.vstack(scene_points)

        pcd_size = self._buf_count_newlines_gen(f"{filepath}/{scene_name}.txt")
        if points.shape[0] != pcd_size:
            print(f"FILE SIZE DOES NOT MATCH FOR {filepath}/{scene_name}.txt")
            print(f"({points.shape[0]} vs. {pcd_size})")

        filebase["raw_segmentation_filepath"] = ""

        # add segment id as additional feature (DUMMY)
        points = np.hstack((points, np.ones(points.shape[0])[..., None]))
        points[:, [9, 10, -1]] = points[
            :, [-1, 9, 10]
        ]  # move segments after RGB

        gt_data = (points[:, -2] + 1) * 1000 + points[:, -1] + 1

        file_len = len(points)
        filebase["file_len"] = file_len

        processed_filepath = self.save_dir / mode / f"{scene_name}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        processed_gt_filepath = (
            self.save_dir / "instance_gt" / mode / f"{scene_name}.txt"
        )
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)

        filebase["color_mean"] = [
            float((points[:, 3] / 255).mean()),
            float((points[:, 4] / 255).mean()),
            float((points[:, 5] / 255).mean()),
        ]
        filebase["color_std"] = [
            float(((points[:, 3] / 255) ** 2).mean()),
            float(((points[:, 4] / 255) ** 2).mean()),
            float(((points[:, 5] / 255) ** 2).mean()),
        ]
        return filebase

    @logger.catch
    def fix_bugs_in_labels(self):
        pass

    def joint_database(self, train_modes=("Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6")):
        # train_modes = ("Area_5")
        for mode in train_modes:
            joint_db = []
            for let_out in train_modes:
                if mode == let_out:
                    continue
                joint_db.extend(self._load_yaml(self.save_dir / (let_out + "_database.yaml")))
            self._save_yaml(self.save_dir / f"train_{mode}_database.yaml", joint_db)
 
        joint_db = []
        for mode in train_modes:
            joint_db.extend(self._load_yaml(self.save_dir / (mode + "_database.yaml")))
        self._save_yaml(self.save_dir / "train_validation_database.yaml", joint_db)

    def _parse_scene_subscene(self, name):
        scene_match = re.match(r"scene(\d{4})_(\d{2})", name)
        return int(scene_match.group(1)), int(scene_match.group(2))


if __name__ == "__main__":
    Fire(S3DISPreprocessing)