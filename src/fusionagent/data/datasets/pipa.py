import csv
import logging
import os.path as osp

import h5py
import numpy as np


class PIPA(object):
    """PIPA adapter for FusionAgent.

    Expected files under <root>/PIPA_FusionAgent:
      - manifest.csv
      - pipa_head.h5
      - pipa_upper.h5

    manifest.csv is produced by prepare_pipa.py and contains train/test0/test1
    rows. The PIPA protocol is represented as 0to1 (gallery=test0,
    query=test1) or 1to0 (gallery=test1, query=test0).

    PIPA supplies a ground-truth head box, not a full-body box. We therefore
    expose two image cues:
      head_data: exact annotated head crop
      body_data: PIPA upper-body crop (3 x head width, 3 x head height)
    face_data aliases head_data only for compatibility with FusionAgent's
    current trainer, which expects a second image stream named face_data.
    """
    dataset_dir = "PIPA_FusionAgent"

    def __init__(self, root="data", **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.manifest_path = osp.join(self.dataset_dir, "manifest.csv")
        self.head_path = osp.join(self.dataset_dir, "pipa_head.h5")
        self.upper_path = osp.join(self.dataset_dir, "pipa_upper.h5")
        self.protocol = kwargs.get("protocol", "0to1")
        self.few_shot = kwargs.get("few_shot", None)
        if self.protocol not in {"0to1", "1to0"}:
            raise ValueError("PIPA protocol must be '0to1' or '1to0'")

        self._check_before_run()
        rows = self._read_manifest()

        train_rows = [r for r in rows if r["split"] == "train"]
        test0_rows = [r for r in rows if r["split"] == "test0"]
        test1_rows = [r for r in rows if r["split"] == "test1"]
        if not train_rows or not test0_rows or not test1_rows:
            raise RuntimeError(
                "PIPA manifest must contain non-empty train, test0, and test1 splits"
            )

        train, num_train_pids, pid2clothes = self._build_train(train_rows)
        if self.protocol == "0to1":
            gallery_rows, query_rows = test0_rows, test1_rows
        else:
            gallery_rows, query_rows = test1_rows, test0_rows

        # Keep test identity IDs unchanged. Use different pseudo-camera IDs for
        # query/gallery so ReID evaluation code never removes valid PIPA mates.
        gallery = self._build_eval(gallery_rows, camid=1)
        query = self._build_eval(query_rows, camid=0)

        self.head_data = h5py.File(self.head_path, "r")
        self.upper_data = h5py.File(self.upper_path, "r")
        # Compatibility aliases used by the upstream FusionAgent trainer.
        self.body_data = self.upper_data
        self.face_data = self.head_data

        self.train = {
            "dataset": train,
            "dataset_name": "pipa",
            "head_data": self.head_data,
            "body_data": self.body_data,
            "face_data": self.face_data,
        }
        self.query = {
            "dataset": query,
            "head_data": self.head_data,
            "body_data": self.body_data,
            "face_data": self.face_data,
        }
        self.gallery = {
            "dataset": gallery,
            "head_data": self.head_data,
            "body_data": self.body_data,
            "face_data": self.face_data,
        }

        self.num_train_pids = num_train_pids
        self.num_train_clothes = 1
        self.pid2clothes = pid2clothes
        self.dataset_name = "pipa"

        logger = logging.getLogger("reid.dataset")
        logger.info(
            "=> PIPA loaded: protocol=%s train=%d query=%d gallery=%d train_ids=%d",
            self.protocol, len(train), len(query), len(gallery), num_train_pids,
        )

    def _check_before_run(self):
        for path in [self.dataset_dir, self.manifest_path, self.head_path, self.upper_path]:
            if not osp.exists(path):
                raise RuntimeError("'{}' is not available".format(path))

    def _read_manifest(self):
        with open(self.manifest_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        required = {"key", "pid", "split", "album_id"}
        if not rows:
            return []
        missing = required.difference(rows[0].keys())
        if missing:
            raise RuntimeError("PIPA manifest missing columns: {}".format(sorted(missing)))
        return rows

    def _build_train(self, rows):
        pids = sorted({int(r["pid"]) for r in rows})
        pid2label = {pid: i for i, pid in enumerate(pids)}
        album_ids = sorted({r["album_id"] for r in rows})
        album2cam = {album: i for i, album in enumerate(album_ids)}

        by_pid = {}
        for r in rows:
            by_pid.setdefault(int(r["pid"]), []).append(r)

        dataset = []
        for pid in pids:
            samples = sorted(by_pid[pid], key=lambda r: r["key"])
            if self.few_shot is not None:
                samples = samples[: min(int(self.few_shot), len(samples))]
            for r in samples:
                dataset.append(
                    (r["key"], pid2label[pid], album2cam[r["album_id"]], 0)
                )

        pid2clothes = np.ones((len(pids), 1), dtype=np.float32)
        return dataset, len(pids), pid2clothes

    @staticmethod
    def _build_eval(rows, camid):
        dataset = []
        for r in sorted(rows, key=lambda x: x["key"]):
            dataset.append((r["key"], int(r["pid"]), camid, 0))
        return dataset
