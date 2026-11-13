import os
import hydra
import threading
import logging
import time
import traceback
import queue
import random
from glob import glob
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from transforms3d import quaternions as tq

from dexonomy.util.np_util import (
    np_array32,
    np_normal_to_rot,
    np_transform_points,
    np_inv_transform_points,
)
from dexonomy.sim import MuJoCo_VisEnv, HandCfg
from dexonomy.util.file_util import load_yaml


TEMPLATE_FEEDBACK_FORMAT = "dexonomy-template-feedback/v1"


class HandTemplateLoader:
    def __init__(
        self,
        xml_path,
        skt_path,
        tmpl_name,
        init_tmpl_dir,
        new_tmpl_dir,
        max_data_path=100,
        max_data_buffer=1024,
        n_worker=4,
        feedback_queue=None,
        feedback_accepting_event=None,
        feedback_cancel_event=None,
        feedback_max_evo=2,
        _autostart=True,
    ):
        # Producer components
        self.data_path_list = []
        self.data_path_lock = threading.Lock()
        self.data_path_condition = threading.Condition(self.data_path_lock)

        # Consumer components
        self.consumer_pool = ThreadPoolExecutor(max_workers=n_worker)

        # Buffer components
        self.data_buffer = {
            "hand_path": [],
            "grasp_qpos": [],
            "rot_c2h": [],
            "trans_c2h": [],
            "hand_cpn_c": [],
            "hand_skt_h": [],
            "n_evo": [],
        }

        self.buffer_lock = threading.Lock()

        # Control flags
        self.running = False
        self.producer_thread = None
        self.feedback_thread = None
        self.n_worker = n_worker
        self.max_data_path = max_data_path
        self.max_data_buffer = max_data_buffer

        # The original implementation discovers validated templates by scanning
        # ``new_tmpl_dir``.  The shared-memory V1.1 pipeline feeds the exact same
        # validated template payloads through this optional queue instead, avoiding
        # one small file/symlink per successful grasp.
        self.feedback_queue = feedback_queue
        self.feedback_accepting_event = feedback_accepting_event
        self.feedback_cancel_event = feedback_cancel_event
        self.feedback_max_evo = feedback_max_evo
        self.feedback_seen_ids = set()
        self.feedback_stats_lock = threading.Lock()
        self.feedback_stats = {
            "format_version": TEMPLATE_FEEDBACK_FORMAT,
            "enabled": feedback_queue is not None,
            "received": 0,
            "accepted": 0,
            "duplicate": 0,
            "rejected": 0,
            "errors": 0,
            "buffer_peak": 1,
            "sampled_slots": 0,
            "sampled_evolved_slots": 0,
            "batches_using_evolved_template": 0,
            "max_sampled_evolution": 0.0,
        }

        # Hand informations
        self.xml_path = xml_path
        self.init_tmpl_dir = init_tmpl_dir
        self.new_tmpl_dir = new_tmpl_dir
        self.tmpl_name = tmpl_name
        assert os.path.exists(os.path.join(init_tmpl_dir, tmpl_name + ".npy"))

        # Read and pre-process skeleton informations
        skt_dict = load_yaml(skt_path)
        vis_env = MuJoCo_VisEnv(hand_cfg=HandCfg(xml_path=xml_path))
        self.skt_data, self.skt_bid = [], []
        for k, v in skt_dict.items():
            self.skt_data.extend(v)
            self.skt_bid.extend([vis_env.get_body_id(k)] * len(v))
        self.skt_data, self.skt_bid = np_array32(self.skt_data), np.array(self.skt_bid)

        # Pre-read the initial template
        hand_data = self._load_data(
            vis_env, os.path.join(init_tmpl_dir, tmpl_name + ".npy")
        )
        for k in self.data_buffer.keys():
            self.data_buffer[k].append(
                torch.from_numpy(hand_data[k])
                if isinstance(hand_data[k], np.ndarray)
                else hand_data[k]
            )
        self.extra_info = {
            "tmpl_name": tmpl_name,
            "hand_cbody": hand_data["hand_cbody"],
            "required_cbody": hand_data["required_cbody"],
        }
        if _autostart:
            self.start()
        return

    def _record_feedback_stat(self, key, amount=1):
        with self.feedback_stats_lock:
            self.feedback_stats[key] += amount

    def get_feedback_stats(self):
        with self.feedback_stats_lock:
            return dict(self.feedback_stats)

    def _record_family_feedback(self, template_name, key):
        """Optional family-level telemetry; single-template behavior is unchanged."""
        return

    def _accepts_template_name(self, template_name):
        """Leave legacy metadata validation to the single-family append path."""
        return True

    def start(self):
        """Start all threads in the pipeline"""
        if self.running:
            return

        self.running = True
        # Start producer thread
        self.producer_thread = threading.Thread(
            target=self._update_data_path, daemon=True
        )
        self.producer_thread.start()

        # Start consumer threads
        for _ in range(self.n_worker):
            self.consumer_pool.submit(self._update_data_buffer)

        if self.feedback_queue is not None:
            self.feedback_thread = threading.Thread(
                target=self._update_feedback_buffer, daemon=True
            )
            self.feedback_thread.start()

    def stop(self):
        """Stop all threads gracefully"""
        if self.feedback_accepting_event is not None:
            self.feedback_accepting_event.clear()
        self.running = False

        # Wake up all waiting threads
        with self.data_path_condition:
            self.data_path_condition.notify_all()

        # Shutdown thread pool
        self.consumer_pool.shutdown(wait=True)

        # Wait for producer thread
        if self.producer_thread:
            self.producer_thread.join()

        if self.feedback_thread:
            self.feedback_thread.join(timeout=2.0)

    def _append_hand_data(self, hand_data):
        """Append one preprocessed template using the original FIFO semantics."""

        with self.buffer_lock:
            for k in self.extra_info.keys():
                assert self.extra_info[k] == hand_data[k], (
                    f"{k} {self.extra_info[k]} {hand_data[k]}"
                )
            if len(self.data_buffer["hand_path"]) >= self.max_data_buffer:
                for k in self.data_buffer.keys():
                    self.data_buffer[k].pop(0)
            for k in self.data_buffer.keys():
                self.data_buffer[k].append(
                    torch.from_numpy(hand_data[k])
                    if isinstance(hand_data[k], np.ndarray)
                    else hand_data[k]
                )
            buffer_length = len(self.data_buffer["hand_path"])
        with self.feedback_stats_lock:
            self.feedback_stats["buffer_peak"] = max(
                self.feedback_stats["buffer_peak"], buffer_length
            )

    def _update_feedback_buffer(self):
        """Load force-closure-approved templates from the in-memory feedback queue."""

        vis_env = MuJoCo_VisEnv(hand_cfg=HandCfg(xml_path=self.xml_path))
        while self.running:
            if (
                self.feedback_cancel_event is not None
                and self.feedback_cancel_event.is_set()
            ):
                return
            try:
                item = self.feedback_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._record_feedback_stat("received")
            template_name = None
            try:
                if not isinstance(item, dict) or item.get("format_version") != (
                    TEMPLATE_FEEDBACK_FORMAT
                ):
                    raise ValueError("invalid template feedback envelope")
                template_id = item.get("template_id")
                raw_data = item.get("data")
                if not isinstance(template_id, str) or not template_id:
                    raise ValueError("template feedback has no template_id")
                if not isinstance(raw_data, dict):
                    raise ValueError("template feedback has no data mapping")
                template_name = raw_data.get("tmpl_name")
                self._record_family_feedback(template_name, "received")
                n_evo = float(np.asarray(raw_data["n_evo"]).reshape(-1)[0])
                if (
                    n_evo > self.feedback_max_evo
                    or not self._accepts_template_name(template_name)
                ):
                    self._record_feedback_stat("rejected")
                    self._record_family_feedback(template_name, "rejected")
                    continue
                with self.feedback_stats_lock:
                    if template_id in self.feedback_seen_ids:
                        self.feedback_stats["duplicate"] += 1
                        duplicate = True
                    else:
                        duplicate = False
                if duplicate:
                    self._record_family_feedback(template_name, "duplicate")
                    continue
                hand_data = self._prepare_data(vis_env, raw_data, template_id)
                if self._append_hand_data(hand_data) is False:
                    self._record_feedback_stat("rejected")
                    self._record_family_feedback(template_name, "rejected")
                    continue
                with self.feedback_stats_lock:
                    self.feedback_seen_ids.add(template_id)
                    self.feedback_stats["accepted"] += 1
                self._record_family_feedback(template_name, "accepted")
            except Exception:
                self._record_feedback_stat("errors")
                self._record_family_feedback(template_name, "errors")
                logging.error(
                    "(Hand Loader) template feedback failed:\n%s",
                    traceback.format_exc(),
                )

    def _update_data_path(self):
        """Producer thread that generates data path lists to read"""
        while self.running:
            # Get all available data path
            template_lst = []
            for name in getattr(self, "template_names", [self.tmpl_name]):
                if self.new_tmpl_dir is not None:
                    template_lst.extend(glob(
                        os.path.join(self.new_tmpl_dir, name, "**/*.npy"),
                        recursive=True,
                    ))
                template_lst.append(
                    os.path.join(self.init_tmpl_dir, name + ".npy")
                )

            # Remove existing data path. NOTE: If there are workers loading data, it may generate repeat or miss path. But the problem is not severe.
            with self.buffer_lock:
                exist_path_set = set(self.data_buffer["hand_path"])
            template_lst = list(set(template_lst).difference(exist_path_set))
            logging.debug(f"(Producer) Current buffer: {len(exist_path_set)}")
            logging.debug(f"(Producer) Unduplicated path list: {len(template_lst)}")

            # Add new data path and notify workers
            with self.data_path_condition:
                if len(self.data_path_list) <= self.max_data_path:
                    n_new = min(
                        len(template_lst), self.max_data_path - len(self.data_path_list)
                    )
                    self.data_path_list.extend(template_lst[:n_new])
                    self.data_path_condition.notify(n_new)

            time.sleep(1.0)  # Control production rate

    def _update_data_buffer(self):
        """Consumer thread that load data and stores results in buffer"""
        vis_env = MuJoCo_VisEnv(hand_cfg=HandCfg(xml_path=self.xml_path))
        while self.running:
            data_path = None

            # Read from data path list
            with self.data_path_condition:
                while self.running and not self.data_path_list:
                    self.data_path_condition.wait()  # Wait for data
                if not self.running:
                    break
                data_path = self.data_path_list.pop(0)  # FIFO processing

            if data_path:
                try:
                    hand_data = self._load_data(vis_env, data_path)
                    self._append_hand_data(hand_data)
                except Exception:
                    error_traceback = traceback.format_exc()
                    logging.error(f"(Hand Loader) {error_traceback}")

    def _load_data(self, vis_env: MuJoCo_VisEnv, data_path: str):
        return self._prepare_data(
            vis_env, np.load(data_path, allow_pickle=True).item(), data_path
        )

    def _prepare_data(self, vis_env: MuJoCo_VisEnv, raw_data, data_path: str):
        """Preprocess a file-backed or in-memory template identically."""

        hand_data = dict(raw_data)
        hand_data["grasp_qpos"] = hand_data["grasp_qpos"].squeeze()
        xmat, xpos = vis_env.forward_kinematics(hand_data["grasp_qpos"][7:])
        skt_xmat, skt_xpos = xmat[self.skt_bid], xpos[self.skt_bid]
        hand_data["hand_skt_h"] = (
            np.reshape(self.skt_data, (-1, 2, 3)) @ skt_xmat.transpose(0, 2, 1)
            + skt_xpos[:, None, :]
        ).reshape(-1, 6)

        h_r = tq.quat2mat(hand_data["grasp_qpos"][3:7]).astype(np.float32)
        h_t = hand_data["grasp_qpos"][:3]
        h_cpn_h = np_inv_transform_points(hand_data["hand_cpn_w"], h_r, h_t)
        init_idx = np.random.randint(low=0, high=len(h_cpn_h))
        rot_c2h = np_normal_to_rot(h_cpn_h[init_idx, 3:].reshape(1, 3)).squeeze(0).T
        trans_c2h = -rot_c2h @ h_cpn_h[init_idx, :3]
        hand_data["rot_c2h"] = rot_c2h
        hand_data["trans_c2h"] = trans_c2h
        hand_data["hand_cpn_c"] = np_transform_points(h_cpn_h, rot_c2h, trans_c2h)
        hand_data["hand_path"] = data_path
        return hand_data

    def get_batched_data(self, data_size: torch.Size, device: torch.device):
        """Get processed data from buffer (thread-safe)"""
        with self.buffer_lock:
            buffer_length = len(self.data_buffer["hand_path"])
            rand_id = torch.randint(low=0, high=buffer_length, size=data_size)
            new_data = {}
            logging.info(f"(Hand Loader) Current buffer: {buffer_length}")
            for k, v in self.data_buffer.items():
                if isinstance(v[0], torch.Tensor):
                    selected = torch.stack(v)[rand_id]
                    new_data[k] = selected.to(device)
                    if k == "n_evo":
                        sampled_n_evo = selected
            for k, v in self.extra_info.items():
                new_data[k] = v
            new_data["audit_template_sources"] = [self.data_buffer["hand_path"][int(i)]
                for i in rand_id.reshape(-1).tolist()]
            evolved = sampled_n_evo > 0
            with self.feedback_stats_lock:
                self.feedback_stats["sampled_slots"] += int(sampled_n_evo.numel())
                self.feedback_stats["sampled_evolved_slots"] += int(evolved.sum())
                if bool(evolved.any()):
                    self.feedback_stats["batches_using_evolved_template"] += 1
                self.feedback_stats["max_sampled_evolution"] = max(
                    self.feedback_stats["max_sampled_evolution"],
                    float(sampled_n_evo.max()),
                )
            return new_data


class MultiHandTemplateLoader(HandTemplateLoader):
    """Share one bounded library while batching only one contact topology at a time."""

    def __init__(self, *, tmpl_names, selection_seed=12, **kwargs):
        self.template_names = sorted(set(tmpl_names))
        if len(self.template_names) < 1:
            raise ValueError("template loading requires at least one family")
        if kwargs.get("max_data_buffer", 1024) < len(self.template_names):
            raise ValueError("template buffer must fit one seed per family")
        self.selection_seed = int(selection_seed)
        self._selection_rng = random.Random(self.selection_seed)
        self._selection_cycle = []
        super().__init__(
            tmpl_name=self.template_names[0], _autostart=False, **kwargs
        )
        try:
            # Seeds stay at the start of the shared buffer so every family
            # remains selectable even when another family dominates feedback.
            self.data_buffer["tmpl_name"] = [self.template_names[0]]
            self.family_extra_info = {
                self.template_names[0]: dict(self.extra_info)
            }
            vis_env = MuJoCo_VisEnv(hand_cfg=HandCfg(xml_path=self.xml_path))
            for name in self.template_names[1:]:
                data = self._load_data(
                    vis_env, os.path.join(self.init_tmpl_dir, name + ".npy")
                )
                if data["tmpl_name"] != name:
                    raise ValueError(f"template payload name does not match {name}")
                self.family_extra_info[name] = {
                    key: data[key] for key in self.extra_info
                }
                for key in self.data_buffer:
                    value = data[key]
                    self.data_buffer[key].append(
                        torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                    )
            self.feedback_stats.update({
                "buffer_peak": len(self.template_names),
                "per_template": {
                    name: {
                        "received": 0, "accepted": 0, "duplicate": 0,
                        "rejected": 0, "errors": 0, "selected_batches": 0,
                        "sampled_slots": 0, "sampled_evolved_slots": 0,
                    }
                    for name in self.template_names
                },
                "selection_policy": "one_family_per_object_batch_balanced_shuffled_cycles",
                "selection_seed": self.selection_seed,
                "template_names": list(self.template_names),
            })
            self.start()
        except BaseException:
            self.stop()
            raise

    def _accepts_template_name(self, template_name):
        """Route incoming feedback only to a selected initial family."""
        return template_name in self.family_extra_info

    def _record_family_feedback(self, template_name, key):
        """Record diagnostics for known families without expanding the library."""
        with self.feedback_stats_lock:
            if template_name in self.feedback_stats["per_template"]:
                self.feedback_stats["per_template"][template_name][key] += 1

    def get_feedback_stats(self):
        """Return a snapshot including family selection and feedback counts."""
        with self.feedback_stats_lock:
            result = dict(self.feedback_stats)
            result["per_template"] = {
                name: dict(stats)
                for name, stats in self.feedback_stats["per_template"].items()
            }
            result["template_names"] = list(self.template_names)
            return result

    def select_template(self):
        """Consume one seeded permutation slot per existing object batch."""
        if not self._selection_cycle:
            self._selection_cycle = list(self.template_names)
            self._selection_rng.shuffle(self._selection_cycle)
        name = self._selection_cycle.pop()
        with self.feedback_stats_lock:
            self.feedback_stats["per_template"][name]["selected_batches"] += 1
        return name

    def _append_hand_data(self, hand_data):
        """Append within a family, evicting the oldest non-seed globally."""
        name = hand_data["tmpl_name"]
        if not self._accepts_template_name(name):
            return False
        with self.buffer_lock:
            for key, expected in self.family_extra_info[name].items():
                if hand_data[key] != expected:
                    raise ValueError(f"inconsistent {key} in template family {name}")
            seed_count = len(self.template_names)
            if self.max_data_buffer == seed_count:
                return False
            if len(self.data_buffer["hand_path"]) >= self.max_data_buffer:
                for values in self.data_buffer.values():
                    values.pop(seed_count)
            for key in self.data_buffer:
                value = hand_data[key]
                self.data_buffer[key].append(
                    torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                )
            buffer_length = len(self.data_buffer["hand_path"])
        with self.feedback_stats_lock:
            self.feedback_stats["buffer_peak"] = max(
                self.feedback_stats["buffer_peak"], buffer_length
            )
        return True

    def get_batched_data(self, data_size, device, tmpl_name=None):
        """Sample evolved variants only within the batch's selected family."""
        name = tmpl_name if tmpl_name is not None else self.select_template()
        with self.buffer_lock:
            indices = [
                index for index, family in enumerate(self.data_buffer["tmpl_name"])
                if family == name
            ]
            if not indices:
                raise ValueError(f"unknown or empty template family: {name}")
            rand_id = torch.randint(low=0, high=len(indices), size=data_size)
            result = {}
            for key, values in self.data_buffer.items():
                if isinstance(values[indices[0]], torch.Tensor):
                    selected = torch.stack([values[index] for index in indices])[rand_id]
                    result[key] = selected.to(device)
            result.update(self.family_extra_info[name])
            result["audit_template_sources"] = [self.data_buffer["hand_path"][indices[int(i)]] for i in rand_id.reshape(-1)]
        sampled_evo = result["n_evo"]
        sampled_slots = int(sampled_evo.numel())
        evolved_slots = int((sampled_evo > 0).sum())
        with self.feedback_stats_lock:
            self.feedback_stats["sampled_slots"] += sampled_slots
            self.feedback_stats["sampled_evolved_slots"] += evolved_slots
            self.feedback_stats["batches_using_evolved_template"] += int(evolved_slots > 0)
            self.feedback_stats["max_sampled_evolution"] = max(
                self.feedback_stats["max_sampled_evolution"], float(sampled_evo.max())
            )
            family_stats = self.feedback_stats["per_template"][name]
            family_stats["sampled_slots"] += sampled_slots
            family_stats["sampled_evolved_slots"] += evolved_slots
        return result


@hydra.main(config_path="../config", config_name="base", version_base=None)
def test_hand_library(cfg):
    tmpl_name = os.listdir(cfg.init_tmpl_dir)[-1].split(".npy")[0]
    hand_library = HandTemplateLoader(
        xml_path=cfg.hand.xml_path,
        skt_path=cfg.hand_skt_path,
        tmpl_name=tmpl_name,
        init_tmpl_dir=cfg.init_tmpl_dir,
        new_tmpl_dir=cfg.new_tmpl_dir,
        n_worker=cfg.n_worker,
    )
    try:
        for _ in range(20):
            for i in range(3):
                time.sleep(0.1)
                d = hand_library.get_batched_data(
                    torch.Size([10, 8192]), torch.device("cpu")
                )
                logging.info(f"Batched data shape: {d['grasp_qpos'].shape}")
            time.sleep(0.8)
    finally:
        hand_library.stop()
        logging.info("Pipeline stopped")

    return


if __name__ == "__main__":
    test_hand_library()
