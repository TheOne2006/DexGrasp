"""Shared grasp filters and data normalization used by the resident backend."""
import logging
import numpy as np
from dexonomy.sim import MuJoCo_OptEnv
from dexonomy.qp.qp_single import ContactQP
from dexonomy.util.np_util import np_array32

class GraspFilter:
    def __init__(self, cfg: dict, sim_env: MuJoCo_OptEnv, log_name: str):
        self.cfg = cfg
        self.sim_env = sim_env
        self.log_name = log_name

    def forward(self, info: dict) -> tuple[bool, dict]:
        contact_thre = self.cfg.contact.thre if "contact" in self.cfg else None
        ho_c, hh_c = self.sim_env.get_contacts(contact_thre)
        if "limit" in self.cfg and self.cfg.limit and not self._limit_filter():
            return False, ho_c
        if "collision" in self.cfg and self.cfg.collision:
            if not self._collision_filter(ho_c, hh_c):
                return False, ho_c
        if "contact" in self.cfg and self.cfg.contact:
            if not self._contact_filter(ho_c, info["required_cbody"]):
                return False, ho_c
        if "qp" in self.cfg and self.cfg.qp:
            if not self._qp_filter(ho_c, info["ext_wrench"], info["ext_center"]):
                return False, ho_c
        return True, ho_c

    def early_stop(self) -> bool:
        contact_thre = self.cfg.contact.thre if "contact" in self.cfg else None
        ho_c, hh_c = self.sim_env.get_contacts(contact_thre)
        return self._collision_filter(ho_c, hh_c, slience=True)

    def _collision_filter(self, ho_c: dict, hh_c: dict, slience: bool = False) -> bool:
        col_cfg = self.cfg.collision
        if len(hh_c["dist"]) > 0 and min(hh_c["dist"]) < col_cfg.hh_thre:
            if not slience:
                logging.debug(f"{self.log_name} collision hh {min(hh_c['dist'])}")
            return False

        if len(ho_c["dist"]) > 0 and min(ho_c["dist"]) < col_cfg.ho_thre:
            if not slience:
                logging.debug(f"{self.log_name} collision ho {min(ho_c['dist'])}")
            return False
        return True

    def _contact_filter(self, ho_c: dict, required_cbody: list[str]):
        contact_cfg = self.cfg.contact
        if len(ho_c["dist"]) == 0:
            logging.debug(f"{self.log_name} no contact")
            return False
        ho_cb = set(ho_c["bn1"])
        if contact_cfg.check_body:
            for rcb in required_cbody:
                if len(ho_cb.intersection(set(rcb))) == 0:
                    logging.debug(f"{self.log_name} required: {rcb} current: {ho_cb}")
                    return False
        return True

    def _qp_filter(
        self, ho_c: dict, ext_wrench: np.ndarray, ext_center: np.ndarray
    ) -> bool:
        qp_cfg = self.cfg.qp
        contact_wrenches, wrench_error = ContactQP(miu_coef=qp_cfg.miu_coef).solve(
            ho_c["pos"], ho_c["normal"], ext_wrench, ext_center
        )
        if wrench_error > qp_cfg.thre or contact_wrenches is None:
            logging.debug(f"{self.log_name} bad QP error {wrench_error}")
            return False
        ho_c["wrench"] = 10 * contact_wrenches
        return True

    def _limit_filter(self) -> bool:
        return self.sim_env.check_qpos_limit(self.cfg.limit.thre)



def _normalize_grasp_data(grasp_data: dict, cfg: dict) -> tuple[dict, None]:
    """Preserve current-format candidate data without modifying the caller."""
    if cfg.legacy_api:
        raise ValueError("DexonomyOpt accepts only the current candidate format")
    return dict(grasp_data), None
