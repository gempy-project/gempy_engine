from dataclasses import dataclass


@dataclass
class MatricesSizes:
    ori_size: int
    sp_size: int
    uni_drift_size: int
    faults_size: int
    dim: int
    n_dips: int = None
    grid_size: int = None

    @property
    def cov_size(self):
        return self.ori_size + self.sp_size + self.uni_drift_size + self.faults_size

    @property
    def drifts_size(self):
        return self.uni_drift_size + self.faults_size
