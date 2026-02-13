from .generalizable_INR.gimmvfi_r import GIMMVFI_R
from .generalizable_INR.gimmvfi_f import GIMMVFI_F
from .generalizable_INR.configs import GIMMVFIConfig
from .generalizable_INR.raft.raft import RAFT as GIMM_RAFT
from .generalizable_INR.flowformer.core.FlowFormer.LatentCostFormer.transformer import FlowFormer as GIMM_FlowFormer
from .generalizable_INR.flowformer.configs.submission import get_cfg as gimm_get_flowformer_cfg
from .utils.utils import InputPadder as GIMMInputPadder, RaftArgs as GIMMRaftArgs, easydict_to_dict
from .generalizable_INR.modules.softsplat import objCudacache as gimm_softsplat_cache


def clear_gimm_caches():
    """Clear cached CUDA kernels and warp grids for GIMM-VFI."""
    from .generalizable_INR.modules.fi_utils import backwarp_tenGrid
    backwarp_tenGrid.clear()
    gimm_softsplat_cache.clear()
