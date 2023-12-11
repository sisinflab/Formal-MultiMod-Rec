def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break


import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        pass
    elif _backend == "pytorch":
        from .bprmf.BPRMF import BPRMF
        from .lightgcn.LightGCN import LightGCN
        from .ngcf.NGCF import NGCF
        from .sgl.SGL import SGL
        from .lightgcn_m.LightGCNM import LightGCNM
        from .grcn.GRCN import GRCN
        from .lattice.LATTICE import LATTICE
        from .freedom.FREEDOM import FREEDOM
        from .mmgcn.MMGCN import MMGCN
        from .vbpr.VBPR import VBPR
        from .bm3.BM3 import BM3
