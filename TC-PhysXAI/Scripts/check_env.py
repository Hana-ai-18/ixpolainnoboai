"""Kiểm tra môi trường Kaggle: import PhysXAI, Model (TC-XFlow), checkpoint, dataset."""
import sys, os, glob
from pathlib import Path

PHYS = Path("/kaggle/working/TC-PhysXAI")
XFLOW = Path("/kaggle/working/Tropical_Typhoon_Architecture")
CKPT_DIR = "/kaggle/input/datasets/gmnguynhng/new-checkpoint"
DATA = "/kaggle/input/datasets/kaggle1234uitvn/tc-ofm"
sys.path.insert(0, str(PHYS)); sys.path.insert(0, str(XFLOW))

ok = True
def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name} {extra}")

check("TC-PhysXAI dir", PHYS.is_dir(), str(PHYS))
check("TC-XFlow Model dir", (XFLOW / "Model").is_dir(), str(XFLOW / "Model"))
try:
    import PhysXAI; check("import PhysXAI", True, f"v{PhysXAI.__version__}")
except Exception as e: check("import PhysXAI", False, repr(e))
try:
    from Model.flow_matching_model import TCFlowMatching, hard_score_from_obs
    from Model.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate
    check("import Model.* (TC-XFlow)", True)
except Exception as e: check("import Model.* (TC-XFlow)", False, repr(e))
ck = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pth")))
check("checkpoints", len(ck) > 0, f"{len(ck)} file(s)")
for c in ck: print("    ", os.path.basename(c))
check("dataset root", os.path.isdir(DATA), DATA)
if os.path.isdir(DATA): print("     contents:", sorted(os.listdir(DATA))[:10])
check("Data1d/test", os.path.isdir(os.path.join(DATA, "Data1d", "test")))
try:
    import torch; check("CUDA", torch.cuda.is_available(), torch.__version__)
except Exception as e: check("torch", False, repr(e))
print("\nALL OK" if ok else "\nCó mục FAIL — xem ở trên.")
