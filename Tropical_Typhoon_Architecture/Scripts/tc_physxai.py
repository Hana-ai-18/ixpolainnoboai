"""run_xai.py — chạy TC-PhysXAI trên checkpoint TC-XFlow (KHÔNG train lại)."""
from __future__ import annotations
import argparse, csv, math, os, sys, time
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from Model.data.loader_training import data_loader        # đúng loader mà train_flowmatching.py dùng
from tc_physxai import (TCPhysXAI, load_tcxflow, move_batch, encode_context,
                        draw_noise, integrate, spread_km, HZ)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--bank", default="/kaggle/working/xai/bank.pt")
    p.add_argument("--build_bank", action="store_true")
    p.add_argument("--bank_max_n", type=int, default=30000)
    p.add_argument("--bank_batch_size", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=4, help="IG tốn bộ nhớ → 4–8")
    p.add_argument("--n_batches", type=int, default=3, help="số batch SẼ xử lý")
    p.add_argument("--stride", type=int, default=1, help="lấy 1 batch mỗi <stride> batch (phủ đều tập test)")
    p.add_argument("--print_n", type=int, default=2, help="in tóm tắt cho bao nhiêu storm mỗi batch")
    p.add_argument("--n_ensemble", type=int, default=30)
    p.add_argument("--ig_steps", type=int, default=16)
    p.add_argument("--ig_k_noise", type=int, default=8, help="số noise cho IG mục tiêu 'spread' (đắt nhất)")
    p.add_argument("--var_M", type=int, default=6, help="số lần nhiễu ERA5 cho C5")
    p.add_argument("--noise_stds", default="0.1", help="σ nhiễu ERA5 cho C5, ví dụ 0.05,0.1,0.2")
    p.add_argument("--tau_km", type=float, default=10.0)
    p.add_argument("--min_sil", type=float, default=0.5)
    p.add_argument("--min_sep_km", type=float, default=75.0)
    p.add_argument("--spatial", default="lateral,spread",
                   help="rỗng = bỏ IG; hoặc lateral,along,displacement,spread")
    p.add_argument("--fast", action="store_true", help="ig_steps<=8, ig_k_noise<=4, var_M<=4")
    p.add_argument("--no_variance", action="store_true")
    p.add_argument("--no_fidelity", action="store_true")
    p.add_argument("--run_c4", action="store_true", help="THỰC NGHIỆM, chưa kiểm chứng")
    p.add_argument("--self_check", action="store_true")
    p.add_argument("--save_ig", action="store_true", help="lưu bản đồ IG gọn (fp16) để vẽ hình")
    p.add_argument("--out_csv", default="/kaggle/working/xai/xai_rows.csv")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def main():
    a = get_args()
    if a.fast:
        a.ig_steps = min(a.ig_steps, 8); a.ig_k_noise = min(a.ig_k_noise, 4); a.var_M = min(a.var_M, 4)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.dirname(os.path.abspath(a.out_csv)); os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.bank)), exist_ok=True)
    print(f"  Device: {device}")

    model, _ = load_tcxflow(a.checkpoint, device)
    xai = TCPhysXAI(model, device, n_ensemble=a.n_ensemble, ig_steps=a.ig_steps,
                    ig_k_noise=a.ig_k_noise, var_M=a.var_M,
                    cluster_kwargs={"min_sil": a.min_sil, "min_sep_km": a.min_sep_km})

    dl = argparse.Namespace(obs_len=8, pred_len=12, batch_size=a.batch_size, num_workers=a.num_workers,
                            other_modal="gph", delim=" ", skip=1, min_ped=1, threshold=0.002,
                            filter_region=False, min_pct_in_scs=15.0, seed=None)

    if a.build_bank or not os.path.exists(a.bank):
        print("Dựng bank từ tập train (chạy 1 lần)...")
        _, tr = data_loader(dl, {"root": a.dataset_root, "type": "train"}, test=False, batch_size=a.bank_batch_size)
        xai.build_bank(tr, max_n=a.bank_max_n); xai.save_bank(a.bank)
    else:
        xai.load_bank(a.bank)

    _, loader = data_loader(dl, {"root": a.dataset_root, "type": a.split}, test=True, batch_size=a.batch_size)

    if a.self_check:        # ensemble của XAI phải cùng bậc spread với model.sample()
        bl = move_batch(list(next(iter(loader))), device)
        with torch.no_grad():
            _, _, all_t = xai.raw.sample(bl, num_ensemble=a.n_ensemble)
            cond = encode_context(xai.raw, bl)
            mine = integrate(xai.raw, cond, draw_noise(xai.raw, a.n_ensemble, bl[0].shape[1], device), bl[0][-1, :, :2])
        for k, s in HZ.items():
            ref = float(spread_km(all_t, [s]).mean()); got = float(spread_km(mine, [s]).mean())
            ratio = got / max(ref, 1e-6)
            flag = "OK" if 0.5 <= ratio <= 2.0 else "LỆCH — DỪNG và kiểm tra integrate()"
            print(f"  [self_check] spread {k}: model.sample={ref:.1f} km   XAI={got:.1f} km   tỉ lệ={ratio:.2f}  {flag}")

    spatial = tuple(x for x in a.spatial.split(",") if x)
    noise_stds = tuple(float(x) for x in a.noise_stds.split(",") if x)
    rows, times, agg = [], [], defaultdict(float)
    done = 0; t_start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for i, batch in enumerate(loader):
        if i % a.stride != 0:
            continue
        if done >= a.n_batches:
            break
        bl = move_batch(list(batch), device)
        tb = time.perf_counter()
        res = xai.explain(bl, spatial_objectives=spatial, run_variance=not a.no_variance,
                          run_fidelity=not a.no_fidelity, run_c4=a.run_c4,
                          noise_stds=noise_stds, tau_km=a.tau_km, seed=i)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - tb)
        for k, v in res.timings.items():
            agg[k] += v
        for b in range(min(a.print_n, res.meta["B"])):
            res.print_summary(b)
        if res.fidelity:
            s = res.fidelity["summary"]
            print("[Fidelity batch %d] P_own=%.2f P_rand=%.2f P_sal=%.2f | R_own=%.2f R_rand=%.2f | "
                  "Spatial-F=%.2f Unc-F=%.2f H-mean=%.2f   (trung bình theo storm trong batch; "
                  "thống kê chuẩn dùng analyze_xai.py)" % (
                      i, s.get("prec_own", float("nan")), s.get("prec_rand", float("nan")),
                      s.get("prec_sal", float("nan")), s.get("rec_own", float("nan")),
                      s.get("rec_rand", float("nan")), s.get("spatial_f", float("nan")),
                      s.get("unc_f", float("nan")), s.get("h_mean_f", float("nan"))))
        rows.extend(res.rows())
        if a.save_ig and res.spatial:
            ig_dir = os.path.join(out_dir, "ig_maps"); os.makedirs(ig_dir, exist_ok=True)
            torch.save({"storms": ["_".join(k) for k in res.meta["keys"]], "dates": res.meta["dates"],
                        "era5_last_frame_z": bl[11][:, :, -1].cpu().half(),
                        "ig_signed_sum_over_time": {o: sp_["ig"].sum(2).half() for o, sp_ in res.spatial.items()}},
                       os.path.join(ig_dir, f"ig_batch{i:04d}.pt"))
        done += 1
        print(f"  batch {done}/{a.n_batches} (loader idx {i}) xong: {times[-1]:.1f}s  "
              f"[tổng {time.time() - t_start:.0f}s]")

    keys = sorted({k for r in rows for k in r})
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nĐã lưu {len(rows)} dòng → {a.out_csv}")

    # ── kiểm tra nhanh chất lượng IG ──
    for obj in spatial:
        errs = [r[f"ig_{obj}_complete_err"] for r in rows if r.get("has_era5") and f"ig_{obj}_complete_err" in r]
        base = [r[f"ig_{obj}_baseline"] for r in rows if r.get("has_era5") and f"ig_{obj}_baseline" in r]
        if errs:
            print(f"  [IG:{obj}] completeness lỗi tương đối: trung vị {np.median(errs):.0%}, p90 {np.percentile(errs, 90):.0%} "
                  f"(cao → tăng --ig_steps) | baseline composite: "
                  f"{sum(b == 'composite_theo_thang' for b in base)}/{len(base)}")

    # ── báo cáo thời gian thật + ngoại suy ──
    if times:
        print("\n  ═══ THỜI GIAN THẬT (giây / batch, trung bình) ═══")
        use = times[1:] if len(times) >= 3 else times           # bỏ batch đầu (khởi tạo CUDA)
        for k in sorted(agg):
            print(f"    {k:<22s} {agg[k] / len(times):8.2f}")
        mean_b = float(np.mean(use))
        n_cand = math.ceil(len(loader) / a.stride)
        print(f"    {'TỔNG / batch':<22s} {mean_b:8.2f}   (B={a.batch_size})")
        print(f"    Ngoại suy toàn bộ {n_cand} batch (loader {len(loader)}, stride {a.stride}): "
              f"≈ {mean_b * n_cand / 3600:.2f} giờ  [giả định các batch tương tự nhau]")
        if device.type == "cuda":
            print(f"    Bộ nhớ GPU đỉnh: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()