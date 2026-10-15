"""
analyze_xai.py — tổng hợp CSV của run_xai.py. Mọi khoảng tin cậy dùng bootstrap THEO STORM
(các cửa sổ cùng storm tương quan mạnh → đơn vị độc lập là storm, không phải cửa sổ).

Cách dùng:
  python analyze_xai.py --csv /kaggle/working/xai/xai_rows.csv --out_dir /kaggle/working/xai/report
  (tuỳ chọn) --enso_csv oni.csv   # cột: year,month,oni   (month 1..12)
"""
from __future__ import annotations
import argparse, os, re
import numpy as np
import pandas as pd


def _ci(bs):
    bs = np.asarray(bs)
    if bs.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def _per_storm(df, col, mask, storms):
    d = df.loc[mask, ["storm", col]].dropna()
    g = d.groupby("storm")[col].agg(["sum", "count"]).reindex(storms).fillna(0.0)
    return g["sum"].to_numpy(float), g["count"].to_numpy(float)


def boot_mean(df, col, mask=None, n_boot=2000, seed=0):
    """(ước lượng, lo, hi, số cửa sổ, số storm) — bootstrap theo storm."""
    if col not in df.columns:
        return (float("nan"),) * 3 + (0, 0)
    mask = pd.Series(True, index=df.index) if mask is None else mask
    storms = df["storm"].unique()
    s, c = _per_storm(df, col, mask, storms)
    if c.sum() == 0:
        return (float("nan"),) * 3 + (0, 0)
    est = s.sum() / c.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(storms), size=(n_boot, len(storms)))
    num, den = s[idx].sum(1), c[idx].sum(1)
    ok = den > 0
    lo, hi = _ci(num[ok] / den[ok])
    return float(est), lo, hi, int(c.sum()), int((c > 0).sum())


def boot_diff(df, col, mask_a, mask_b, n_boot=2000, seed=0):
    """Hiệu trung bình (A − B) với bootstrap theo storm. p_boot là gần đúng (theo dấu)."""
    if col not in df.columns:
        return (float("nan"),) * 4 + (0, 0, 0, 0)
    storms = df["storm"].unique()
    sa, ca = _per_storm(df, col, mask_a, storms)
    sb, cb = _per_storm(df, col, mask_b, storms)
    if ca.sum() == 0 or cb.sum() == 0:
        return (float("nan"),) * 4 + (int(ca.sum()), int(cb.sum()), int((ca > 0).sum()), int((cb > 0).sum()))
    est = sa.sum() / ca.sum() - sb.sum() / cb.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(storms), size=(n_boot, len(storms)))
    na, da, nb, db = sa[idx].sum(1), ca[idx].sum(1), sb[idx].sum(1), cb[idx].sum(1)
    ok = (da > 0) & (db > 0)
    bs = na[ok] / da[ok] - nb[ok] / db[ok]
    lo, hi = _ci(bs)
    p = float(2 * min((bs <= 0).mean(), (bs >= 0).mean())) if bs.size else float("nan")
    return float(est), lo, hi, p, int(ca.sum()), int(cb.sum()), int((ca > 0).sum()), int((cb > 0).sum())


def storm_spearman(df, xcol, ycol, n_boot=1000, seed=0):
    from scipy.stats import spearmanr
    if xcol not in df.columns or ycol not in df.columns:
        return float("nan"), float("nan"), float("nan"), 0
    g = df.groupby("storm")[[xcol, ycol]].mean().dropna()
    if len(g) < 8:
        return float("nan"), float("nan"), float("nan"), len(g)
    rho = spearmanr(g[xcol], g[ycol]).correlation
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        s = g.iloc[rng.integers(0, len(g), len(g))]
        r = spearmanr(s[xcol], s[ycol]).correlation
        if np.isfinite(r):
            bs.append(r)
    lo, hi = _ci(bs)
    return float(rho), lo, hi, len(g)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", default="/kaggle/working/xai/report")
    p.add_argument("--recurve_deg", type=float, default=45.0, help="|GT quay hướng 24→72h| >= ngưỡng ⇒ 'đổi hướng'")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--enso_csv", default="")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out = []

    def P(*x):
        s = " ".join(str(i) for i in x); print(s); out.append(s)

    def F(est, lo, hi, nd=3):
        return f"{est:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"

    df = pd.read_csv(a.csv)
    df["has_era5"] = df["has_era5"].astype(bool)
    df["year"] = df["storm"].astype(str).str.split("_").str[0]
    d_era = df["has_era5"]
    P("=" * 70)
    P(f"Số cửa sổ: {len(df)} | số storm: {df['storm'].nunique()} | có ERA5: {int(d_era.sum())}")
    P("Cửa sổ/storm:", df.groupby("storm").size().describe().round(1).to_dict())
    P("LƯU Ý: khoảng tin cậy là bootstrap theo storm; ít storm ⇒ khoảng rộng, đó là thông tin trung thực.")

    # ── 1. phân phối clustering (để hiệu chỉnh ngưỡng) ──
    if "sil_raw" in df.columns:
        P("\n[1] Phân phối clustering ensemble (dùng để chọn --min_sil / --min_sep_km):")
        for c in ("sil_raw", "sep_km", "spread_72h_km"):
            if c in df.columns:
                q = df[c].dropna().quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_dict()
                P(f"   {c}: {q}")
        df["multi"] = (df["n_clusters"] >= 2).astype(float)
        est, lo, hi, n, ns = boot_mean(df, "multi", n_boot=a.n_boot)
        P(f"   Tỉ lệ cửa sổ có >= 2 kịch bản: {F(est, lo, hi)}  (n={n}, storm={ns})")

    # ── 2. |IG| theo tầng ──
    objs = sorted({c.split("_")[1] for c in df.columns if c.startswith("ig_") and "_frac_" in c})
    P("\n[2] Phần trăm |IG| theo tầng khí quyển (chỉ cửa sổ có ERA5):")
    for obj in objs:
        P(f"   Mục tiêu '{obj}':")
        for lv in ("200hPa", "500hPa", "850hPa", "925hPa", "SST"):
            est, lo, hi, n, ns = boot_mean(df, f"ig_{obj}_frac_{lv}", d_era, a.n_boot)
            P(f"      {lv:7s} {F(est, lo, hi)}  (n={n}, storm={ns})")
        sst = (df.loc[d_era, f"ig_{obj}_frac_SST"] > 0.10).mean() if f"ig_{obj}_frac_SST" in df else float("nan")
        P(f"      Tỉ lệ cửa sổ có SST > 10% |IG| (nghi tương quan giả): {sst:.1%}")

    # ── 3. đổi hướng vs không (A1 rút gọn) ──
    rec = d_era & (df["gt_turn_deg"].abs() >= a.recurve_deg)
    non = d_era & (df["gt_turn_deg"].abs() < a.recurve_deg)
    P(f"\n[3] Đổi hướng (|GT quay| >= {a.recurve_deg:.0f}°) vs không — A1 rút gọn")
    P(f"   Số cửa sổ: đổi hướng={int(rec.sum())} (storm={df.loc[rec, 'storm'].nunique()}), "
      f"không={int(non.sum())} (storm={df.loc[non, 'storm'].nunique()})")
    P("   Lưu ý: chưa kiểm soát mùa/cường độ; cần phân tích nhân tố nhiễu trước khi kết luận.")
    cols = [c for c in df.columns if c.startswith("ig_") and (c.endswith("ratio200_500_last") or "_frac_" in c
                                                                or c.endswith("anom500_sigma"))]
    cols += [c for c in ("spread_48h_km", "spread_72h_km", "multi") if c in df.columns]
    cols += [c for c in df.columns if c.startswith("var_ctx_frac")]
    for c in cols:
        est, lo, hi, pv, na, nb, sa, sb = boot_diff(df, c, rec, non, a.n_boot)
        P(f"   {c:36s} Δ(đổi hướng − không) = {F(est, lo, hi)}  p_boot≈{pv:.3f}")

    # ── 4. Fidelity ──
    fc = [c for c in df.columns if c.startswith("fid_")]
    if fc:
        P("\n[4] Fidelity (trung bình theo storm, bootstrap theo storm):")
        for c in fc:
            est, lo, hi, n, ns = boot_mean(df, c, n_boot=a.n_boot)
            P(f"   {c:24s} {F(est, lo, hi)}  (n={n}, storm={ns})")
        for own, base, name in [("fid_prec_own", "fid_prec_rand", "Precision: đề xuất − ngẫu nhiên"),
                                ("fid_prec_own", "fid_prec_sal", "Precision: đề xuất − saliency gradient"),
                                ("fid_rec_own", "fid_rec_rand", "Recall: đề xuất − ngẫu nhiên"),
                                ("fid_rec_own", "fid_rec_sal", "Recall: đề xuất − saliency gradient")]:
            if own in df.columns and base in df.columns:
                tmp = df.copy(); tmp["_d"] = tmp[own] - tmp[base]
                est, lo, hi, n, ns = boot_mean(tmp, "_d", n_boot=a.n_boot)
                P(f"   {name:42s} {F(est, lo, hi)}  → {'khác 0 rõ' if lo > 0 or hi < 0 else 'KHÔNG tách khỏi 0'}")
        m = {c: boot_mean(df, c, n_boot=a.n_boot)[0] for c in ("fid_prec_own", "fid_rec_own", "fid_unc_f")}
        vals = [v for v in m.values() if v == v and v > 0]
        if vals:
            P(f"   H-mean-F (điều hoà của P, R, Unc-F, tính từ các trung bình): {len(vals) / sum(1 / v for v in vals):.3f}")
        if "fid_k_before" in df.columns and "fid_k_after_top" in df.columns:
            sub = df["fid_k_before"] >= 2
            if sub.any():
                merged = (df.loc[sub, "fid_k_after_top"] < df.loc[sub, "fid_k_before"]).mean()
                P(f"   Trong cửa sổ có >=2 kịch bản: xoá top-IG làm giảm số kịch bản ở {merged:.1%} (chỉ mô tả)")

    # ── 5. theo tháng ──
    P("\n[5] Theo tháng (kịch bản và |IG| tầng 500/200 hPa, mục tiêu đầu tiên):")
    if "month" in df.columns and objs:
        o = objs[0]
        for mth in sorted(df["month"].dropna().unique()):
            mk = df["month"] == mth
            e5, _, _, n5, ns = boot_mean(df, f"ig_{o}_frac_500hPa", mk & d_era, 500)
            e2, _, _, _, _ = boot_mean(df, f"ig_{o}_frac_200hPa", mk & d_era, 500)
            P(f"   tháng {int(mth) + 1:2d}: cửa sổ={int(mk.sum()):5d} storm={df.loc[mk, 'storm'].nunique():3d} | "
              f"500hPa={e5:.2f}  200hPa={e2:.2f}")

    # ── 6. ENSO (tuỳ chọn) ──
    if a.enso_csv and os.path.exists(a.enso_csv):
        e = pd.read_csv(a.enso_csv)
        e["year"] = pd.to_numeric(e["year"], errors="coerce"); e["month"] = pd.to_numeric(e["month"], errors="coerce")
        df["year_n"] = pd.to_numeric(df["year"], errors="coerce"); df["month1"] = df["month"] + 1
        mg = df.merge(e.rename(columns={"year": "year_n", "month": "month1"})[["year_n", "month1", "oni"]],
                      on=["year_n", "month1"], how="left")
        mg["has_era5"] = mg["has_era5"].astype(bool)
        ln = mg["has_era5"] & (mg["oni"] <= -0.5); en = mg["has_era5"] & (mg["oni"] >= 0.5)
        P(f"\n[6] ENSO: La Niña={int(ln.sum())} cửa sổ ({mg.loc[ln, 'storm'].nunique()} storm), "
          f"El Niño={int(en.sum())} ({mg.loc[en, 'storm'].nunique()} storm)")
        for c in [c for c in mg.columns if c.startswith("ig_") and "_frac_" in c] + \
                 [c for c in ("spread_72h_km", "multi") if c in mg.columns]:
            est, lo, hi, pv, *_ = boot_diff(mg, c, ln, en, a.n_boot)
            P(f"   {c:36s} Δ(La Niña − El Niño) = {F(est, lo, hi)}  p_boot≈{pv:.3f}")

    # ── 7. thăm dò: spread vs độ mạnh dị thường GPH500 ──
    P("\n[7] THĂM DÒ — tương quan Spearman ở mức storm (không phải kiểm định giả thuyết đã định trước):")
    for o in objs:
        df[f"abs_anom500_{o}"] = df[f"ig_{o}_anom500_sigma"].abs()
        rho, lo, hi, n = storm_spearman(df[d_era], "spread_72h_km", f"abs_anom500_{o}")
        P(f"   spread_72h vs |dị thường GPH500| ({o}): rho={rho:.2f} [{lo:.2f}, {hi:.2f}]  (storm={n})")

    with open(os.path.join(a.out_dir, "analysis_report.txt"), "w") as f:
        f.write("\n".join(out))
    P(f"\nĐã lưu báo cáo → {os.path.join(a.out_dir, 'analysis_report.txt')}")


if __name__ == "__main__":
    main()