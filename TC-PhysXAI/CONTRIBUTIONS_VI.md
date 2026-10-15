# TC-PhysXAI v1.4.3 — Các Đóng Góp Khoa Học

**Tác giả:** Nguyễn Hồng Gam  
**Phiên bản:** 1.4.3  
**Cơ sở:** TC-XFlow (ESWA 2025) — mô hình dự báo đường đi bão bằng Flow Matching có điều kiện  
**Ngày:** Tháng 10/2026

---

## Tổng Quan

TC-PhysXAI là framework giải thích AI (XAI) được thiết kế đặc biệt cho mô hình dự báo bão nhiệt đới TC-XFlow. Thay vì dùng SHAP hay LIME (các phương pháp XAI tổng quát không hiểu vật lý khí tượng), TC-PhysXAI xây dựng 8 thành phần (C1–C8) khai thác cấu trúc riêng của mô hình Flow Matching và tích hợp kiến thức vật lý khí tượng.

---

## C1 — Phân Bổ Thời Gian (Temporal Attribution)

### Ý nghĩa
Trong số 8 cửa sổ quan sát 6h (từ t−48h đến t−0h), cửa sổ nào **quan trọng nhất** cho dự báo?

### Phương pháp
Phương pháp **perturbation-based**: lần lượt thay thế từng cửa sổ thời gian bằng giá trị trung bình không gian-thời gian của cơn bão đó (SPATIOTEMPORAL MEAN PERTURBATION), rồi đo độ thay đổi CRPS (Continuous Ranked Probability Score).

$$\Delta\text{CRPS}_t = \text{CRPS}(\hat{y} | x_{\setminus t}) - \text{CRPS}(\hat{y} | x)$$

Cửa sổ nào có $\Delta\text{CRPS}$ cao nhất → quan trọng nhất.

### Cơ sở khoa học
- **CRPS**: Matheson & Winkler (1976) — thước đo dự báo xác suất tiêu chuẩn  
- **Perturbation-based attribution**: Zeiler & Fergus (2014) — xóa vùng để đo tầm quan trọng
- **Baseline**: Trung bình không gian-thời gian của chính cơn bão (không phải trung bình khí hậu đa cơn bão)

### Đầu ra
- `delta_crps[T_obs]`: độ thay đổi CRPS cho từng cửa sổ
- `dominant_step`: cửa sổ quan trọng nhất
- Biểu đồ thanh (bar chart)

---

## C2 — Phân Bổ Không Gian (Spatial Attribution via Integrated Gradients)

### Ý nghĩa
Vùng không gian nào trên bản đồ ERA5 (13 kênh khí tượng) **ảnh hưởng nhiều nhất** đến dự báo?

### Phương pháp
**Integrated Gradients (IG)** — tích phân gradient từ baseline (zero) đến input thực:

$$A_i = (x_i - x_i^{\text{ref}}) \times \int_0^1 \frac{\partial F(\tilde{x}(\alpha))}{\partial \tilde{x}_i} d\alpha$$

với $\tilde{x}(\alpha) = x^{\text{ref}} + \alpha(x - x^{\text{ref}})$, xấp xỉ bằng tổng Riemann 30 bước.

### Cơ sở khoa học
- **Integrated Gradients**: Sundararajan, Taly & Yan (2017, ICML) — thỏa mãn tiên đề Sensitivity + Completeness  
- **Physical object detection**: Phát hiện vùng cao áp/thấp áp qua ngưỡng phần trăm attribution

### Đầu ra
- `channel_attribution[13]`: tầm quan trọng mỗi kênh ERA5
- `detected_objects`: danh sách vật thể khí tượng (gió đông cao, SST ấm, v.v.)

---

## C3 — Độ Trung Thực Xác Suất H-mean-F

### Ý nghĩa
Đo lường phương pháp XAI **tin cậy** đến mức nào — attribution có thực sự phản ánh hành vi mô hình không?

### Phương pháp
Kết hợp hài hòa 3 chỉ số:

$$H\text{-mean-F} = 0.4 \cdot P + 0.4 \cdot R + 0.2 \cdot U$$

- **P (Precision/Faithfulness)**: AOPC gap (MoRF − LeRF) — xóa kênh quan trọng trước phải làm dự báo xấu hơn xóa kênh ít quan trọng
- **R (Robustness)**: ổn định attribution khi thêm nhiễu nhỏ vào input
- **U (Uncertainty calibration)**: độ hiệu chỉnh spread dự báo ensemble

### Cơ sở khoa học
- **AOPC/MoRF/LeRF**: Samek et al. (2017, IEEE TNNLS) — chuẩn đánh giá faithfulness  
- **H-mean-F**: Mở rộng từ F1-score đa chiều của Rijsbergen (1979)

---

## C4 — Phân Bổ Tích Phân ODE (FM Integration-Path Attribution)

### Ý nghĩa
Dọc theo đường giải ODE từ nhiễu Gaussian $z_0$ đến quỹ đạo dự báo $z_1$, **thời điểm nào** và **kênh ERA5 nào** ảnh hưởng nhiều nhất đến trường vận tốc FM?

### Phương pháp
Tích phân attribution dọc path ODE ($t \in [0,1]$):

$$A_j = \int_0^1 \left|\frac{\partial v_\theta(z_t, c, t)}{\partial c_j}\right| dt$$

xấp xỉ bằng **DDIM** 20 bước với Jacobian vector product (JVP).

### Cơ sở khoa học
- **Flow Matching**: Lipman et al. (2022, ICLR 2023) — ODE: $\dot{z}_t = v_\theta(z_t, c, t)$
- **FreeFM principle**: Phân bổ "free" qua path ODE thay vì chỉ tại đầu ra
- **DDIM**: Song et al. (2021) — discretization hiệu quả

### Đầu ra
- `ode_time_attribution[n_steps]`: attribution theo thời gian ODE  
- `channel_attribution[13]`: tầm quan trọng kênh ERA5 dọc path

---

## C5 — Phân Rã Phương Sai (Noise-Context Variance Decomposition)

### Ý nghĩa
Sự không chắc chắn trong dự báo đến từ đâu — **nhiễu ngẫu nhiên** ($z_0$) hay **mơ hồ về điều kiện khí tượng** (context $c$)?

### Phương pháp
Áp dụng **Luật Phương Sai Tổng** (Law of Total Variance):

$$V_{\text{total}} = \underbrace{\mathbb{E}_c[V(y|c, z_0)]}_{\text{nhiễu ngẫu nhiên}} + \underbrace{V_c[\mathbb{E}(y|c, z_0)]}_{\text{mơ hồ context}}$$

- **V_noise**: phương sai trên K mẫu $z_0$ với context cố định
- **V_context**: phương sai khi pertub nhẹ context (10 biến thể nhiễu nhỏ)

### Cơ sở khoa học
- **Law of Total Variance**: xác suất cổ điển (Eve's Law)  
- **Ứng dụng AI**: Kendall & Gal (2017, NeurIPS) — tách uncertainty epistemic/aleatoric

### Đầu ra
- `frac_noise` / `frac_context`: tỷ lệ phương sai từ mỗi nguồn  
- `predictability_horizon`: bước dự báo mà V_context vượt V_noise (mất dự báo được)

---

## C6 — Phân Tích Phản Thực (FM Counterfactual via ODE Adjoint)

### Ý nghĩa
**Nếu** bão này có khả năng tái cong (recurvature), cần **thay đổi điều kiện khí tượng như thế nào** (phần nhỏ nhất có thể)?

### Phương pháp
Bài toán tối ưu trên không gian context:

$$\min_{\Delta c} \|\Delta c\|^2 \quad \text{s.t.} \quad \Phi(c + \Delta c) \in \mathcal{T}_{\text{target}}$$

Giải bằng **gradient descent có hình chiếu** (projected gradient) 80 bước trên không gian context FM.

### Cơ sở khoa học
- **ODE Adjoint**: Chen et al. (2018, NeurIPS) — gradient qua ODE không cần lưu trajectory  
- **Counterfactual XAI**: Wachter, Mittelstadt & Russell (2017, Harvard JOLT)  
- **Projected gradient**: Madry et al. (2018) — đảm bảo constraint

### Đầu ra
- `counterfactual_track`: quỹ đạo bão dưới điều kiện phản thực  
- `channel_delta[13]`: thay đổi cần thiết ở mỗi kênh ERA5  
- `context_change_norm`: cường độ can thiệp tổng thể

---

## C7 — XAI Phân Đoạn Thích Nghi (Segment-Level XAI + AOPC Faithfulness)

### Ý nghĩa
Quan sát bão được chia thành 3 **giai đoạn vật lý**:
- **Pha A** (t−48h → t−24h): Tiền thân từ xa (Remote Precursor) — ridge Thái Bình Dương, tương tác monsoon
- **Pha B** (t−24h → t−12h): Tiến hóa synoptic — nhiễu loạn, đường trough
- **Pha C** (t−12h → t−0h): Tác động tức thời — steering flow trực tiếp

Giai đoạn nào quan trọng hơn? Attribution đó có **đáng tin (faithful)** không?

### Phương pháp

**Phần 1 — Fixed Segment Attribution:**  
Mask từng pha, đo $\Delta\text{CRPS}$ tương tự C1 nhưng ở cấp độ pha.

**Phần 2 — Adaptive Phase Discovery:**  
Tìm điểm phân chia tối ưu $\tau^*$ bằng grid search:

$$\tau^* = \arg\max_\tau \Delta\text{CRPS}(\tau) \quad \tau \in \{t_1, ..., t_{T-1}\}$$

**Phần 3 — AOPC Faithfulness (MoRF/LeRF):**  
Đánh giá độ tin cậy attribution theo Samek et al. (2017):

$$\text{AOPC}_{\text{MoRF}} = \frac{1}{L}\sum_{k=1}^{L}[\text{CRPS}_{\text{full}} - \text{CRPS}_{1..k}^{\text{MoRF}}]$$

$$\text{Faithfulness} = \text{AOPC}_{\text{MoRF}} - \text{AOPC}_{\text{LeRF}} > 0$$

MoRF (Most Relevant First) xóa pha quan trọng trước → CRPS tăng nhanh → faithful.  
LeRF (Least Relevant First) xóa pha ít quan trọng trước → CRPS tăng chậm.

**Phần 4 — Global Attribution Mining:**  
Tập hợp attribution qua nhiều cơn bão để tìm pattern toàn cục (e.g., pha A quan trọng hơn cho bão tái cong).

### Cơ sở khoa học
- **AOPC/MoRF/LeRF**: Samek, Binder, et al. (2017, IEEE TNNLS) — tiêu chuẩn vàng đánh giá XAI  
- **XAI4Traj**: Nguyen et al. (2025, Neurocomputing) — XAI cho quỹ đạo, cảm hứng phân đoạn vật lý
- **Adaptive segmentation**: Tìm điểm gãy tối ưu bằng grid search trên không gian liên tục

### Đầu ra
- `phase_results`: ΔCRPS của mỗi pha A/B/C  
- `aopc_faithfulness`: giá trị faithfulness (> 0 = đáng tin)  
- `dominant_phase`: pha quan trọng nhất  
- `tau_star`: điểm phân chia tối ưu thích nghi

---

## C8 — Phân Bổ Đặc Trưng Quỹ Đạo (Track Feature Attribution)

### Ý nghĩa
Đây là đóng góp mới nhất và phức tạp nhất. C8 trả lời câu hỏi:

> *Tại sao mô hình dự báo cơn bão này **chuyển hướng** / **tăng tốc** / **có khả năng tái cong cao**?*

Thay vì phân bổ trực tiếp ảnh hưởng lên dự báo tọa độ (lat/lon), C8 phân bổ ảnh hưởng lên **các đặc trưng vật lý của quỹ đạo dự báo**:

| Đặc trưng | Ý nghĩa vật lý |
|-----------|----------------|
| `speed` | Tốc độ tịnh tiến trung bình (km/6h) |
| `curvature` | Độ cong quỹ đạo (rad/km) — bão có lượn không? |
| `recurvature_prob` | Xác suất tái cong (chuyển sang hướng Đông-Bắc) |
| `heading_change` | Góc thay đổi hướng di chuyển (°) |
| `displacement` | Tổng quãng đường dự báo (km) |
| `spread` | Độ phân tán ensemble (km) — uncertainty |
| `bifurcation_idx` | Chỉ số phân kỳ ensemble — có 2 kịch bản rõ ràng không? |

### Phương pháp

**Bước 1 — Trích xuất đặc trưng:**  
Từ K ensemble trajectories $\{y^{(k)}\}_{k=1}^K$, tính 7 đặc trưng vật lý.

*Tốc độ:*
$$\text{speed} = \frac{1}{T-1}\sum_{t=1}^{T-1} \|y_t - y_{t-1}\|_2 \cdot s_{\text{km}}$$

*Độ cong (bình phương uốn cong):*
$$\kappa_t = \frac{|v_t \times a_t|}{|v_t|^3 + \epsilon}, \quad v_t = y_t - y_{t-1}, \quad a_t = v_t - v_{t-1}$$

*Góc circular mean* (theo Mardia & Jupp 2000):
$$\bar{\theta} = \text{atan2}\!\left(\frac{1}{K}\sum_k \sin\theta_k,\ \frac{1}{K}\sum_k \cos\theta_k\right)$$

với convention: $\theta = \text{atan2}(\Delta\text{lon}, \Delta\text{lat})$ → Bắc=0°, Đông=90°.

*Chỉ số tái cong:* Phần trăm members có góc cuối > 270° (hướng NE).

*Chỉ số phân kỳ η² (đã hiệu chỉnh null Gaussian):*

Phân tích thành phần chính (PCA) trên vị trí cuối $\{y_T^{(k)}\}$:
$$\eta^2_{\text{raw}} = \frac{B}{B+W} = \frac{\text{between-group variance}}{\text{total variance on PC1}}$$

Tuy nhiên, với phân phối đơn mode $\mathcal{N}(0,\sigma^2)$ chia đôi qua median, $\eta^2_{\text{raw}} \approx 0.637$ ngay cả khi **không có bifurcation thực sự** (vì $\mathbb{E}[x|x\geq 0] = \sigma\sqrt{2/\pi} \approx 0.798\sigma$ — trung bình half-normal). Do đó phải hiệu chỉnh:

$$\eta^2_{\text{corrected}} = \text{clip}\!\left(\frac{\eta^2_{\text{raw}} - 0.637}{1.0 - 0.637},\ 0,\ 1\right)$$

Ngưỡng: $\eta^2_{\text{corrected}} > 0.30$ → "BIMODAL ENSEMBLE".

**Bước 2 — Channel-Zeroing Attribution:**  
Lần lượt zero-out từng kênh ERA5 (trong toàn bộ cửa sổ quan sát), tái dự báo, đo thay đổi đặc trưng:

$$A_{f,c} = f_{\text{feat}}(\hat{y}|\text{baseline}) - f_{\text{feat}}(\hat{y}|x_{\setminus c})$$

cho $f \in \{1..7\}$ đặc trưng, $c \in \{1..13\}$ kênh → ma trận attribution $A \in \mathbb{R}^{7 \times 13}$.

**Bước 3 — Window Attribution:**  
Tương tự nhưng mask theo cửa sổ thời gian (không phải kênh):

$$A_{f,t} = f_{\text{feat}}(\hat{y}|\text{baseline}) - f_{\text{feat}}(\hat{y}|x_{\setminus t})$$

→ ma trận $A_w \in \mathbb{R}^{7 \times T_{\text{obs}}}$.

**Bước 4 — Bootstrap Confidence Intervals (đã sửa):**

Mỗi lần bootstrap $b = 1..B$:
1. Lấy mẫu lại K ensemble **MỚI** để tính baseline $f_{\text{feat}}^{(b)}$ (không dùng baseline cố định)
2. Lấy mẫu lại K mới để tính $f_{\text{feat, perturbed}}^{(b)}$ với kênh c bị zero

$$A_{f,c}^{(b)} = f_{\text{feat}}^{(b)} - f_{\text{feat, perturbed}}^{(b)}$$

CI ở mức $\alpha = 5\%$: $[\text{P}_{2.5}(A^{(b)}),\ \text{P}_{97.5}(A^{(b)})]$.

Ô không chứa 0 → **significant** (p < 0.05).

*Lý do phải resample baseline:* Nếu baseline cố định, ta chỉ bootstrap phần perturbation → đánh giá thấp phương sai thực của attribution statistic (Efron 1979). Resample cả hai đảm bảo CI đúng về mặt lý thuyết.

**Bước 5 — MoRF-AOPC Faithfulness:**  
Sắp xếp kênh theo |attribution| giảm dần, zero-out lần lượt:

$$\text{AOPC} = \frac{1}{L}\sum_{k=1}^{L} \frac{|f_{\text{feat}}(\hat{y}|\text{baseline}) - f_{\text{feat}}(\hat{y}|x_{\setminus\{1..k\}}^{\text{MoRF}})|}{|\text{baseline}|+ 0.01}$$

AOPC > 0 và tăng đơn điệu → attribution faithful.

### Cơ sở khoa học
- **Channel-zeroing**: Zeiler & Fergus (2014, ECCV) — occlusion sensitivity  
- **η² bifurcation correction**: Bình thường hóa null Gaussian từ half-normal distribution theory (David & Nagaraja 2003, §2.1) — $\mathbb{E}[|X|] = \sigma\sqrt{2/\pi}$ với $X \sim \mathcal{N}(0,\sigma^2)$  
- **Circular mean**: Mardia & Jupp (2000, §2.2) — thống kê góc  
- **Bootstrap CI**: Efron & Tibshirani (1993) — cần resample cả hai phần của statistic  
- **MoRF-AOPC**: Samek et al. (2017, IEEE TNNLS) — faithfulness cho regression

### Đầu ra
- `attribution_matrix[7, 13]`: ma trận attribution đặc trưng × kênh  
- `window_attribution[7, T_obs]`: attribution theo cửa sổ thời gian  
- `significance_mask[7, 13]`: boolean — có ý nghĩa thống kê hay không  
- `aopc_faithfulness`: độ tin cậy attribution  
- `physical_interpretation`: mô tả ngôn ngữ tự nhiên (tiếng Anh)

---

## K-Sensitivity Ablation (Thực nghiệm E8 mở rộng)

### Ý nghĩa
Kết quả C8 có ổn định khi thay đổi kích thước ensemble K (số mẫu dự báo)?

### Phương pháp
Chạy C8 với K ∈ {5, 10, 20}. Với mỗi cặp (K, K_ref=20), tính **Spearman ρ** giữa ranking attribution (tương quan hạng):

$$\rho_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}, \quad d_i = r_i^{(K)} - r_i^{(K_{\text{ref}})}$$

- ρ ≥ 0.85 → **STABLE** (K đủ lớn)  
- 0.70 ≤ ρ < 0.85 → **MARGINAL**  
- ρ < 0.70 → **UNSTABLE** (nên tăng K)

### Cơ sở khoa học
- **Spearman correlation**: Spearman (1904) — đo tương quan thứ hạng không tham số  
- **Ổn định ensmeble XAI**: Tương tự bootstrap stability tests trong Stability of Feature Selection (Nogueira et al. 2018)

---

## Thực Nghiệm E1–E8

| Thực nghiệm | Đánh giá gì? | Metric chính |
|-------------|-------------|--------------|
| **E1** | So sánh TC-PhysXAI với Gradient / LIME / SHAP | H-mean-F, thời gian tính |
| **E2** | Attribution tái cong (recurvature) theo thời gian và không gian | ΔCRPS, IG spatial maps |
| **E3** | Attribution tăng cường độ nhanh (Rapid Intensification) + SST | ΔAtt_SST, RI hit rate |
| **E4** | Tính nhất quán đa module (C1/C2/C4 có đồng thuận không?) | Rank correlation |
| **E5** | Benchmard H-mean-F fidelity (ablation theo từng thành phần) | H-mean-F per component |
| **E6** | Phân tích kịch bản phản thực (C6 adjoint) | Context change norm |
| **E7** | Adaptive Segment-Level XAI mở rộng (C7): fixed A/B/C + τ* + global mining | AOPC faithfulness |
| **E8** | Track Feature Attribution (C8): feature × channel/window + K-sensitivity | Spearman ρ, AOPC |

---

## Tóm Tắt Cải Tiến Khoa Học v1.4.x

### v1.4.0 → v1.4.1: Thêm C7 (Segment XAI)
- Phân đoạn vật lý 3 pha A/B/C dựa trên cấu trúc khí tượng học  
- AOPC MoRF/LeRF faithfulness test  
- Adaptive phase discovery (tìm τ* tối ưu)  
- Global attribution mining (pattern qua nhiều bão)

### v1.4.1 → v1.4.2: Thêm C8 (Track Feature Attribution)
- Trích xuất 7 đặc trưng vật lý từ ensemble trajectory  
- Ma trận attribution 7×13 (đặc trưng × kênh ERA5)  
- Attribution theo cửa sổ thời gian 7×8  
- Bootstrap CI với resample đúng chuẩn

### v1.4.2 → v1.4.3: Sửa Lỗi Khoa Học Nghiêm Trọng

#### Lỗi 1 (CRITICAL): η² bifurcation index — sai diễn giải null
- **Vấn đề**: Với dữ liệu đơn mode $\mathcal{N}(0,\sigma^2)$, η²_raw ≈ 0.637 (không phải 0!). Ngưỡng cũ 0.25 nằm **dưới** giá trị null → mọi cơn bão đều bị gán nhãn "BIMODAL" sai.
- **Sửa**: Hiệu chỉnh null: `η²_corr = clip((η²_raw − 0.637) / 0.363, 0, 1)`. Ngưỡng mới: 0.30.
- **Nguồn**: David & Nagaraja (2003) §2.1 — half-normal mean $\mathbb{E}[|X|] = \sigma\sqrt{2/\pi}$

#### Lỗi 2 (SIGNIFICANT): Bootstrap CI — baseline không được resample
- **Vấn đề**: Baseline `base_vec` được tính **1 lần** trước vòng lặp bootstrap → chỉ resample phần perturbation → đánh giá thấp phương sai → CI quá hẹp → significant giả.
- **Sửa**: Tính lại `base_vec_b` trong mỗi lần lặp b. CI bây giờ bao gồm variance từ cả baseline và perturbation.
- **Nguồn**: Efron & Tibshirani (1993, §6.1) — bootstrap của statistic phức hợp

#### Lỗi 3 (MINOR): AOPC denominator gây overflow
- **Vấn đề**: `denom = |f| + 1e-6` → khi `recurvature_prob ≈ 0`, AOPC → ∞
- **Sửa**: `denom = max(|f|, 0.01)` — clamp nhỏ nhưng đủ tránh overflow

#### Lỗi 4 (MINOR): scipy không có fallback khi Kaggle thiếu
- **Vấn đề**: `from scipy.stats import spearmanr` crash nếu scipy chưa cài
- **Sửa**: try/except với fallback numpy-only Spearman ρ thuần

---

## Yêu Cầu Môi Trường

```
Python   ≥ 3.9
PyTorch  ≥ 2.0
numpy    ≥ 1.24
matplotlib ≥ 3.7
pandas   ≥ 2.0
scipy    ≥ 1.10   (optional — có fallback nếu thiếu)
```

**Phần cứng khuyến nghị:**
- GPU: NVIDIA T4 hoặc P100 (Kaggle)
- RAM: ≥ 16 GB
- Thời gian chạy toàn bộ E1–E8: ~60–90 phút trên T4

---

## Tài Liệu Tham Khảo

1. Lipman et al. (2022/2023, ICLR) — Flow Matching for Generative Modeling  
2. Sundararajan, Taly & Yan (2017, ICML) — Axiomatic Attribution for Deep Networks (IG)  
3. Samek, Binder, et al. (2017, IEEE TNNLS) — Evaluating the Visualization of What a DNN has Learned (AOPC/MoRF)  
4. Chen et al. (2018, NeurIPS) — Neural Ordinary Differential Equations (Adjoint)  
5. Kendall & Gal (2017, NeurIPS) — What Uncertainties Do We Need in Bayesian Deep Learning?  
6. Wachter, Mittelstadt & Russell (2017, Harvard JOLT) — Counterfactual Explanations Without Opening the Black Box  
7. Mardia & Jupp (2000) — Directional Statistics §2.2 (circular mean)  
8. Efron & Tibshirani (1993) — An Introduction to the Bootstrap §6.1  
9. David & Nagaraja (2003) — Order Statistics §2.1 (half-normal mean)  
10. Matheson & Winkler (1976) — Scoring Rules for Continuous Probability Distributions (CRPS)  
11. Nguyen et al. (2025, Neurocomputing) — XAI4Traj: Towards Explainable Trajectory Classification  
12. Zeiler & Fergus (2014, ECCV) — Visualizing and Understanding CNNs (occlusion sensitivity)
