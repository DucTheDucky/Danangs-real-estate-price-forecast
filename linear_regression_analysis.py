"""
Computer Project MAI391: Linear Regression Analysis for Real Estate in Da Nang
Author: MAI391 Project Team
Task: Predict and analyze property prices by district and property type based on area.
Upgraded: Stratified Models (Apartment vs Landed) + Interaction Terms (Area x District, Area x Property_Type).
Evaluates: Train/Test errors (MSE, RMSE, MAE, MAPE, R2), 5-Fold Cross Validation, and Runtime.
"""

import sys
import os
import csv
import math
import time
import random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

random.seed(42)

def percentile(data, p):
    """Compute percentile using linear interpolation."""
    s = sorted(data)
    idx = (len(s) - 1) * p
    low = int(idx)
    high = math.ceil(idx)
    weight = idx - low
    return s[low] * (1 - weight) + s[high] * weight

def median(data):
    s = sorted(data)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

def train_test_split(X, y, test_size=0.2, seed=42):
    rnd = random.Random(seed)
    indices = list(range(len(X)))
    rnd.shuffle(indices)
    split_point = int(len(X) * (1 - test_size))
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]
    return [X[i] for i in train_idx], [X[i] for i in test_idx], [y[i] for i in train_idx], [y[i] for i in test_idx]

def fit_simple_linear_regression(x_train, y_train):
    n = len(x_train)
    if n < 2:
        return 0.0, 0.0
    mx = sum(x_train) / n
    my = sum(y_train) / n
    ssxx = sum((xi - mx) ** 2 for xi in x_train)
    ssxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x_train, y_train))
    if ssxx == 0:
        return 0.0, my
    slope = ssxy / ssxx
    intercept = my - slope * mx
    return slope, intercept

def predict_simple(x, slope, intercept):
    return [intercept + slope * xi for xi in x]

def compute_metrics(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    my = sum(y_true) / n
    ss_tot = sum((yi - my) ** 2 for yi in y_true)
    ss_res = sum((yi - yhat) ** 2 for yi, yhat in zip(y_true, y_pred))
    mse = ss_res / n
    rmse = math.sqrt(mse)
    mae = sum(abs(yi - yhat) for yi, yhat in zip(y_true, y_pred)) / n
    mape = sum(abs(yi - yhat) / yi for yi, yhat in zip(y_true, y_pred) if yi > 0) / n * 100.0
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return mse, rmse, mae, mape, r2

def cross_val_r2(x, y, k=5, seed=42):
    n = len(x)
    if n < k or k < 2:
        return 0.0
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    fold_size = n // k
    r2_scores = []
    for i in range(k):
        val_idx = set(indices[i * fold_size : (i + 1) * fold_size] if i < k - 1 else indices[i * fold_size :])
        train_idx = [idx for idx in indices if idx not in val_idx]
        slope, intercept = fit_simple_linear_regression([x[idx] for idx in train_idx], [y[idx] for idx in train_idx])
        y_val_pred = predict_simple([x[idx] for idx in val_idx], slope, intercept)
        _, _, _, _, r2 = compute_metrics([y[idx] for idx in val_idx], y_val_pred)
        r2_scores.append(r2)
    return sum(r2_scores) / len(r2_scores)

# Matrix operations for Multiple Linear Regression
def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def matmul(A, B):
    p, q, r = len(A), len(A[0]), len(B[0])
    res = [[0.0] * r for _ in range(p)]
    for i in range(p):
        for k in range(q):
            aik = A[i][k]
            for j in range(r):
                res[i][j] += aik * B[k][j]
    return res

def matvecmul(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def invert_matrix(A):
    n = len(A)
    M = [[float(A[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        pivot_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[pivot_row] = M[pivot_row], M[i]
        pivot = M[i][i] if abs(M[i][i]) > 1e-12 else 1e-6
        for j in range(2 * n):
            M[i][j] /= pivot
        for r in range(n):
            if r != i:
                factor = M[r][i]
                for j in range(2 * n):
                    M[r][j] -= factor * M[i][j]
    return [[M[i][j + n] for j in range(n)] for i in range(n)]

def fit_multiple_linear_regression(X_train, y_train, ridge=1e-2):
    Xt = transpose(X_train)
    XtX = matmul(Xt, X_train)
    for i in range(1, len(XtX)):
        XtX[i][i] += ridge
    invXtX = invert_matrix(XtX)
    Xty = matvecmul(Xt, y_train)
    return matvecmul(invXtX, Xty)

def predict_multiple(X, weights):
    return [sum(X[i][j] * weights[j] for j in range(len(weights))) for i in range(len(X))]

def cross_val_r2_multiple(X, y, k=5, seed=42, ridge=1e-2):
    n = len(X)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    fold_size = n // k
    r2_scores = []
    for i in range(k):
        val_idx = set(indices[i * fold_size : (i + 1) * fold_size] if i < k - 1 else indices[i * fold_size :])
        train_idx = [idx for idx in indices if idx not in val_idx]
        w = fit_multiple_linear_regression([X[idx] for idx in train_idx], [y[idx] for idx in train_idx], ridge=ridge)
        preds = predict_multiple([X[idx] for idx in val_idx], w)
        _, _, _, _, r2 = compute_metrics([y[idx] for idx in val_idx], preds)
        r2_scores.append(r2)
    return sum(r2_scores) / len(r2_scores)

def filter_iqr(subset, mult=1.5):
    p = [float(r['Price_VND']) / 1e9 for r in subset]
    a = [float(r['Area']) for r in subset]
    q1_p, q3_p = percentile(p, 0.25), percentile(p, 0.75)
    iqr_p = q3_p - q1_p
    q1_a, q3_a = percentile(a, 0.25), percentile(a, 0.75)
    iqr_a = q3_a - q1_a
    return [r for r in subset if (q1_p - mult * iqr_p <= float(r['Price_VND']) / 1e9 <= q3_p + mult * iqr_p) and
                                 (q1_a - mult * iqr_a <= float(r['Area']) <= q3_a + mult * iqr_a)]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, 'ADY_Project_properties_cleaned_MissValuesRemoved.csv'),
        os.path.join(script_dir, '..', 'ADY_Project_properties_cleaned_MissValuesRemoved.csv'),
        os.path.join(os.getcwd(), 'ADY_Project_properties_cleaned_MissValuesRemoved.csv'),
    ]
    csv_file = next((p for p in candidates if os.path.exists(p)), None)
    if not csv_file:
        raise FileNotFoundError("Không tìm thấy 'ADY_Project_properties_cleaned_MissValuesRemoved.csv'.")

    print(f"Đang đọc dữ liệu từ: {csv_file}")
    with open(csv_file, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # 1. Filter Da Nang
    dn_rows = [r for r in rows if r['Province'].strip() == 'Đà Nẵng']

    # 2. Deduplicate titles & filter unit-entry errors (Price >= 100M VND, Price/m2 >= 1M VND)
    seen = set()
    cleaned_rows = []
    unit_errors = 0
    for r in dn_rows:
        if r['Title'] not in seen:
            seen.add(r['Title'])
            if float(r['Price_VND']) >= 1e8 and float(r['Price_Per_M2']) >= 1e6:
                cleaned_rows.append(r)
            else:
                unit_errors += 1

    print(f"Tổng số bản ghi ban đầu: {len(rows)}")
    print(f"Bản ghi tại Đà Nẵng: {len(dn_rows)}")
    print(f"Loại bỏ tin trùng lặp & {unit_errors} lỗi đơn vị nhập liệu -> Còn lại: {len(cleaned_rows)}")

    # =========================================================================
    # PART A: Linear Regression by District (Simple OLS)
    # =========================================================================
    districts = [
        'Quận Sơn Trà', 'Quận Hải Châu', 'Quận Thanh Khê',
        'Quận Ngũ Hành Sơn', 'Quận Cẩm Lệ', 'Quận Liên Chiểu', 'Huyện Hòa Vang'
    ]
    district_results = []
    print("\n" + "=" * 80)
    print("PHẦN A: LINEAR REGRESSION ĐƠN BIẾN THEO TỪNG QUẬN/HUYỆN")
    print("=" * 80)

    for d in districts:
        raw_d = [r for r in cleaned_rows if r['District'].strip() == d]
        inliers_d = filter_iqr(raw_d, mult=1.4)

        x_clean = [float(r['Area']) for r in inliers_d]
        y_clean = [float(r['Price_VND']) / 1e9 for r in inliers_d]
        x_train, x_test, y_train, y_test = train_test_split(x_clean, y_clean, test_size=0.2, seed=42)

        t0 = time.perf_counter()
        slope, intercept = fit_simple_linear_regression(x_train, y_train)
        train_time_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        y_train_pred = predict_simple(x_train, slope, intercept)
        y_test_pred = predict_simple(x_test, slope, intercept)
        test_time_ms = (time.perf_counter() - t0) * 1000.0

        _, tr_rmse, tr_mae, _, tr_r2 = compute_metrics(y_train, y_train_pred)
        _, te_rmse, te_mae, _, te_r2 = compute_metrics(y_test, y_test_pred)

        slope_full, intercept_full = fit_simple_linear_regression(x_clean, y_clean)
        _, full_rmse, _, _, full_r2 = compute_metrics(y_clean, predict_simple(x_clean, slope_full, intercept_full))
        cv5_r2 = cross_val_r2(x_clean, y_clean, k=5, seed=42)

        district_results.append({
            'district': d, 'n_raw': len(raw_d), 'n_clean': len(inliers_d),
            'slope_ty_per_m2': slope_full, 'intercept_ty': intercept_full,
            'r2_full': full_r2, 'rmse_full': full_rmse, 'r2_cv5': cv5_r2,
            'train_r2': tr_r2, 'train_rmse': tr_rmse, 'train_mae': tr_mae,
            'test_r2': te_r2, 'test_rmse': te_rmse, 'test_mae': te_mae,
            'train_time_ms': train_time_ms, 'test_time_ms': test_time_ms,
            'area_med': median(x_clean), 'price_med': median(y_clean),
            'price_per_m2_trieu': (sum(y_clean) / sum(x_clean)) * 1000.0
        })
        print(f"[{d:18s}] N={len(inliers_d):3d} | Slope={slope_full:.4f}, Intercept={intercept_full:+.4f} | Train R2={tr_r2:.4f}, Test R2={te_r2:.4f}, CV5={cv5_r2:.4f}")

    # =========================================================================
    # PART B: STRATIFIED MODELS + INTERACTION TERMS
    # Tách riêng 2 nhóm: (1) Căn hộ/Condotel và (2) BĐS Thổ cư
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHẦN B: MÔ HÌNH PHÂN TÁCH 2 NHÓM + BIẾN TƯƠNG TÁC (STRATIFIED + INTERACTION)")
    print("=" * 80)

    # --- GROUP 1: CĂN HỘ / CONDOTEL ---
    # Reference District: Quận Sơn Trà (mẫu lớn nhất: 146)
    apt_raw = [r for r in cleaned_rows if r['Property_Type'].strip() == 'Căn hộ/Condotel']
    apt_clean = filter_iqr(apt_raw, mult=1.5)
    apt_dists = ['Quận Ngũ Hành Sơn', 'Quận Hải Châu', 'Quận Thanh Khê', 'Quận Cẩm Lệ', 'Quận Liên Chiểu']

    X_apt, y_apt = [], []
    for r in apt_clean:
        a = float(r['Area'])
        p = float(r['Price_VND']) / 1e9
        d_vec = [1.0 if r['District'].strip() == d else 0.0 for d in apt_dists]
        inter_d = [a * v for v in d_vec]
        X_apt.append([1.0, a] + d_vec + inter_d)
        y_apt.append(p)

    X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(X_apt, y_apt, test_size=0.2, seed=42)

    t0 = time.perf_counter()
    w_apt = fit_multiple_linear_regression(X_tr_a, y_tr_a, ridge=1e-2)
    apt_train_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    y_tr_pred_a = predict_multiple(X_tr_a, w_apt)
    y_te_pred_a = predict_multiple(X_te_a, w_apt)
    apt_test_ms = (time.perf_counter() - t0) * 1000.0

    _, a_tr_rmse, a_tr_mae, a_tr_mape, a_tr_r2 = compute_metrics(y_tr_a, y_tr_pred_a)
    _, a_te_rmse, a_te_mae, a_te_mape, a_te_r2 = compute_metrics(y_te_a, y_te_pred_a)
    a_cv5 = cross_val_r2_multiple(X_apt, y_apt, k=5, seed=42, ridge=1e-2)

    print(f"\n[NHÓM 1: CĂN HỘ / CONDOTEL] (N_raw={len(apt_raw)}, N_clean={len(apt_clean)}, dim={len(X_apt[0])})")
    print(f"  Train: R2 = {a_tr_r2:.4f}, RMSE = {a_tr_rmse:.4f} tỷ, MAE = {a_tr_mae:.4f} tỷ, MAPE = {a_tr_mape:.1f}%")
    print(f"  Test : R2 = {a_te_r2:.4f}, RMSE = {a_te_rmse:.4f} tỷ, MAE = {a_te_mae:.4f} tỷ, MAPE = {a_te_mape:.1f}%")
    print(f"  5-Fold CV R2 = {a_cv5:.4f} | Thời gian: Train = {apt_train_ms:.4f} ms, Inference = {apt_test_ms:.4f} ms")

    apt_names = ['Bias (Mốc: Sơn Trà)', 'Area (Độ dốc Sơn Trà)'] + [f'D_{d}' for d in apt_dists] + [f'Area × {d}' for d in apt_dists]
    print("  Trọng số mô hình Căn hộ:")
    for name, w in zip(apt_names, w_apt):
        print(f"    {name:32s}: {w:+.4f}")

    # --- GROUP 2: BẤT ĐỘNG SẢN THỔ CƯ (Đất nền, Nhà phố, Nhà riêng, Biệt thự) ---
    # Reference District: Quận Sơn Trà, Reference Property Type: Đất nền
    land_raw = [r for r in cleaned_rows if r['Property_Type'].strip() != 'Căn hộ/Condotel']
    land_clean = filter_iqr(land_raw, mult=1.5)
    land_dists = ['Quận Hải Châu', 'Quận Thanh Khê', 'Quận Ngũ Hành Sơn', 'Quận Cẩm Lệ', 'Quận Liên Chiểu', 'Huyện Hòa Vang']
    land_pts = ['Nhà phố', 'Nhà riêng', 'Biệt thự']

    X_land, y_land = [], []
    for r in land_clean:
        a = float(r['Area'])
        p = float(r['Price_VND']) / 1e9
        d_vec = [1.0 if r['District'].strip() == d else 0.0 for d in land_dists]
        pt_vec = [1.0 if r['Property_Type'].strip() == pt else 0.0 for pt in land_pts]
        inter_d = [a * v for v in d_vec]
        inter_pt = [a * v for v in pt_vec]
        X_land.append([1.0, a] + d_vec + pt_vec + inter_d + inter_pt)
        y_land.append(p)

    X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(X_land, y_land, test_size=0.2, seed=42)

    t0 = time.perf_counter()
    w_land = fit_multiple_linear_regression(X_tr_l, y_tr_l, ridge=1e-2)
    land_train_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    y_tr_pred_l = predict_multiple(X_tr_l, w_land)
    y_te_pred_l = predict_multiple(X_te_l, w_land)
    land_test_ms = (time.perf_counter() - t0) * 1000.0

    _, l_tr_rmse, l_tr_mae, l_tr_mape, l_tr_r2 = compute_metrics(y_tr_l, y_tr_pred_l)
    _, l_te_rmse, l_te_mae, l_te_mape, l_te_r2 = compute_metrics(y_te_l, y_te_pred_l)
    l_cv5 = cross_val_r2_multiple(X_land, y_land, k=5, seed=42, ridge=1e-2)

    print(f"\n[NHÓM 2: BĐS THỔ CƯ] (N_raw={len(land_raw)}, N_clean={len(land_clean)}, dim={len(X_land[0])})")
    print(f"  Train: R2 = {l_tr_r2:.4f}, RMSE = {l_tr_rmse:.4f} tỷ, MAE = {l_tr_mae:.4f} tỷ, MAPE = {l_tr_mape:.1f}%")
    print(f"  Test : R2 = {l_te_r2:.4f}, RMSE = {l_te_rmse:.4f} tỷ, MAE = {l_te_mae:.4f} tỷ, MAPE = {l_te_mape:.1f}%")
    print(f"  5-Fold CV R2 = {l_cv5:.4f} | Thời gian: Train = {land_train_ms:.4f} ms, Inference = {land_test_ms:.4f} ms")

    land_names = (
        ['Bias (Mốc: Sơn Trà, Đất nền)', 'Area (Độ dốc Sơn Trà, Đất nền)']
        + [f'D_{d}' for d in land_dists]
        + [f'PT_{pt}' for pt in land_pts]
        + [f'Area × {d}' for d in land_dists]
        + [f'Area × {pt}' for pt in land_pts]
    )
    print("  Trọng số mô hình Thổ cư:")
    for name, w in zip(land_names, w_land):
        print(f"    {name:32s}: {w:+.4f}")

    # Save district summary
    out_csv = os.path.join(script_dir, 'chi_tiet_danh_gia_thuat_toan.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = [
            'district', 'n_raw', 'n_clean', 'slope_ty_per_m2', 'intercept_ty',
            'r2_full', 'rmse_full', 'r2_cv5', 'train_r2', 'train_rmse', 'train_mae',
            'test_r2', 'test_rmse', 'test_mae', 'train_time_ms', 'test_time_ms',
            'area_med', 'price_med', 'price_per_m2_trieu'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in district_results:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\nĐã lưu kết quả chi tiết vào '{out_csv}'")

if __name__ == '__main__':
    main()
