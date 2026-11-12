"""
Computer Project MAI391: Linear Regression Analysis for Real Estate in Da Nang
Author: MAI391 Project Team
Task: Predict and analyze property prices by district and property type based on area.
Evaluates: Train/Test errors (MSE, RMSE, MAE, R2), 5-Fold Cross Validation, and Runtime.
"""

import sys
import os
import csv
import math
import time
import random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Set random seed for reproducibility
random.seed(42)

def percentile(data, p):
    """Compute percentile using linear interpolation (same as pandas/numpy)."""
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
    """Split data into train and test sets."""
    rnd = random.Random(seed)
    indices = list(range(len(X)))
    rnd.shuffle(indices)
    split_point = int(len(X) * (1 - test_size))
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]
    
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test

def fit_simple_linear_regression(x_train, y_train):
    """
    Closed-form Simple Linear Regression:
    slope = Cov(x, y) / Var(x)
    intercept = mean(y) - slope * mean(x)
    """
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
    """Compute MSE, RMSE, MAE, R2."""
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    my = sum(y_true) / n
    ss_tot = sum((yi - my) ** 2 for yi in y_true)
    ss_res = sum((yi - yhat) ** 2 for yi, yhat in zip(y_true, y_pred))
    
    mse = ss_res / n
    rmse = math.sqrt(mse)
    mae = sum(abs(yi - yhat) for yi, yhat in zip(y_true, y_pred)) / n
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return mse, rmse, mae, r2

def cross_val_r2(x, y, k=5, seed=42):
    """K-Fold cross-validation R2 score."""
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
        
        x_tr = [x[idx] for idx in train_idx]
        y_tr = [y[idx] for idx in train_idx]
        x_val = [x[idx] for idx in val_idx]
        y_val = [y[idx] for idx in val_idx]
        
        slope, intercept = fit_simple_linear_regression(x_tr, y_tr)
        y_val_pred = predict_simple(x_val, slope, intercept)
        _, _, _, r2 = compute_metrics(y_val, y_val_pred)
        r2_scores.append(r2)
        
    return sum(r2_scores) / len(r2_scores)

# Matrix operations for Multiple Linear Regression (w = (X^T X)^-1 X^T y)
def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def matmul(A, B):
    p, q, r = len(A), len(A[0]), len(B[0])
    res = [[0.0] * r for _ in range(p)]
    for i in range(p):
        for k in range(q):
            for j in range(r):
                res[i][j] += A[i][k] * B[k][j]
    return res

def matvecmul(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def invert_matrix(A):
    """Gauss-Jordan elimination for matrix inversion."""
    n = len(A)
    # Augment with identity
    M = [[float(A[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        # Pivot
        pivot_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[pivot_row] = M[pivot_row], M[i]
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            # Add small ridge regularization if singular
            pivot += 1e-6
        for j in range(2 * n):
            M[i][j] /= pivot
        for r in range(n):
            if r != i:
                factor = M[r][i]
                for j in range(2 * n):
                    M[r][j] -= factor * M[i][j]
    return [[M[i][j + n] for j in range(n)] for i in range(n)]

def fit_multiple_linear_regression(X_train, y_train):
    """w = (X^T X)^-1 X^T y"""
    Xt = transpose(X_train)
    XtX = matmul(Xt, X_train)
    # Add tiny ridge for numerical stability
    for i in range(len(XtX)):
        XtX[i][i] += 1e-7
    invXtX = invert_matrix(XtX)
    Xty = matvecmul(Xt, y_train)
    weights = matvecmul(invXtX, Xty)
    return weights

def predict_multiple(X, weights):
    return [sum(X[i][j] * weights[j] for j in range(len(weights))) for i in range(len(X))]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, 'ADY_Project_properties_cleaned_MissValuesRemoved.csv'),
        os.path.join(script_dir, '..', 'ADY_Project_properties_cleaned_MissValuesRemoved.csv'),
        os.path.join(os.getcwd(), 'ADY_Project_properties_cleaned_MissValuesRemoved.csv'),
    ]
    csv_file = None
    for path in candidates:
        if os.path.exists(path):
            csv_file = path
            break
            
    if not csv_file:
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu 'ADY_Project_properties_cleaned_MissValuesRemoved.csv'.\n"
            f"Đã tìm kiếm tại các đường dẫn:\n" + "\n".join(f" - {p}" for p in candidates)
        )
    print(f"Đang đọc dữ liệu từ: {csv_file}")
    with open(csv_file, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    
    # 1. Filter Da Nang
    dn_rows = [r for r in rows if r['Province'].strip() == 'Đà Nẵng']
    
    # 2. Deduplicate titles
    seen = set()
    cleaned_rows = []
    for r in dn_rows:
        if r['Title'] not in seen:
            seen.add(r['Title'])
            cleaned_rows.append(r)
            
    print(f"Tổng số bản ghi ban đầu: {len(rows)}")
    print(f"Bản ghi tại Đà Nẵng: {len(dn_rows)}")
    print(f"Bản ghi sau khi loại bỏ tin đăng trùng lặp: {len(cleaned_rows)}")
    
    # Function to filter IQR on Price and Area
    def filter_iqr(subset, mult=1.4):
        p = [float(r['Price_VND']) / 1e9 for r in subset]
        a = [float(r['Area']) for r in subset]
        q1_p, q3_p = percentile(p, 0.25), percentile(p, 0.75)
        iqr_p = q3_p - q1_p
        q1_a, q3_a = percentile(a, 0.25), percentile(a, 0.75)
        iqr_a = q3_a - q1_a
        return [r for r in subset if (q1_p - mult * iqr_p <= float(r['Price_VND']) / 1e9 <= q3_p + mult * iqr_p) and 
                                     (q1_a - mult * iqr_a <= float(r['Area']) <= q3_a + mult * iqr_a)]

    districts = [
        'Quận Sơn Trà', 'Quận Hải Châu', 'Quận Thanh Khê', 
        'Quận Ngũ Hành Sơn', 'Quận Cẩm Lệ', 'Quận Liên Chiểu', 'Huyện Hòa Vang'
    ]
    
    # =========================================================================
    # PART A: Linear Regression by District
    # =========================================================================
    district_results = []
    print("\n" + "="*80)
    print("KẾT QUẢ LINEAR REGRESSION THEO TỪNG QUẬN/HUYỆN (GIÁ THEO DIỆN TÍCH)")
    print("="*80)
    
    for d in districts:
        raw_d = [r for r in cleaned_rows if r['District'].strip() == d]
        inliers_d = filter_iqr(raw_d, mult=1.4)
        
        x_raw = [float(r['Area']) for r in raw_d]
        y_raw = [float(r['Price_VND']) / 1e9 for r in raw_d]
        slope_raw, intercept_raw = fit_simple_linear_regression(x_raw, y_raw)
        _, _, _, r2_raw = compute_metrics(y_raw, predict_simple(x_raw, slope_raw, intercept_raw))
        
        x_clean = [float(r['Area']) for r in inliers_d]
        y_clean = [float(r['Price_VND']) / 1e9 for r in inliers_d]
        
        # Train / Test split (80% - 20%)
        x_train, x_test, y_train, y_test = train_test_split(x_clean, y_clean, test_size=0.2, seed=42)
        
        # Measure training runtime
        t0 = time.perf_counter()
        slope, intercept = fit_simple_linear_regression(x_train, y_train)
        train_time_ms = (time.perf_counter() - t0) * 1000.0
        
        # Measure testing/inference runtime
        t0 = time.perf_counter()
        y_train_pred = predict_simple(x_train, slope, intercept)
        y_test_pred = predict_simple(x_test, slope, intercept)
        test_time_ms = (time.perf_counter() - t0) * 1000.0
        
        tr_mse, tr_rmse, tr_mae, tr_r2 = compute_metrics(y_train, y_train_pred)
        te_mse, te_rmse, te_mae, te_r2 = compute_metrics(y_test, y_test_pred)
        
        # Whole cleaned dataset metrics (for alignment with prior baseline table)
        slope_full, intercept_full = fit_simple_linear_regression(x_clean, y_clean)
        full_mse, full_rmse, full_mae, full_r2 = compute_metrics(y_clean, predict_simple(x_clean, slope_full, intercept_full))
        cv5_r2 = cross_val_r2(x_clean, y_clean, k=5, seed=42)
        
        area_med = median(x_clean)
        price_med = median(y_clean)
        price_per_m2_avg = (sum(y_clean) / sum(x_clean)) * 1000.0 # Triệu VND / m2
        
        district_results.append({
            'district': d,
            'n_raw': len(raw_d),
            'n_clean': len(inliers_d),
            'slope_ty_per_m2': slope_full,
            'intercept_ty': intercept_full,
            'r2_full': full_r2,
            'rmse_full': full_rmse,
            'r2_cv5': cv5_r2,
            'slope_train': slope,
            'intercept_train': intercept,
            'train_rmse': tr_rmse,
            'train_mae': tr_mae,
            'train_r2': tr_r2,
            'test_rmse': te_rmse,
            'test_mae': te_mae,
            'test_r2': te_r2,
            'train_time_ms': train_time_ms,
            'test_time_ms': test_time_ms,
            'slope_raw': slope_raw,
            'r2_raw': r2_raw,
            'area_med': area_med,
            'price_med': price_med,
            'price_per_m2_trieu': price_per_m2_avg
        })
        
        print(f"[{d}]")
        print(f"  Số mẫu: Raw={len(raw_d)}, Clean={len(inliers_d)} (loại {len(raw_d)-len(inliers_d)} ngoại lai)")
        print(f"  Hệ số hồi quy: Slope = {slope_full:.4f} tỷ/m2 ({slope_full*1000:.1f} triệu/m2), Intercept = {intercept_full:.4f} tỷ")
        print(f"  Đánh giá Full: R2 = {full_r2:.4f}, RMSE = {full_rmse:.4f} tỷ, R2_CV5 = {cv5_r2:.4f}")
        print(f"  Train: R2 = {tr_r2:.4f}, RMSE = {tr_rmse:.4f} tỷ, MAE = {tr_mae:.4f} tỷ")
        print(f"  Test:  R2 = {te_r2:.4f}, RMSE = {te_rmse:.4f} tỷ, MAE = {te_mae:.4f} tỷ")
        print(f"  Thời gian: Train = {train_time_ms:.4f} ms, Inference = {test_time_ms:.4f} ms")

    # =========================================================================
    # PART B: Linear Regression by Property Type
    # =========================================================================
    property_types = ['Đất nền', 'Căn hộ/Condotel', 'Nhà phố', 'Nhà riêng', 'Biệt thự']
    property_results = []
    print("\n" + "="*80)
    print("KẾT QUẢ LINEAR REGRESSION THEO TỪNG LOẠI HÌNH BẤT ĐỘNG SẢN")
    print("="*80)
    
    for pt in property_types:
        raw_pt = [r for r in cleaned_rows if r['Property_Type'].strip() == pt]
        inliers_pt = filter_iqr(raw_pt, mult=1.5)
        
        x_clean = [float(r['Area']) for r in inliers_pt]
        y_clean = [float(r['Price_VND']) / 1e9 for r in inliers_pt]
        
        x_train, x_test, y_train, y_test = train_test_split(x_clean, y_clean, test_size=0.2, seed=42)
        
        t0 = time.perf_counter()
        slope, intercept = fit_simple_linear_regression(x_train, y_train)
        train_time_ms = (time.perf_counter() - t0) * 1000.0
        
        t0 = time.perf_counter()
        y_train_pred = predict_simple(x_train, slope, intercept)
        y_test_pred = predict_simple(x_test, slope, intercept)
        test_time_ms = (time.perf_counter() - t0) * 1000.0
        
        tr_mse, tr_rmse, tr_mae, tr_r2 = compute_metrics(y_train, y_train_pred)
        te_mse, te_rmse, te_mae, te_r2 = compute_metrics(y_test, y_test_pred)
        
        slope_full, intercept_full = fit_simple_linear_regression(x_clean, y_clean)
        full_mse, full_rmse, full_mae, full_r2 = compute_metrics(y_clean, predict_simple(x_clean, slope_full, intercept_full))
        cv5_r2 = cross_val_r2(x_clean, y_clean, k=5, seed=42)
        
        property_results.append({
            'property_type': pt,
            'n_raw': len(raw_pt),
            'n_clean': len(inliers_pt),
            'slope_ty_per_m2': slope_full,
            'intercept_ty': intercept_full,
            'r2_full': full_r2,
            'rmse_full': full_rmse,
            'r2_cv5': cv5_r2,
            'train_rmse': tr_rmse,
            'train_mae': tr_mae,
            'train_r2': tr_r2,
            'test_rmse': te_rmse,
            'test_mae': te_mae,
            'test_r2': te_r2,
            'train_time_ms': train_time_ms,
            'test_time_ms': test_time_ms,
            'area_med': median(x_clean),
            'price_med': median(y_clean)
        })
        
        print(f"[{pt}]")
        print(f"  Số mẫu: Raw={len(raw_pt)}, Clean={len(inliers_pt)}")
        print(f"  Hệ số hồi quy: Slope = {slope_full:.4f} tỷ/m2 ({slope_full*1000:.1f} triệu/m2), Intercept = {intercept_full:.4f} tỷ")
        print(f"  Full Dataset: R2 = {full_r2:.4f}, RMSE = {full_rmse:.4f} tỷ, R2_CV5 = {cv5_r2:.4f}")
        print(f"  Train: R2 = {tr_r2:.4f}, RMSE = {tr_rmse:.4f} tỷ, MAE = {tr_mae:.4f} tỷ")
        print(f"  Test:  R2 = {te_r2:.4f}, RMSE = {te_rmse:.4f} tỷ, MAE = {te_mae:.4f} tỷ")
        print(f"  Thời gian: Train = {train_time_ms:.4f} ms, Inference = {test_time_ms:.4f} ms")

    # =========================================================================
    # PART C: Multiple Linear Regression (Area + Districts Dummy + Property Type Dummy)
    # =========================================================================
    print("\n" + "="*80)
    print("MÔ HÌNH HỒI QUY ĐA BIẾN TOÀN DIỆN (MULTIPLE LINEAR REGRESSION)")
    print("="*80)
    
    # Prepare clean dataset across all Da Nang
    all_clean = filter_iqr(cleaned_rows, mult=1.5)
    
    # Feature matrix X: [1 (bias), Area, District_Dummies (6), PropertyType_Dummies (4)]
    # Reference District: Hòa Vang
    # Reference Property Type: Đất nền
    ref_d = 'Huyện Hòa Vang'
    other_districts = [d for d in districts if d != ref_d]
    ref_pt = 'Đất nền'
    other_pts = [p for p in property_types if p != ref_pt]
    
    X_multi = []
    y_multi = []
    
    for r in all_clean:
        area = float(r['Area'])
        price = float(r['Price_VND']) / 1e9
        d_vec = [1.0 if r['District'].strip() == d else 0.0 for d in other_districts]
        pt_vec = [1.0 if r['Property_Type'].strip() == p else 0.0 for p in other_pts]
        row_feat = [1.0, area] + d_vec + pt_vec
        X_multi.append(row_feat)
        y_multi.append(price)
        
    X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X_multi, y_multi, test_size=0.2, seed=42)
    
    t0 = time.perf_counter()
    w_multi = fit_multiple_linear_regression(X_tr_m, y_tr_m)
    multi_train_time_ms = (time.perf_counter() - t0) * 1000.0
    
    t0 = time.perf_counter()
    y_tr_pred_m = predict_multiple(X_tr_m, w_multi)
    y_te_pred_m = predict_multiple(X_te_m, w_multi)
    multi_test_time_ms = (time.perf_counter() - t0) * 1000.0
    
    m_tr_mse, m_tr_rmse, m_tr_mae, m_tr_r2 = compute_metrics(y_tr_m, y_tr_pred_m)
    m_te_mse, m_te_rmse, m_te_mae, m_te_r2 = compute_metrics(y_te_m, y_te_pred_m)
    
    print(f"Tổng số mẫu mô hình đa biến: {len(all_clean)}")
    print(f"Train R2 = {m_tr_r2:.4f}, Train RMSE = {m_tr_rmse:.4f} tỷ, Train MAE = {m_tr_mae:.4f} tỷ")
    print(f"Test R2  = {m_te_r2:.4f}, Test RMSE  = {m_te_rmse:.4f} tỷ, Test MAE  = {m_te_mae:.4f} tỷ")
    print(f"Thời gian tính nghiệm Closed-Form Normal Equation: {multi_train_time_ms:.4f} ms")
    print(f"Thời gian suy diễn Test: {multi_test_time_ms:.4f} ms")
    
    feature_names = ['Hệ số chặn (Bias/Intercept)', 'Diện tích (Area - Tỷ/m2)'] + [f'Quận: {d}' for d in other_districts] + [f'Loại: {p}' for p in other_pts]
    print("\nCác hệ số hồi quy đa biến:")
    for name, w in zip(feature_names, w_multi):
        print(f"  {name:35s}: {w:+.4f}")
        
    # Save detailed evaluation to CSV in script directory
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
            
    print(f"\nĐã lưu kết quả chi tiết đánh giá vào '{out_csv}'")

if __name__ == '__main__':
    main()
