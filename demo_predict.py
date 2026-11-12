"""
================================================================================
DEMO DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN TP. ĐÀ NẴNG - MÔ HÌNH LINEAR REGRESSION (MAI391)
================================================================================
Inputs:
  1. Loại hình BĐS: Đất nền, Căn hộ/Condotel, Nhà phố, Nhà riêng, Biệt thự
  2. Diện tích: Giá trị số dương (m²)
  3. Khu vực: Quận Sơn Trà, Quận Hải Châu, Quận Thanh Khê, Quận Ngũ Hành Sơn,
              Quận Cẩm Lệ, Quận Liên Chiểu, Huyện Hòa Vang

Output:
  - Giá dự đoán theo Mô hình Hồi quy Đa biến (Tỷ VNĐ & VNĐ chi tiết)
  - Đơn giá trung bình ước lượng (Triệu VNĐ/m²)
  - Khoảng giá ước lượng (± RMSE)
  - So sánh với Mô hình Đơn biến theo riêng từng Quận
  - Giải thích chi tiết các thành phần đóng góp của phương trình hồi quy
"""

import sys
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# HỆ SỐ MÔ HÌNH HỒI QUY ĐA BIẾN (MULTIPLE LINEAR REGRESSION)
# Mốc tham chiếu (Baseline): Huyện Hòa Vang & Đất nền
# ==============================================================================
BIAS_INTERCEPT = -7.2100
COEF_AREA = 0.0771  # Tỷ VNĐ trên mỗi m²

COEF_DISTRICTS = {
    'Quận Sơn Trà': 10.7985,
    'Quận Hải Châu': 8.7161,
    'Quận Thanh Khê': 7.2628,
    'Quận Ngũ Hành Sơn': 7.0718,
    'Quận Cẩm Lệ': 5.9537,
    'Quận Liên Chiểu': 4.9287,
    'Huyện Hòa Vang': 0.0000   # Baseline
}

COEF_PROPERTY_TYPES = {
    'Nhà phố': 3.9637,
    'Biệt thự': 1.6782,
    'Nhà riêng': -0.4602,
    'Căn hộ/Condotel': -1.6104,
    'Đất nền': 0.0000        # Baseline
}

# RMSE và MAE riêng theo từng loại hình BĐS (đo trên tập dữ liệu sạch kiểm thử)
SEGMENT_ERRORS = {
    'Căn hộ/Condotel': {'rmse': 1.4256, 'mae': 1.0897, 'mape_pct': 22.0},
    'Nhà riêng':       {'rmse': 1.7418, 'mae': 1.3633, 'mape_pct': 24.5},
    'Nhà phố':         {'rmse': 5.6526, 'mae': 4.1830, 'mape_pct': 28.0},
    'Đất nền':         {'rmse': 5.6915, 'mae': 4.5255, 'mape_pct': 32.0},
    'Biệt thự':        {'rmse': 13.7796, 'mae': 8.2288, 'mape_pct': 35.0}
}
RMSE_MULTI_GLOBAL = 3.2427  # Tỷ VNĐ (RMSE gộp toàn thành phố)

# ==============================================================================
# HỆ SỐ MÔ HÌNH ĐƠN BIẾN THEO TỪNG QUẬN (DISTRICT-SPECIFIC LINEAR REGRESSION)
# y = slope * Area + intercept (Tỷ VNĐ)
# ==============================================================================
DISTRICT_SIMPLE_MODELS = {
    'Quận Sơn Trà': {'slope': 0.1493, 'intercept': -2.3131, 'rmse': 4.8903, 'r2': 0.6030},
    'Quận Hải Châu': {'slope': 0.1114, 'intercept': -0.5703, 'rmse': 4.5071, 'r2': 0.4585},
    'Quận Thanh Khê': {'slope': 0.1021, 'intercept': -1.4955, 'rmse': 3.2024, 'r2': 0.5155},
    'Quận Ngũ Hành Sơn': {'slope': 0.0960, 'intercept': -2.7304, 'rmse': 3.0967, 'r2': 0.6979},
    'Quận Cẩm Lệ': {'slope': 0.0761, 'intercept': -1.1550, 'rmse': 2.7935, 'r2': 0.5421},
    'Quận Liên Chiểu': {'slope': 0.0319, 'intercept': 2.1575, 'rmse': 2.1861, 'r2': 0.1449},
    'Huyện Hòa Vang': {'slope': -0.0074, 'intercept': 3.5268, 'rmse': 0.7594, 'r2': 0.1077}
}

VALID_PROPERTY_TYPES = list(COEF_PROPERTY_TYPES.keys())
VALID_DISTRICTS = list(COEF_DISTRICTS.keys())

def format_vnd(amount_billion):
    """Quy đổi số tỷ sang chuỗi tiền tệ VNĐ dễ đọc."""
    vnd = amount_billion * 1e9
    if vnd >= 1e9:
        ty = int(vnd // 1e9)
        trieu = int((vnd % 1e9) // 1e6)
        if trieu > 0:
            return f"{ty} Tỷ {trieu:,} Triệu VNĐ ({vnd:,.0f} VNĐ)"
        return f"{ty} Tỷ VNĐ ({vnd:,.0f} VNĐ)"
    elif vnd >= 1e6:
        trieu = int(vnd // 1e6)
        return f"{trieu:,} Triệu VNĐ ({vnd:,.0f} VNĐ)"
    else:
        return f"{vnd:,.0f} VNĐ"

def predict(property_type, area, district):
    """
    Hàm tính toán dự đoán từ 3 inputs.
    """
    if property_type not in COEF_PROPERTY_TYPES:
        raise ValueError(f"Loại BĐS '{property_type}' không hợp lệ. Chọn 1 trong: {VALID_PROPERTY_TYPES}")
    if district not in COEF_DISTRICTS:
        raise ValueError(f"Khu vực '{district}' không hợp lệ. Chọn 1 trong: {VALID_DISTRICTS}")
    if area <= 0:
        raise ValueError("Diện tích phải là một số thực dương lớn hơn 0.")

    # 1. Dự đoán theo Mô hình Hồi quy Đa biến
    w_district = COEF_DISTRICTS[district]
    w_prop = COEF_PROPERTY_TYPES[property_type]
    area_contrib = COEF_AREA * area
    
    pred_price_multi = BIAS_INTERCEPT + area_contrib + w_district + w_prop
    
    # Giới hạn giá trị dự đoán không âm
    pred_price_multi_clean = max(pred_price_multi, 0.3)
    
    unit_price_m2 = (pred_price_multi_clean / area) * 1000.0 # Triệu VNĐ / m²
    
    # Lấy sai số đặc thù của riêng phân khúc loại hình BĐS này
    seg_err = SEGMENT_ERRORS.get(property_type, {'rmse': 3.24, 'mae': 2.32, 'mape_pct': 25.0})
    seg_mae = seg_err['mae']
    mape_pct = seg_err['mape_pct']
    
    # Khoảng giá thực tế dựa trên sai số trung bình tuyệt đối (MAE) và tỷ lệ % phân khúc
    range_lower_realistic = max(pred_price_multi_clean * (1 - mape_pct / 100.0), 0.3)
    range_upper_realistic = pred_price_multi_clean * (1 + mape_pct / 100.0)
    
    # Khoảng giá lý thuyết theo RMSE gộp toàn cục (dễ bị kéo lệch bởi phân khúc lớn)
    range_lower_global = max(pred_price_multi_clean - RMSE_MULTI_GLOBAL, 0.2)
    range_upper_global = pred_price_multi_clean + RMSE_MULTI_GLOBAL
    
    # 2. Dự đoán theo Mô hình Đơn biến của Quận
    dist_model = DISTRICT_SIMPLE_MODELS.get(district, {})
    pred_price_simple = dist_model['slope'] * area + dist_model['intercept']
    pred_price_simple_clean = max(pred_price_simple, 0.3)
    
    return {
        'property_type': property_type,
        'area': area,
        'district': district,
        'pred_price_multi': pred_price_multi_clean,
        'unit_price_m2': unit_price_m2,
        'range_lower_realistic': range_lower_realistic,
        'range_upper_realistic': range_upper_realistic,
        'range_lower_global': range_lower_global,
        'range_upper_global': range_upper_global,
        'seg_mae': seg_mae,
        'seg_rmse': seg_err['rmse'],
        'mape_pct': mape_pct,
        'breakdown': {
            'bias': BIAS_INTERCEPT,
            'area_contrib': area_contrib,
            'district_name': district,
            'district_contrib': w_district,
            'property_name': property_type,
            'property_contrib': w_prop
        },
        'simple_model': {
            'pred_price': pred_price_simple_clean,
            'slope': dist_model['slope'],
            'intercept': dist_model['intercept'],
            'r2': dist_model['r2'],
            'rmse': dist_model['rmse']
        }
    }

def print_result(res):
    print("\n" + "=" * 76)
    print("                 KẾT QUẢ DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN (DEMO)")
    print("=" * 76)
    print("  INPUTS ĐÃ NHẬP:")
    print(f"    1. Loại hình BĐS : {res['property_type']}")
    print(f"    2. Diện tích     : {res['area']} m²")
    print(f"    3. Khu vực       : {res['district']}")
    print("-" * 76)
    
    print("  OUTPUT - KẾT QUẢ DỰ BÁO CHÍNH (MÔ HÌNH HỒI QUY ĐA BIẾN):")
    print(f"    * GIÁ DỰ ĐOÁN    : {res['pred_price_multi']:.3f} Tỷ VNĐ")
    print(f"      -> Bằng chữ    : {format_vnd(res['pred_price_multi'])}")
    print(f"    * ĐƠN GIÁ ƯỚC TÍNH: {res['unit_price_m2']:.2f} Triệu VNĐ / m²")
    print(f"    * KHOẢNG GIÁ DỰ BÁO THỰC TẾ (±{res['mape_pct']:.0f}% sai số phân khúc):")
    print(f"      [ {res['range_lower_realistic']:.2f} Tỷ  <--->  {res['range_upper_realistic']:.2f} Tỷ VNĐ ]")
    print(f"      (Sai số tuyệt đối MAE đặc thù phân khúc này chỉ ~ {res['seg_mae']:.2f} Tỷ)")
    print("-" * 76)
    print(f"  * LƯU Ý VỀ SAI SỐ RMSE TOÀN CỤC ({RMSE_MULTI_GLOBAL:.2f} Tỷ):")
    print(f"    Con số RMSE = {RMSE_MULTI_GLOBAL:.2f} Tỷ là sai số trung bình bình phương tính gộp toàn bộ")
    print(f"    thị trường (bị kéo phình to bởi phân khúc Biệt thự 20-50 tỷ và Nhà phố lớn).")
    print(f"    Đối với {res['property_type']}, sai số thực tế chỉ khoảng ±{res['seg_mae']:.2f} Tỷ.")
    print("-" * 76)
    
    b = res['breakdown']
    print("  GIẢI TRÌNH CÁC THÀNH PHẦN ĐÓNG GÓP VÀO GIÁ DỰ BÁO (MULTIPLE OLS):")
    print(f"    Công thức: ŷ = w₀ + (w_Area × Diện tích) + w_Khu_vực + w_Loại_hình")
    print(f"      + Hệ số chặn cơ sở (Bias w₀)          : {b['bias']:+8.4f} Tỷ")
    print(f"      + Đóng góp diện tích (+0.0771 × {res['area']:<5.1f} m²): {b['area_contrib']:+8.4f} Tỷ")
    print(f"      + Vị trí địa lý ({b['district_name']:<17s})   : {b['district_contrib']:+8.4f} Tỷ")
    print(f"      + Loại hình ({b['property_name']:<17s})       : {b['property_contrib']:+8.4f} Tỷ")
    print(f"      -------------------------------------------------------")
    print(f"      = Tổng giá trị dự đoán                : {res['pred_price_multi']:+8.4f} Tỷ VNĐ")
    print("-" * 76)
    
    sm = res['simple_model']
    print("  THAM KHẢO THÊM - MÔ HÌNH ĐƠN BIẾN THEO RIÊNG QUẬN NÀY:")
    print(f"    Phương trình riêng của {res['district']}:")
    print(f"      ŷ = {sm['slope']:.4f} × Area + ({sm['intercept']:.4f})")
    print(f"      -> Giá tính theo mô hình đơn biến      : {sm['pred_price']:.3f} Tỷ VNĐ ({format_vnd(sm['pred_price'])})")
    print(f"      -> Độ tin cậy mô hình của quận         : R² = {sm['r2']:.4f}, RMSE = {sm['rmse']:.3f} Tỷ")
    print("=" * 76 + "\n")

def interactive_mode():
    print("=" * 76)
    print("    CHƯƠNG TRÌNH DEMO DỰ ĐOÁN GIÁ BĐS ĐÀ NẴNG (MAI391 PROJECT)")
    print("=" * 76)
    
    # 1. Chọn loại BĐS
    print("\n[Bước 1/3] Chọn Loại hình BĐS:")
    for idx, p in enumerate(VALID_PROPERTY_TYPES, 1):
        print(f"   {idx}. {p}")
    while True:
        try:
            choice = input(f"-> Vui lòng chọn (1-{len(VALID_PROPERTY_TYPES)}) [Mặc định: 1]: ").strip()
            if not choice:
                chosen_prop = VALID_PROPERTY_TYPES[0]
                break
            idx = int(choice)
            if 1 <= idx <= len(VALID_PROPERTY_TYPES):
                chosen_prop = VALID_PROPERTY_TYPES[idx - 1]
                break
            print("   Lựa chọn không hợp lệ, vui lòng nhập lại.")
        except ValueError:
            print("   Vui lòng nhập số nguyên.")
            
    # 2. Nhập diện tích
    print("\n[Bước 2/3] Nhập Diện tích:")
    while True:
        try:
            area_in = input("-> Diện tích sử dụng (m²) [Ví dụ: 100]: ").strip()
            if not area_in:
                chosen_area = 100.0
                break
            chosen_area = float(area_in)
            if chosen_area > 0:
                break
            print("   Diện tích phải lớn hơn 0.")
        except ValueError:
            print("   Vui lòng nhập số thực hợp lệ (ví dụ: 85, 120.5).")
            
    # 3. Chọn khu vực
    print("\n[Bước 3/3] Chọn Khu vực (Quận/Huyện):")
    for idx, d in enumerate(VALID_DISTRICTS, 1):
        print(f"   {idx}. {d}")
    while True:
        try:
            choice = input(f"-> Vui lòng chọn (1-{len(VALID_DISTRICTS)}) [Mặc định: 1]: ").strip()
            if not choice:
                chosen_district = VALID_DISTRICTS[0]
                break
            idx = int(choice)
            if 1 <= idx <= len(VALID_DISTRICTS):
                chosen_district = VALID_DISTRICTS[idx - 1]
                break
            print("   Lựa chọn không hợp lệ, vui lòng nhập lại.")
        except ValueError:
            print("   Vui lòng nhập số nguyên.")
            
    # Tính toán và xuất kết quả
    res = predict(chosen_prop, chosen_area, chosen_district)
    print_result(res)

def main():
    parser = argparse.ArgumentParser(description="Demo dự đoán giá BĐS Đà Nẵng bằng Linear Regression")
    parser.add_argument('--property', type=str, default=None, help=f"Loại BĐS: {VALID_PROPERTY_TYPES}")
    parser.add_argument('--area', type=float, default=None, help="Diện tích (m²)")
    parser.add_argument('--district', type=str, default=None, help=f"Khu vực: {VALID_DISTRICTS}")
    
    args = parser.parse_args()
    
    # Nếu người dùng truyền đủ đối số thì chạy dự đoán ngay
    if args.property and args.area and args.district:
        res = predict(args.property, args.area, args.district)
        print_result(res)
    else:
        # Nếu không truyền đối số, chạy giao diện tương tác CLI thân thiện
        interactive_mode()

if __name__ == '__main__':
    main()
