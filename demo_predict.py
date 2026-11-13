"""
================================================================================
DEMO DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN TP. ĐÀ NẴNG - LINEAR REGRESSION CẢI TIẾN (MAI391)
================================================================================
Nâng cấp mô hình:
  1. Phân tách 2 nhóm mô hình chuyên biệt:
     - Nhóm 1: Căn hộ / Condotel (Test R² = 0.6648, Test RMSE = 0.997 Tỷ, MAE = 0.719 Tỷ)
     - Nhóm 2: BĐS Thổ cư: Đất nền, Nhà phố, Nhà riêng, Biệt thự (Test R² = 0.6759, RMSE = 4.720 Tỷ)
  2. Bổ sung Biến tương tác (Interaction Terms: Area × District & Area × Property_Type)
     giúp mỗi khu vực và loại hình có đơn giá/m² (Slope) riêng biệt, sát thực tế.

Inputs:
  1. Loại hình BĐS
  2. Diện tích (m²)
  3. Khu vực (Quận / Huyện Đà Nẵng)
"""

import sys
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# NHÓM 1: MÔ HÌNH CĂN HỘ / CONDOTEL + BIẾN TƯƠNG TÁC (Mốc chuẩn: Quận Sơn Trà)
# ==============================================================================
APT_MODEL = {
    'bias': 1.0410,
    'slope_base': 0.0682,
    'rmse': 0.9971,
    'mae': 0.7187,
    'r2_test': 0.6648,
    'mape_pct': 21.9,
    'district_intercept': {
        'Quận Sơn Trà': 0.0000,
        'Quận Ngũ Hành Sơn': -0.7440,
        'Quận Hải Châu': -1.1990,
        'Quận Thanh Khê': -1.3988,
        'Quận Cẩm Lệ': 0.0677,
        'Quận Liên Chiểu': -0.3424,
        'Huyện Hòa Vang': -0.3424,  # Fallback theo Liên Chiểu do Hòa Vang không có mẫu căn hộ
    },
    'district_slope_delta': {
        'Quận Sơn Trà': 0.0000,
        'Quận Ngũ Hành Sơn': -0.0162,
        'Quận Hải Châu': 0.0101,
        'Quận Thanh Khê': -0.0115,
        'Quận Cẩm Lệ': -0.0380,
        'Quận Liên Chiểu': -0.0389,
        'Huyện Hòa Vang': -0.0389,
    }
}

# ==============================================================================
# NHÓM 2: MÔ HÌNH BĐS THỔ CƯ + BIẾN TƯƠNG TÁC (Mốc chuẩn: Quận Sơn Trà, Đất nền)
# ==============================================================================
LAND_MODEL = {
    'bias': 0.0927,
    'slope_base': 0.1226,
    'rmse': 4.7201,
    'mae': 3.5794,
    'r2_test': 0.6759,
    'mape_pct': 30.0,
    'district_intercept': {
        'Quận Sơn Trà': 0.0000,
        'Quận Hải Châu': -3.6405,
        'Quận Thanh Khê': -5.7161,
        'Quận Ngũ Hành Sơn': -3.3048,
        'Quận Cẩm Lệ': -2.6013,
        'Quận Liên Chiểu': -0.4785,
        'Huyện Hòa Vang': 1.2320,
    },
    'district_slope_delta': {
        'Quận Sơn Trà': 0.0000,
        'Quận Hải Châu': -0.0090,
        'Quận Thanh Khê': 0.0099,
        'Quận Ngũ Hành Sơn': -0.0059,
        'Quận Cẩm Lệ': -0.0452,
        'Quận Liên Chiểu': -0.0726,
        'Huyện Hòa Vang': -0.1121,
    },
    'property_intercept': {
        'Đất nền': 0.0000,
        'Nhà phố': 8.1275,
        'Nhà riêng': 6.3376,
        'Biệt thự': 1.8478,
    },
    'property_slope_delta': {
        'Đất nền': 0.0000,
        'Nhà phố': -0.0256,
        'Nhà riêng': -0.0603,
        'Biệt thự': 0.0053,
    }
}

# Mô hình đơn biến theo quận để tham khảo đối chiếu
DISTRICT_SIMPLE_MODELS = {
    'Quận Sơn Trà': {'slope': 0.1493, 'intercept': -2.3131, 'rmse': 4.8903, 'r2': 0.6030},
    'Quận Hải Châu': {'slope': 0.1114, 'intercept': -0.5703, 'rmse': 4.5071, 'r2': 0.4585},
    'Quận Thanh Khê': {'slope': 0.1021, 'intercept': -1.4955, 'rmse': 3.2024, 'r2': 0.5155},
    'Quận Ngũ Hành Sơn': {'slope': 0.0960, 'intercept': -2.7304, 'rmse': 3.0967, 'r2': 0.6979},
    'Quận Cẩm Lệ': {'slope': 0.0761, 'intercept': -1.1550, 'rmse': 2.7935, 'r2': 0.5421},
    'Quận Liên Chiểu': {'slope': 0.0319, 'intercept': 2.1575, 'rmse': 2.1861, 'r2': 0.1449},
    'Huyện Hòa Vang': {'slope': -0.0074, 'intercept': 3.5268, 'rmse': 0.7594, 'r2': 0.1077}
}

VALID_PROPERTY_TYPES = ['Đất nền', 'Căn hộ/Condotel', 'Nhà phố', 'Nhà riêng', 'Biệt thự']
VALID_DISTRICTS = list(LAND_MODEL['district_intercept'].keys())

def format_vnd(amount_billion):
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
    return f"{vnd:,.0f} VNĐ"

def predict(property_type, area, district):
    if property_type not in VALID_PROPERTY_TYPES:
        raise ValueError(f"Loại BĐS '{property_type}' không hợp lệ. Chọn 1 trong: {VALID_PROPERTY_TYPES}")
    if district not in VALID_DISTRICTS:
        raise ValueError(f"Khu vực '{district}' không hợp lệ. Chọn 1 trong: {VALID_DISTRICTS}")
    if area <= 0:
        raise ValueError("Diện tích phải là một số thực dương lớn hơn 0.")

    warning_msg = None
    if property_type == 'Căn hộ/Condotel':
        group_name = "Nhóm 1: Mô hình Căn hộ / Condotel (có Biến tương tác)"
        m = APT_MODEL
        if district == 'Huyện Hòa Vang':
            warning_msg = "Lưu ý: Tập dữ liệu Hòa Vang hiện chưa có mẫu Căn hộ; hệ số tạm tham chiếu khu vực lân cận Liên Chiểu."
        d_int = m['district_intercept'][district]
        d_slp = m['district_slope_delta'][district]
        p_int = 0.0
        p_slp = 0.0
    else:
        group_name = "Nhóm 2: Mô hình BĐS Thổ cư (có Biến tương tác)"
        m = LAND_MODEL
        d_int = m['district_intercept'][district]
        d_slp = m['district_slope_delta'][district]
        p_int = m['property_intercept'][property_type]
        p_slp = m['property_slope_delta'][property_type]

    effective_intercept = m['bias'] + d_int + p_int
    effective_slope = m['slope_base'] + d_slp + p_slp

    raw_pred = effective_intercept + effective_slope * area
    pred_price = max(raw_pred, 0.3)
    unit_price_m2 = (pred_price / area) * 1000.0

    # Khoảng dự báo: giới hạn biên độ tối đa không vượt quá 30% giá trị để tránh dải quá rộng với BĐS giá nhỏ
    margin = min(m['mae'], pred_price * 0.30)
    range_lower = max(pred_price - margin, 0.3)
    range_upper = pred_price + margin

    dist_model = DISTRICT_SIMPLE_MODELS[district]
    pred_simple = max(dist_model['slope'] * area + dist_model['intercept'], 0.3)

    return {
        'property_type': property_type,
        'area': area,
        'district': district,
        'group_name': group_name,
        'warning_msg': warning_msg,
        'pred_price': pred_price,
        'unit_price_m2': unit_price_m2,
        'range_lower': range_lower,
        'range_upper': range_upper,
        'model_rmse': m['rmse'],
        'model_mae': m['mae'],
        'model_r2': m['r2_test'],
        'effective_intercept': effective_intercept,
        'effective_slope': effective_slope,
        'components': {
            'bias': m['bias'],
            'slope_base': m['slope_base'],
            'd_int': d_int,
            'd_slp': d_slp,
            'p_int': p_int,
            'p_slp': p_slp,
        },
        'simple_model': {
            'pred_price': pred_simple,
            'slope': dist_model['slope'],
            'intercept': dist_model['intercept'],
            'r2': dist_model['r2'],
            'rmse': dist_model['rmse'],
        }
    }

def print_result(res):
    print("\n" + "=" * 78)
    print("        KẾT QUẢ DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN (LINEAR REGRESSION CẢI TIẾN)")
    print("=" * 78)
    print("  INPUTS ĐÃ NHẬP:")
    print(f"    1. Loại hình BĐS : {res['property_type']}")
    print(f"    2. Diện tích     : {res['area']} m²")
    print(f"    3. Khu vực       : {res['district']}")
    print(f"  -> Định tuyến mô hình: {res['group_name']} (Test R² = {res['model_r2']:.4f})")
    if res['warning_msg']:
        print(f"  [!] {res['warning_msg']}")
    print("-" * 78)

    print("  OUTPUT - KẾT QUẢ DỰ BÁO CHÍNH:")
    print(f"    * GIÁ DỰ ĐOÁN    : {res['pred_price']:.3f} Tỷ VNĐ")
    print(f"      -> Bằng chữ    : {format_vnd(res['pred_price'])}")
    print(f"    * ĐƠN GIÁ ƯỚC TÍNH: {res['unit_price_m2']:.2f} Triệu VNĐ / m²")
    print(f"    * KHOẢNG GIÁ DỰ BÁO (± MAE phân khúc = {res['model_mae']:.2f} Tỷ | RMSE = {res['model_rmse']:.2f} Tỷ):")
    print(f"      [ {res['range_lower']:.2f} Tỷ  <--->  {res['range_upper']:.2f} Tỷ VNĐ ]")
    print("-" * 78)

    c = res['components']
    print("  GIẢI TRÌNH PHƯƠNG TRÌNH CÓ BIẾN TƯƠNG TÁC (INTERACTION TERMS):")
    print(f"    Phương trình rút gọn cho ({res['property_type']} tại {res['district']}):")
    print(f"      ŷ = {res['effective_slope']:.4f} × Area + ({res['effective_intercept']:+.4f})")
    print(f"    Chi tiết cấu thành hệ số:")
    print(f"      + Độ dốc riêng (Slope) = {c['slope_base']:.4f} (Mốc) + ({c['d_slp']:+.4f} Quận) + ({c['p_slp']:+.4f} Loại)")
    print(f"                             = {res['effective_slope']:.4f} Tỷ/m² ({res['effective_slope']*1000:.1f} triệu/m²)")
    print(f"      + Hệ số chặn (Intercept)= {c['bias']:+.4f} (Mốc) + ({c['d_int']:+.4f} Quận) + ({c['p_int']:+.4f} Loại)")
    print(f"                             = {res['effective_intercept']:+.4f} Tỷ VNĐ")
    print(f"      => Giá trị = {res['effective_slope']:.4f} × {res['area']:.1f} + ({res['effective_intercept']:+.4f}) = {res['pred_price']:.4f} Tỷ VNĐ")
    print("-" * 78)

    sm = res['simple_model']
    print("  ĐỐI CHIẾU VỚI MÔ HÌNH ĐƠN BIẾN CỦA RIÊNG QUẬN:")
    print(f"    Phương trình đơn biến {res['district']}: ŷ = {sm['slope']:.4f} × Area + ({sm['intercept']:+.4f})")
    print(f"    -> Giá ước tính đơn biến: {sm['pred_price']:.3f} Tỷ VNĐ (R² = {sm['r2']:.4f})")
    print("=" * 78 + "\n")

def interactive_mode():
    print("=" * 78)
    print("  DEMO DỰ ĐOÁN GIÁ BĐS ĐÀ NẴNG (PHÂN NHÓM CĂN HỘ/THỔ CƯ + BIẾN TƯƠNG TÁC)")
    print("=" * 78)

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
            print("   Vui lòng nhập số thực hợp lệ.")

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

    res = predict(chosen_prop, chosen_area, chosen_district)
    print_result(res)

def main():
    parser = argparse.ArgumentParser(description="Demo dự đoán giá BĐS Đà Nẵng (Stratified + Interaction Linear Regression)")
    parser.add_argument('--property', type=str, default=None, help=f"Loại BĐS: {VALID_PROPERTY_TYPES}")
    parser.add_argument('--area', type=float, default=None, help="Diện tích (m²)")
    parser.add_argument('--district', type=str, default=None, help=f"Khu vực: {VALID_DISTRICTS}")

    args = parser.parse_args()
    if args.property and args.area and args.district:
        res = predict(args.property, args.area, args.district)
        print_result(res)
    else:
        interactive_mode()

if __name__ == '__main__':
    main()
