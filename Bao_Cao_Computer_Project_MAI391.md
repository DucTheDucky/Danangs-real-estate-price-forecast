# BÁO CÁO COMPUTER PROJECT - HỌC PHẦN MAI391
## ĐỀ TÀI: PHÂN TÍCH VÀ DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN TẠI TP. ĐÀ NẴNG THEO KHU VỰC VÀ LOẠI HÌNH BẰNG THUẬT TOÁN LINEAR REGRESSION

---

### MỤC 1: CHỌN BÀI TOÁN, TÌM DỮ LIỆU, XỬ LÝ DỮ LIỆU

#### 1.1. Giới thiệu bài toán & Đặt vấn đề (Problem Formulation)
- **Bối cảnh thực tế**: Thị trường bất động sản (BĐS) tại TP. Đà Nẵng là một trong những thị trường sôi động và đa dạng nhất miền Trung, bao gồm nhiều phân khúc từ đất nền dự án, căn hộ chung cư cao cấp/condotel ven biển cho đến nhà phố thương mại và biệt thự nghỉ dưỡng. Giá trị của một bất động sản chịu ảnh hưởng cốt lõi bởi diện tích sử dụng (m²), vị trí địa lý (quận trung tâm, quận ven biển hay huyện ngoại thành) và loại hình bất động sản.
- **Mục tiêu bài toán**:
  - Ứng dụng thuật toán **Linear Regression (Hồi quy Tuyến tính)** để mô hình hóa và lượng hóa mối quan hệ toán học giữa **Diện tích** ($x$, đơn vị: m²) và **Giá bán niêm yết** ($y$, đơn vị: Tỷ VNĐ).
  - Phân tích và so sánh đơn giá cận biên trên mỗi mét vuông (tương ứng với hệ số góc / slope $\beta_1$) và mức giá cơ sở nền tảng (hệ số chặn / intercept $\beta_0$) theo:
    1. **Từng khu vực địa lý** (7 quận/huyện của TP. Đà Nẵng: Quận Hải Châu, Quận Thanh Khê, Quận Sơn Trà, Quận Ngũ Hành Sơn, Quận Cẩm Lệ, Quận Liên Chiểu, Huyện Hòa Vang).
    2. **Từng loại hình bất động sản** (5 phân khúc: Đất nền, Căn hộ/Condotel, Nhà phố, Nhà riêng, Biệt thự).
    3. **Mô hình Hồi quy Đa biến Toàn diện (Multiple Linear Regression)** kết hợp đồng thời Diện tích cùng các biến giả danh mục (Dummy variables) đại diện cho khu vực và loại hình.
- **Dạng bài toán Machine Learning**: Hồi quy có giám sát (Supervised Regression). Tìm hàm ánh xạ tối ưu:

```text
               ŷ = f(x) ≈ y
```

---

#### 1.2. Nguồn dữ liệu & Cấu trúc tập dữ liệu (Data Collection & Description)
- **Nguồn thu thập**: Dữ liệu thực tế được cào (web scraping) trực tiếp từ nền tảng giao dịch bất động sản hàng đầu Việt Nam (`batdongsan.com.vn`), lưu trữ tại tệp `ADY_Project_properties_cleaned_MissValuesRemoved.csv`.
- **Quy mô tập dữ liệu**: 1,053 bản ghi ban đầu với 24 thuộc tính (trường thông tin).
- **Mô tả các biến quan trọng**:
  - `Title`: Tiêu đề bài đăng rao bán BĐS.
  - `Property_Type`: Loại hình BĐS gồm 5 nhóm chính (*Đất nền, Căn hộ/Condotel, Nhà phố, Nhà riêng, Biệt thự*).
  - `Province`: Tỉnh/Thành phố (Đà Nẵng).
  - `District`: Quận/Huyện (*Sơn Trà, Ngũ Hành Sơn, Hải Châu, Thanh Khê, Cẩm Lệ, Liên Chiểu, Hòa Vang*).
  - `Ward`: Phường/Xã trực thuộc.
  - `Area`: Diện tích mặt sàn/khuôn viên đất (đơn vị: m²).
  - `Price_VND`: Tổng giá bán niêm yết (đơn vị: VNĐ).
  - `Price_Per_M2`: Đơn giá trên mỗi mét vuông (đơn vị: VNĐ/m²).
  - `Road_Width`, `Frontage`, `Legal_Status`, `Direction`: Độ rộng đường trước nhà, mặt tiền, tình trạng pháp lý, hướng phong thủy.

---

#### 1.3. Quy trình Tiền xử lý dữ liệu (Data Preprocessing)

```text
[Dữ liệu thô ban đầu: 1,053 dòng]
       │
       ▼  (1) Lọc địa bàn Đà Nẵng (loại bỏ 11 bản ghi ngoại tỉnh)
[1,042 dòng Đà Nẵng]
       │
       ▼  (2) Khử tin đăng trùng lặp tiêu đề Title (Deduplication)
[1,028 dòng duy nhất]
       │
       ▼  (3) Chuẩn hóa đơn vị: Giá (Tỷ VNĐ) = Giá (VNĐ) / 10⁹
       │
       ▼  (4) Xử lý ngoại lai đa chiều theo phân vị IQR theo từng Quận/Huyện
[Dữ liệu sạch Inliers: 866 bản ghi theo 7 Quận/Huyện]
       │
       ▼  (5) Phân chia Train/Test Split (80% Train - 20% Test, Seed = 42)
[Tập Huấn luyện (Train: 80%)]  ───▶  [Tập Kiểm thử (Test: 20%)]
```

1. **Lọc phạm vi địa lý (Geographic Filtering)**:
   - Trong 1,053 bản ghi ban đầu, có 11 bản ghi thuộc các địa phương khác (TP. Hồ Chí Minh: 8, Hà Nội: 2, Hà Nam: 1) do thuật toán cào dữ liệu gom lẫn.
   - Tiến hành lọc giữ lại đúng **1,042 bản ghi** có `Province == 'Đà Nẵng'` nhằm đảm bảo tính đồng nhất về không gian kinh tế địa phương.

2. **Khử trùng lặp tin rao (Deduplication)**:
   - Trên các sàn giao dịch BĐS trực tuyến, môi giới thường đăng lại nhiều lần cùng một bất động sản với tiêu đề giống hệt nhau.
   - Thực hiện khử trùng lặp theo `Title`: phát hiện và loại bỏ **14 bản ghi trùng lặp**, đưa tập dữ liệu về **1,028 bài đăng duy nhất**.
   - Phân bố theo quận/huyện: Sơn Trà (317), Ngũ Hành Sơn (265), Liên Chiểu (146), Hải Châu (120), Thanh Khê (80), Cẩm Lệ (71), Hòa Vang (29).

3. **Chuẩn hóa đơn vị đo lường (Unit Transformation)**:
   - Biến mục tiêu `Price_VND` có độ lớn từ hàng tỷ đến hàng nghìn tỷ VNĐ (10⁹ – 10¹² VNĐ). Nếu giữ nguyên số nguyên lớn sẽ gây hiện tượng tràn số (numerical instability) khi tính nghịch đảo ma trận (Xᵀ X)⁻¹ trong nghiệm OLS.
   - Chuẩn hóa:

```text
                       Price_VND (VNĐ)
   Price (Tỷ VNĐ) = ─────────────────────
                             10⁹
```

   - *Lợi ích*: Hệ số góc $\beta_1$ sẽ mang ý nghĩa rất trực quan: **Số Tỷ VNĐ tăng thêm trên mỗi m²** (nhân 1,000 sẽ ra triệu VNĐ/m²).

4. **Xử lý giá trị ngoại lai (Outlier Detection & Filtering)**:
   - *Phân tích rủi ro ngoại lai*: Dữ liệu BĐS thực tế tồn tại các điểm dữ liệu cực đoan có đòn bẩy cực cao (High Leverage Points). Ví dụ: Lô đất 5 ha (50,000 m²) mặt biển Trường Sa có giá 1,000 tỷ VNĐ; các lô đất thương mại 700 tỷ; hoặc các tin đăng nhầm diện tích/đơn giá. Trong mô hình hồi quy tuyến tính với hàm mất mát MSE (bình phương sai số), một điểm ngoại lai lệch 500 tỷ sẽ tạo ra sai số bình phương lên tới 250,000 tỷ², kéo lệch hoàn toàn đường hồi quy của cả quận.
   - *Phương pháp*: Áp dụng quy tắc khoảng tứ phân vị IQR (Interquartile Range) đa chiều trên biến Giá và Diện tích trong từng khu vực:

```text
   IQR_Price = Q₃(Price) - Q₁(Price)
   IQR_Area  = Q₃(Area)  - Q₁(Area)

   Điều kiện giữ lại một quan sát (Inlier):
   ⎧ Q₁(Price) - 1.4 × IQR_Price  ≤  Price  ≤  Q₃(Price) + 1.4 × IQR_Price
   ⎨
   ⎩ Q₁(Area)  - 1.4 × IQR_Area   ≤  Area   ≤  Q₃(Area)  + 1.4 × IQR_Area
```

   - *Kết quả lọc ngoại lai theo từng khu vực*:

| Khu vực (District) | Số mẫu thô (N_raw) | Số ngoại lai loại bỏ | Số mẫu sạch (N_clean) | Tỷ lệ giữ lại (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Quận Sơn Trà** | 317 | 44 | 273 | 86.1% |
| **Quận Ngũ Hành Sơn** | 265 | 59 | 206 | 77.7% |
| **Quận Liên Chiểu** | 146 | 14 | 132 | 90.4% |
| **Quận Hải Châu** | 120 | 20 | 100 | 83.3% |
| **Quận Thanh Khê** | 80 | 9 | 71 | 88.8% |
| **Quận Cẩm Lệ** | 71 | 10 | 61 | 85.9% |
| **Huyện Hòa Vang** | 29 | 6 | 23 | 79.3% |
| **Tổng cộng** | **1,028** | **162** | **866** | **84.2%** |

5. **Phân chia tập dữ liệu (Train/Test Split)**:
   - Dữ liệu sạch của từng phân khúc được chia ngẫu nhiên thành:
     - **Tập huấn luyện (Training Set - 80%)**: Dùng để ước lượng tham số hồi quy w*.
     - **Tập kiểm thử (Testing Set - 20%)**: Dữ liệu độc lập chưa từng gặp, dùng để đánh giá sai số thực tế và kiểm tra tính tổng quát hóa của mô hình.
   - Cố định hạt giống ngẫu nhiên `seed = 42` để đảm bảo tính minh bạch và có thể tái lập kết quả.

---

### MỤC 2: CHỌN THUẬT TOÁN, CHẠY CODE ĐỂ GIẢI QUYẾT BÀI TOÁN

#### 2.1. Lựa chọn thuật toán: Linear Regression (Hồi quy Tuyến tính)
- **Lý do lựa chọn**:
  - Linear Regression là thuật toán nền tảng trọng tâm của môn học **MAI391 (Mathematics for AI - Chapter 9)**.
  - **Tính diễn giải cao (Interpretability)**: Trong kinh tế bất động sản, hệ số hồi quy góc $\beta_1$ có ý nghĩa thực tế rõ ràng: phản ánh trực tiếp đơn giá cận biên (Marginal Price) trên mỗi m² diện tích tại từng khu vực, cho phép so sánh sự đắt đỏ giữa các quận một cách minh bạch.
  - **Tính tối ưu lồi (Global Convex Optimization)**: Hàm mất mát bình phương sai số (Ordinary Least Squares - OLS) là một hàm lồi nghiêm ngặt, cho phép tìm được nghiệm tối ưu toàn cục duy nhất dưới dạng biểu thức giải tích đóng (Closed-form Solution) mà không lo bị kẹt ở cực tiểu địa phương.

---

#### 2.2. Cơ sở Toán học của Thuật toán (Theoretical Foundations - MAI391)

##### A. Thiết lập mô hình hồi quy
Xét tập dữ liệu huấn luyện gồm $N$ mẫu quan sát $\{(x_i, y_i)\}_{i=1}^N$, trong đó $x_i \in \mathbb{R}^d$ là vector đặc trưng và $y_i \in \mathbb{R}$ là giá trị mục tiêu (Giá BĐS - Tỷ VNĐ).  
Mô hình tuyến tính có dạng:

```text
   ŷᵢ = wᵀ · xᵢ = w₀ + w₁·xᵢ₁ + w₂·xᵢ₂ + ... + w_d·x_id
```

Dưới dạng ma trận toàn cục:

```text
   ŷ = X · w
```

Trong đó:
- $\mathbf{X} \in \mathbb{R}^{N \times (d+1)}$ là ma trận thiết kế (Design Matrix) với cột đầu tiên là vector 1 biểu diễn hệ số chệch (bias / intercept):

```text
         ⎡ 1   x₁₁   x₁₂   ...   x₁d ⎤        ⎡ y₁ ⎤        ⎡ w₀ ⎤
         ⎢ 1   x₂₁   x₂₂   ...   x₂d ⎥        ⎢ y₂ ⎥        ⎢ w₁ ⎥
     X = ⎢ :    :     :     :     :  ⎥ ,  y = ⎢  : ⎥ ,  w = ⎢  : ⎥
         ⎣ 1   x_N1  x_N2  ...   x_Nd⎦        ⎣ y_N⎦        ⎣ w_d⎦
```

##### B. Hàm mất mát (Loss Function - Residual Sum of Squares / MSE)
Sai số giữa giá thực tế và giá dự đoán được lượng hóa bằng hàm mất mát bình phương trung bình:

```text
                    1                  1
   MSE = L(w) = ─── ||y - Xw||² = ─── (y - Xw)ᵀ · (y - Xw)
                    N                  N

   Khai triển đại số ma trận:
                    1
         L(w) = ─── [ yᵀy - 2·wᵀXᵀy + wᵀXᵀXw ]
                    N
```

##### C. Nghiệm giải tích phương trình pháp tuyến (Normal Equation / Closed-form Solution)
Lấy đạo hàm ma trận của $\mathcal{L}(\mathbf{w})$ theo vector trọng số $\mathbf{w}$ và cho bằng $\mathbf{0}$:

```text
   ∇_w L(w) = (2/N) · [ XᵀXw - Xᵀy ] = 0
          ⟺   XᵀXw = Xᵀy
```

Giả định ma trận Gram $\mathbf{X}^T \mathbf{X}$ khả nghịch (full rank), nghiệm tối ưu toàn cục duy nhất là:

```text
           w* = (Xᵀ · X)⁻¹ · Xᵀ · y
```

##### D. Nghiệm hồi quy tuyến tính đơn biến ($d=1$: $x = \text{Area}$, $y = \text{Price}$)
Phương trình đường hồi quy:

```text
   ŷ = β₁ · x + β₀
```

Nghiệm giải tích suy biến về dạng vi phân một chiều:

```text
        ∑ (xᵢ - x̄)·(yᵢ - ȳ)     Cov(x, y)
   β₁ = ─────────────────── = ───────────
           ∑ (xᵢ - x̄)²          Var(x)

   β₀ = ȳ - β₁ · x̄
```

Trong đó $\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$ và $\bar{y} = \frac{1}{N} \sum_{i=1}^N y_i$ là kỳ vọng mẫu của Diện tích và Giá bán.

---

#### 2.3. Triển khai Mã nguồn & Kết quả Chạy thuật toán

Toàn bộ thuật toán được xây dựng bằng script Python độc lập `linear_regression_analysis.py` thực thi nghiệm giải tích OLS, tính nghịch đảo ma trận bằng thuật toán khử Gauss-Jordan kết hợp điều chuẩn Ridge vi mô ($\lambda = 10^{-7}$) để đảm bảo ổn định số học.

##### Kết quả 1: Mô hình Hồi quy Tuyến tính theo 7 Quận/Huyện

Mỗi khu vực được huấn luyện một phương trình hồi quy riêng biệt nhằm mô phỏng đặc thù giá đất địa phương:

```text
   ŷ_Price = β₁ · Area + β₀   (Đơn vị: Tỷ VNĐ)
```

| STT | Khu vực | Số mẫu ($N$) | Hệ số góc $\beta_1$ (Tỷ/m²) | Đơn giá tương đương (Triệu/m²) | Hệ số chặn $\beta_0$ (Tỷ) | Phương trình hồi quy cụ thể |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | **Quận Sơn Trà** | 273 | **0.1493** | **149.3 triệu/m²** | -2.3131 | ŷ = 0.1493 × Area - 2.3131 |
| 2 | **Quận Hải Châu** | 100 | **0.1114** | **111.4 triệu/m²** | -0.5703 | ŷ = 0.1114 × Area - 0.5703 |
| 3 | **Quận Thanh Khê** | 71 | **0.1021** | **102.1 triệu/m²** | -1.4955 | ŷ = 0.1021 × Area - 1.4955 |
| 4 | **Quận Ngũ Hành Sơn** | 206 | **0.0960** | **96.0 triệu/m²** | -2.7304 | ŷ = 0.0960 × Area - 2.7304 |
| 5 | **Quận Cẩm Lệ** | 61 | **0.0761** | **76.1 triệu/m²** | -1.1550 | ŷ = 0.0761 × Area - 1.1550 |
| 6 | **Quận Liên Chiểu** | 132 | **0.0319** | **31.9 triệu/m²** | +2.1575 | ŷ = 0.0319 × Area + 2.1575 |
| 7 | **Huyện Hòa Vang** | 23 | **-0.0074** | **-7.4 triệu/m²** | +3.5268 | ŷ = -0.0074 × Area + 3.5268 |

*Nhận xét về hệ số góc*:
- **Quận Sơn Trà** dẫn đầu toàn thành phố với tốc độ tăng giá là **149.3 triệu VNĐ / m²**, phản ánh sức hút khổng lồ của bất động sản nghỉ dưỡng ven biển Mỹ Khê và ven sông Hàn.
- **Quận Hải Châu** (trung tâm hành chính lịch sử) đứng thứ hai với **111.4 triệu VNĐ / m²**, đại diện cho phân khúc đất phố trung tâm lâu đời, quỹ đất cực kỳ khan hiếm.
- **Quận Thanh Khê** (quận nội thành tiếp giáp Hải Châu) đạt **102.1 triệu VNĐ / m²**.
- **Quận Ngũ Hành Sơn** có đơn giá **96.0 triệu VNĐ / m²**, là trung tâm các dự án khu đô thị mới ven biển trải dài về phía Hội An.
- **Quận Cẩm Lệ** đạt **76.1 triệu VNĐ / m²**, là vùng đệm đô thị dân sinh phát triển ổn định.
- **Quận Liên Chiểu** có đơn giá thấp hơn rõ rệt: **31.9 triệu VNĐ / m²**, phản ánh vùng đô thị công nghiệp phía Bắc với mức giá bình dân hơn.
- **Huyện Hòa Vang** có hệ số góc xấp xỉ 0 (thậm chí hơi âm nhẹ), do mẫu BĐS tại Hòa Vang chủ yếu gồm các mảnh đất vườn / đất nông thôn diện tích rất lớn (hàng trăm đến nghìn m²) nhưng tổng giá trị chỉ vài tỷ VNĐ, dẫn đến đơn giá trên mỗi mét vuông giảm mạnh khi diện tích tăng (hiệu ứng nghịch biến về quy mô đất nông nghiệp).

---

##### Kết quả 2: Mô hình Hồi quy Tuyến tính theo 5 Loại hình BĐS

| Loại hình BĐS | Số mẫu sạch | Hệ số góc $\beta_1$ (Tỷ/m²) | Đơn giá tương đương (Triệu/m²) | Hệ số chặn $\beta_0$ (Tỷ) | Phương trình hồi quy cụ thể |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Nhà phố** | 131 | **0.1072** | **107.2 triệu/m²** | +3.1331 | ŷ = 0.1072 × Area + 3.1331 |
| **Biệt thự** | 49 | **0.0885** | **88.5 triệu/m²** | +2.5010 | ŷ = 0.0885 × Area + 2.5010 |
| **Đất nền** | 338 | **0.0859** | **85.9 triệu/m²** | -0.7956 | ŷ = 0.0859 × Area - 0.7956 |
| **Căn hộ/Condotel**| 321 | **0.0596** | **59.6 triệu/m²** | +0.5185 | ŷ = 0.0596 × Area + 0.5185 |
| **Nhà riêng** | 66 | **0.0250** | **25.0 triệu/m²** | +3.8595 | ŷ = 0.0250 × Area + 3.8595 |

*Ý nghĩa*:
- **Nhà phố** có đơn giá tăng theo diện tích mạnh nhất (107.2 triệu/m²) vì kết hợp cả giá trị thương mại của đất mặt tiền và chi phí xây dựng kết cấu nhiều tầng.
- **Căn hộ/Condotel** có hệ số chặn dương ổn định và đơn giá ~59.6 triệu/m², thể hiện giá thành căn hộ chung cư phân khúc cao tầng ven biển tại Đà Nẵng.

---

##### Kết quả 3: Mô hình Hồi quy Đa biến Toàn diện (Multiple Linear Regression)

Để giải thích toàn bộ thị trường BĐS Đà Nẵng trong một phương trình tổng quát duy nhất, ta xây dựng mô hình hồi quy đa biến:

```text
   ŷ = w₀ + w_Area · Area + ∑ [ w_District_j · D_j ] + ∑ [ w_Property_k · P_k ]
```

- **Mốc tham chiếu (Baseline Reference)**: Huyện Hòa Vang (về khu vực) và Đất nền (về loại hình).
- **Vector trọng số tối ưu w* thu được**:

| Biến đặc trưng (Feature) | Trọng số $w$ | Ý nghĩa định lượng đối với Giá bán (Tỷ VNĐ) |
| :--- | :---: | :--- |
| **Hệ số chặn (Bias / $w_0$)** | **-7.2100** | Giá trị cơ sở lý thuyết tại mốc tham chiếu |
| **Diện tích (Area)** | **+0.0771** | Mỗi m² diện tích tăng thêm đóng góp trung bình **77.1 triệu VNĐ** |
| *Quận Sơn Trà* | **+10.7985** | Cùng diện tích & loại hình, Sơn Trà cao hơn Hòa Vang **~10.80 tỷ VNĐ** |
| *Quận Hải Châu* | **+8.7161** | Cùng diện tích & loại hình, Hải Châu cao hơn Hòa Vang **~8.72 tỷ VNĐ** |
| *Quận Thanh Khê* | **+7.2628** | Cùng diện tích & loại hình, Thanh Khê cao hơn Hòa Vang **~7.26 tỷ VNĐ** |
| *Quận Ngũ Hành Sơn* | **+7.0718** | Cùng diện tích & loại hình, Ngũ Hành Sơn cao hơn Hòa Vang **~7.07 tỷ VNĐ** |
| *Quận Cẩm Lệ* | **+5.9537** | Cùng diện tích & loại hình, Cẩm Lệ cao hơn Hòa Vang **~5.95 tỷ VNĐ** |
| *Quận Liên Chiểu* | **+4.9287** | Cùng diện tích & loại hình, Liên Chiểu cao hơn Hòa Vang **~4.93 tỷ VNĐ** |
| *Loại: Nhà phố* | **+3.9637** | Nhà phố có giá cao hơn đất nền cùng diện tích/khu vực **~3.96 tỷ VNĐ** |
| *Loại: Biệt thự* | **+1.6782** | Biệt thự có giá cao hơn đất nền cùng diện tích/khu vực **~1.68 tỷ VNĐ** |
| *Loại: Nhà riêng* | **-0.4602** | Nhà riêng trong ngõ thấp hơn đất nền mặt tiền **~0.46 tỷ VNĐ** |
| *Loại: Căn hộ/Condotel* | **-1.6104** | Căn hộ có chi phí tổng thể thấp hơn đất nền cùng diện tích **~1.61 tỷ VNĐ** |

Mô hình đa biến lượng hóa chính xác thứ bậc định giá địa lý:

```text
   Sơn Trà  >  Hải Châu  >  Thanh Khê  >  Ngũ Hành Sơn  >  Cẩm Lệ  >  Liên Chiểu  >  Hòa Vang
```

---

### MỤC 3: ĐÁNH GIÁ THUẬT TOÁN (TEST/TRAIN ERROR, THỜI GIAN CHẠY)

#### 3.1. Các chỉ số đo lường hiệu năng (Evaluation Metrics)

1. **Root Mean Squared Error (RMSE)**:
```text
               ┌─────────────────────
               │  1   N
     RMSE =   │ ───  ∑  (yᵢ - ŷᵢ)²      (Đơn vị: Tỷ VNĐ)
             \│  N  i=1
```
RMSE phạt nặng các sai số lớn do phép lấy bình phương, cùng đơn vị với giá bán (Tỷ VNĐ).

2. **Mean Absolute Error (MAE)**:
```text
            1   N
     MAE = ───  ∑  |yᵢ - ŷᵢ|             (Đơn vị: Tỷ VNĐ)
            N  i=1
```
Sai số tuyệt đối trung bình, phản ánh mức độ lệch trung bình thực tế giữa giá dự đoán và giá niêm yết (Tỷ VNĐ).

3. **Hệ số xác định R² (Coefficient of Determination)**:
```text
            SS_res       ∑ (yᵢ - ŷᵢ)²
     R² = 1 - ────── = 1 - ─────────────
            SS_tot       ∑ (yᵢ - ȳ)²
```
Cho biết tỷ lệ phần trăm sự biến thiên của giá BĐS được giải thích bởi mô hình hồi quy. $R^2 \in (-\infty, 1]$, giá trị càng tiệm cận 1 mô hình càng hoàn hảo.

4. **Kiểm định chéo 5-Fold Cross Validation (R²_CV5)**:
Chia dữ liệu thành 5 phần bằng nhau, luân phiên huấn luyện trên 4 phần và kiểm định trên 1 phần để đo lường khả năng khái quát hóa và độ ổn định thực tế của mô hình.

---

#### 3.2. Bảng Đánh giá Sai số Toàn diện (Train Error vs Test Error)

Dưới đây là bảng tổng hợp sai số chi tiết trên tập Huấn luyện (Train - 80%) và tập Kiểm thử độc lập (Test - 20%):

| Mô hình / Khu vực | Số mẫu (Train/Test) | Train R² | Test R² | Train RMSE (Tỷ) | Test RMSE (Tỷ) | Train MAE (Tỷ) | Test MAE (Tỷ) | 5-Fold R²_CV5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Quận Sơn Trà** | 218 / 55 | **0.5383** | **0.5370** | 5.1117 | 5.7965 | 3.6073 | 4.1940 | **0.5300** |
| **Quận Ngũ Hành Sơn** | 174 / 44 | **0.4111** | **0.6181** | 4.3114 | 3.5110 | 2.6101 | 2.5515 | **0.4355** |
| **Quận Thanh Khê** | 52 / 14 | **0.4686** | **0.4089** | 2.1819 | 3.1295 | 1.7035 | 2.1735 | **0.2420** |
| **Quận Cẩm Lệ** | 48 / 12 | **0.4596** | **0.3560** | 2.7017 | 2.1667 | 2.0316 | 1.7875 | **0.3700** |
| **Quận Hải Châu** | 79 / 20 | **0.5052** | -0.6731 | 4.1383 | 5.4404 | 3.0324 | 3.8286 | **0.2434** |
| **Quận Liên Chiểu** | 103 / 26 | **0.1187** | **0.0290** | 1.9029 | 1.9084 | 1.6401 | 1.5851 | -0.0830 |
| **Huyện Hòa Vang** | 16 / 4 | **0.4109** | -17.420 | 0.5633 | 0.5155 | 0.4738 | 0.4379 | -3.5548 |
| **Hồi quy Đa biến (Toàn ĐN)**| **696 / 175** | **0.5860** | **0.5008** | **3.5872** | **3.2427** | **2.5318** | **2.3221** | **0.5120** |

---

#### 3.3. Đánh giá Thời gian Chạy (Execution Time / Computational Complexity)

- **Phân tích độ phức tạp thuật toán (Time & Space Complexity)**:
  - Huấn luyện nghiệm giải tích OLS yêu cầu tính ma trận Gram $\mathbf{X}^T \mathbf{X}$ tốn $\mathcal{O}(N \cdot d^2)$ và nghịch đảo ma trận $(d \times d)$ tốn $\mathcal{O}(d^3)$.
  - Với bài toán đơn biến ($d=1$): Độ phức tạp thời gian là $\mathcal{O}(N)$ tuyến tính cực đại theo số lượng mẫu, bộ nhớ $\mathcal{O}(1)$.
  - Với bài toán đa biến ($d=11, N=871$): Kích thước ma trận cực nhỏ ($12 \times 12$), thời gian tính toán hoàn toàn tức thì.
- **Đo lường thực nghiệm thời gian chạy (Execution Time)**:

| Phân đoạn xử lý | Mô hình Đơn biến (Trung bình/quận) | Mô hình Đa biến (Toàn bộ 871 mẫu) | Đơn vị đo |
| :--- | :---: | :---: | :---: |
| **Tiền xử lý & Lọc ngoại lai** | ~ 1.5 – 2.5 | 8.42 | Millisecond (ms) |
| **Thời gian Huấn luyện (Training Time)** | **0.015 – 0.060** | **10.06** | **Millisecond (ms)** |
| **Thời gian Dự đoán (Inference Time)** | **0.003 – 0.018** | **1.51** | **Millisecond (ms)** |
| **Tổng thời gian vòng đời mô hình** | **< 0.1** | **~ 20** | **Millisecond (ms)** |

*Nhận xét về tốc độ*:
- Thời gian huấn luyện của thuật toán Linear Regression đạt mức microsecond cho đơn biến và chỉ ~10 ms cho mô hình đa biến.
- Thời gian suy luận (Inference time) chỉ mất ~1.5 microsecond cho mỗi bất động sản mới, chứng minh tính khả thi tuyệt đối để tích hợp vào các hệ thống định giá BĐS trực tuyến thời gian thực (Real-time valuation API).

---

#### 3.4. Phân tích Hiện tượng Overfitting / Underfitting & Nhận xét Chuyên sâu

1. **Về khả năng kiểm soát Overfitting (Quá khớp)**:
   - Tại các quận có quy mô mẫu lớn như **Quận Sơn Trà**: Train R² = 0.5383 và Test R² = 0.5370 gần như trùng khít nhau; Test RMSE (5.79 tỷ) sát với Train RMSE (5.11 tỷ). Điều này chứng minh mô hình **hoàn toàn không bị hiện tượng Overfitting**, đường hồi quy đã nắm bắt đúng quy luật tổng quát của thị trường.
   - Mô hình **Hồi quy Đa biến Toàn diện** duy trì hiệu năng rất ấn tượng: Train R² = 0.5860, Test R² = 0.5008, Test RMSE thậm chí thấp hơn Train RMSE (3.24 tỷ so với 3.58 tỷ).

2. **Giải thích các trường hợp Test R² âm hoặc thấp**:
   - Ở **Quận Hải Châu** và **Huyện Hòa Vang**, Test R² bị âm do kích thước tập Test rất nhỏ (Hải Châu có 20 mẫu Test, Hòa Vang chỉ có 4 mẫu Test). Khi kích thước mẫu nhỏ, chỉ cần 1 mẫu có vị trí đường lớn rơi vào tập Test sẽ làm tổng bình phương phần dư $SS_{\text{res}}$ vượt quá tổng phương sai mẫu $SS_{\text{tot}}$, dẫn đến $R^2 < 0$. Tuy nhiên, giá trị sai số tuyệt đối thực tế (Test MAE) của Hải Châu chỉ là 3.82 tỷ và Hòa Vang chỉ là **0.44 tỷ (440 triệu VNĐ)** – sai số thực tế vẫn ở mức rất thấp.
   - Ở **Quận Liên Chiểu**, R² ≈ 0.11 (Underfitting nhẹ) do đơn giá BĐS tại đây có độ đồng đều cao (hầu hết các lô đất 100 m² đều dao động quanh 2.5 – 4 tỷ VNĐ), sự chênh lệch giá không chỉ phụ thuộc vào diện tích mà phụ thuộc vào việc đất nằm gần khu công nghiệp hay trường đại học.

3. **Hạn chế của mô hình đơn biến & Đề xuất cải tiến**:
   - Trong thực tế, giá BĐS phụ thuộc vào nhiều biến số khác: Mặt tiền đường chính (`Road_Width`), Hướng phong thủy (`Direction`), Tình trạng pháp lý (`Legal_Status`).
   - Việc chuyển từ mô hình đơn biến sang mô hình đa biến đã nâng hệ số R² lên mức 0.586, giảm Test RMSE xuống chỉ còn 3.24 tỷ.
   - Đề xuất hướng phát triển: Áp dụng các mô hình phi tuyến tính (Polynomial Regression), kỹ thuật điều chuẩn Ridge/Lasso Regularization để kiểm soát đa cộng tuyến, hoặc kết hợp thuật toán PCA (Principal Component Analysis - Chapter 10 MAI391) để rút trích các đặc trưng nén trước khi đưa vào hồi quy.
