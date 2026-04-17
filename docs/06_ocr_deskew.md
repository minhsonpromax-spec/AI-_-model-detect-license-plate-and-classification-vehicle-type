# 06 – OCR Biển Số & Xử Lý Deskew

## Mục Lục

1. [Tổng Quan](#1-tổng-quan)
2. [Luồng Đọc Biển Số](#2-luồng-đọc-biển-số)
3. [Deskew – 4 Cách Thử](#3-deskew--4-cách-thử)
4. [Chi Tiết `helpers/utils_rotate.py`](#4-chi-tiết-helpersutils_rotatepy)
5. [Chi Tiết `helpers/ocr.py`](#5-chi-tiết-helpersocr-py)
6. [Bảng Class Names – `constant/ocr.py`](#6-bảng-class-names--constantocrpy)
7. [Tích Hợp Trong `plate_pipeline.py`](#7-tích-hợp-trong-plate_pipelinepy)

---

## 1. Tổng Quan

Khi YOLO phát hiện vùng biển số (bounding box), bước tiếp theo là đọc chính xác ký tự trên đó. Thách thức thực tế:

- Xe đi qua cổng **ở nhiều góc nghiêng khác nhau** → ảnh biển số bị lệch
- Ánh sáng không đều, bóng đổ → độ tương phản thấp
- Biển số Việt Nam có 2 dạng: **1 dòng** (xe máy) và **2 dòng** (ô tô)

Giải pháp: thử **4 cách chỉnh ảnh** (deskew) khác nhau, lấy kết quả OCR đầu tiên hợp lệ.

---

## 2. Luồng Đọc Biển Số

```
┌─────────────────────────────────────────────────────┐
│               YOLO Tracking (model_lp.pt)           │
│  → crop = vùng ảnh biển số (numpy array BGR)        │
└───────────────────┬─────────────────────────────────┘
                    │ crop
                    ▼
┌─────────────────────────────────────────────────────┐
│            _read_plate(crop)                        │
│  Thử 4 cách deskew: (cc=0,ct=0) (cc=0,ct=1)        │
│                     (cc=1,ct=0) (cc=1,ct=1)         │
└──────────┬──────────────────┬───────────────────────┘
           │                  │
           ▼                  ▼
    utils_rotate.deskew()   (lặp lại 4 lần)
           │
           ▼
┌─────────────────────────────────────────────────────┐
│            ocr.read_plate(model_ocr, rotated)       │
│  YOLO OCR model_ocr.pt phát hiện từng ký tự        │
│  → sắp xếp theo vị trí → ghép chuỗi biển số        │
│  → validate regex → trả về (plate_text, avg_conf)  │
└─────────────────────────────────────────────────────┘
           │
           ▼
   Kết quả đầu tiên != "unknown" → dùng luôn
   Nếu cả 4 đều "unknown" → trả về ("unknown", 0.0)
```

---

## 3. Deskew – 4 Cách Thử

Hàm `_read_plate` trong `plate_pipeline.py` thử 4 tổ hợp tham số:

```python
def _read_plate(self, crop: np.ndarray) -> tuple[str, float]:
    """Thử deskew các góc xoay khác nhau, lấy kết quả OCR đầu tiên hợp lệ."""
    for cc in range(2):       # cc = 0 hoặc 1
        for ct in range(2):   # ct = 0 hoặc 1
            rotated = utils_rotate.deskew(crop, cc, ct)
            lp, conf = ocr.read_plate(self._ocr_model, rotated)
            logger.debug("[OCR] cc=%d ct=%d → '%s' conf=%.2f", cc, ct, lp, conf)
            if lp != "unknown":
                return lp, conf
    return "unknown", 0.0
```

### Ý Nghĩa 4 Tổ Hợp

| Lần thử | `cc` | `ct` | Mô tả                                          |
| ------- | ---- | ---- | ---------------------------------------------- |
| 1       | 0    | 0    | Ảnh gốc + đường thẳng tham chiếu toàn bộ ảnh   |
| 2       | 0    | 1    | Ảnh gốc + bỏ qua đường thẳng quá gần cạnh trên |
| 3       | 1    | 0    | Tăng tương phản (CLAHE) + đường thẳng toàn bộ  |
| 4       | 1    | 1    | Tăng tương phản (CLAHE) + bỏ qua cạnh trên     |

**`cc` (change_contrast)**:

- `0` = dùng ảnh gốc
- `1` = áp CLAHE để tăng độ tương phản trước khi tính góc nghiêng

**`ct` (center_threshold)**:

- `0` = dùng tất cả đường thẳng Hough tìm được
- `1` = bỏ qua đường thẳng có tâm y < 7px (tránh nhiễu từ cạnh biên ảnh)

> **Tại sao thử 4 cách?**
> Không có cách nào tối ưu cho mọi điều kiện. Ánh sáng tốt → ảnh gốc đủ. Ánh sáng yếu → cần CLAHE. Ảnh có viền sáng trên cùng → `ct=1` tránh bị nhiễu. Thứ tự thử ưu tiên ảnh gốc trước (ít xử lý hơn = nhanh hơn).

---

## 4. Chi Tiết `helpers/utils_rotate.py`

### 4.1 Tổng Quan Các Hàm

```
changeContrast(img)              → tăng tương phản bằng CLAHE trong không gian LAB
compute_skew(src_img, center_thres) → tính góc nghiêng bằng Canny + HoughLinesP
rotate_image(image, angle)       → xoay ảnh theo góc tính được
deskew(src_img, change_cons, center_thres) → pipeline: [tăng contrast?] → tính góc → xoay
```

---

### 4.2 `changeContrast` – Tăng Tương Phản CLAHE

```python
def changeContrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)    # chuyển sang không gian LAB
    l_channel, a, b = cv2.split(lab)               # tách kênh L (độ sáng)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)                    # áp CLAHE chỉ lên kênh L
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img
```

**Tại sao dùng không gian LAB?**

- Kênh `L` = độ sáng, kênh `A/B` = màu sắc
- CLAHE chỉ áp lên `L` → các ký tự nét hơn, không bị lệch màu
- Tốt hơn so với tăng tương phản trực tiếp trên BGR

**CLAHE** (Contrast Limited Adaptive Histogram Equalization):

- `clipLimit=3.0` → giới hạn khuếch đại để tránh khuếch đại noise
- `tileGridSize=(8,8)` → chia ảnh thành 8×8 ô, cân bằng histogram cục bộ → hiệu quả hơn toàn cục

---

### 4.3 `compute_skew` – Tính Góc Nghiêng

```python
def compute_skew(src_img, center_thres):
    img = cv2.medianBlur(src_img, 3)               # làm mịn, giảm noise

    # Phát hiện cạnh
    edges = cv2.Canny(img,
                      threshold1=30, threshold2=100,
                      apertureSize=3, L2gradient=True)

    # Phát hiện đường thẳng dài
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30,
                             minLineLength=w / 1.5,   # chỉ lấy đường dài
                             maxLineGap=h / 3.0)      # cho phép gián đoạn

    if lines is None:
        return 1   # không tìm được → giả sử nghiêng 1°

    # Tìm đường thẳng có tâm y thấp nhất (gần đỉnh ảnh nhất)
    min_line = 100
    min_line_pos = 0
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            center_point = [((x1 + x2) / 2), ((y1 + y2) / 2)]
            if center_thres == 1:
                if center_point[1] < 7:    # bỏ qua đường sát cạnh trên
                    continue
            if center_point[1] < min_line:
                min_line = center_point[1]
                min_line_pos = i

    # Tính góc trung bình của đường thẳng đó
    angle = 0.0
    cnt = 0
    for x1, y1, x2, y2 in lines[min_line_pos]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30:           # chỉ lấy góc nhỏ (< 30°)
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt) * 180 / math.pi  # chuyển radian → độ
```

**Tại sao lấy đường thẳng gần đỉnh ảnh nhất?**

Trong ảnh biển số đã crop, dòng ký tự trên cùng thường là cạnh ngang rõ nhất. Đường nào có `center_y` nhỏ nhất = nằm cao nhất trong ảnh → đại diện cho hướng nằm ngang của biển số.

**Pipeline Canny + Hough:**

```
Ảnh biển số crop
      ↓ medianBlur (giảm noise)
      ↓ Canny (phát hiện cạnh → ảnh nhị phân)
      ↓ HoughLinesP (tìm đoạn thẳng dài trong ảnh cạnh)
      ↓ Chọn đường thẳng nằm cao nhất
      ↓ arctan2(dy, dx) → góc radian → đổi sang độ
```

---

### 4.4 `rotate_image` – Xoay Ảnh

```python
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)  # tâm ảnh
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                             flags=cv2.INTER_LINEAR)
    return result
```

- Xoay quanh **tâm ảnh** (không bị lệch vị trí)
- Scale = `1.0` (giữ nguyên kích thước)
- `INTER_LINEAR` = nội suy song tuyến → chất lượng đủ tốt, nhanh

---

### 4.5 `deskew` – Hàm Gộp

```python
def deskew(src_img, change_cons, center_thres):
    if change_cons == 1:
        # Tăng tương phản trước khi tính góc (nhưng xoay ảnh gốc)
        return rotate_image(src_img, compute_skew(changeContrast(src_img), center_thres))
    else:
        # Tính góc trực tiếp trên ảnh gốc
        return rotate_image(src_img, compute_skew(src_img, center_thres))
```

> **Chú ý quan trọng:** Khi `change_cons=1`, CLAHE chỉ dùng để **tính góc** (compute_skew nhận ảnh đã tăng contrast). Nhưng ảnh thực sự được xoay vẫn là `src_img` gốc. Điều này giúp phát hiện góc chính xác hơn mà không làm thay đổi màu sắc/chất lượng ảnh đầu ra cho OCR.

---

## 5. Chi Tiết `helpers/ocr.py`

### 5.1 Hàm Chính `read_plate`

```python
def read_plate(model, img):
    results = model(img, verbose=False)[0]

    if results.boxes is None:
        return "unknown", 0.0

    boxes = results.boxes

    # Lọc số lượng ký tự hợp lệ (6–12 ký tự)
    if len(boxes) < 6 or len(boxes) > 12:
        return "unknown", 0.0
    ...
```

Model `model_ocr.pt` detect từng **ký tự riêng lẻ** như object detection – mỗi box là 1 ký tự với class index tương ứng ký tự đó.

---

### 5.2 Thu Thập Tọa Độ Ký Tự

```python
center_list = []
total_conf = 0
count = 0

for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])

    if conf < 0.5:          # bỏ qua ký tự confidence thấp
        continue

    label = CLASS_NAMES[cls_id]   # tra bảng → ký tự thực
    x_c = (x1 + x2) / 2          # tâm x của ký tự
    y_c = (y1 + y2) / 2          # tâm y của ký tự

    center_list.append([x_c, y_c, label])
    total_conf += conf
    count += 1
```

Kết quả: `center_list` chứa tọa độ tâm và nhãn của từng ký tự đã lọc.

---

### 5.3 Phân Loại 1 Dòng vs 2 Dòng

Đây là bước quan trọng để xử lý cả biển số xe máy (1 dòng) và ô tô (2 dòng).

```python
LP_type = "1"   # mặc định 1 dòng

l_point = min(center_list, key=lambda x: x[0])   # điểm trái nhất
r_point = max(center_list, key=lambda x: x[0])   # điểm phải nhất

for ct in center_list:
    if l_point[0] != r_point[0]:
        if not check_point_linear(ct[0], ct[1],
                                  l_point[0], l_point[1],
                                  r_point[0], r_point[1]):
            LP_type = "2"   # có ký tự lệch khỏi đường thẳng → 2 dòng
            break
```

**Thuật toán kiểm tra 1 dòng:**

- Nối điểm trái nhất và phải nhất thành đường thẳng
- Nếu tất cả ký tự nằm trên đường thẳng đó (sai số ≤ 5px) → 1 dòng
- Nếu có ký tự lệch ra → 2 dòng

```
Ví dụ 1 dòng (xe máy 51F-12345):
  5  1  F  -  1  2  3  4  5
  ●  ●  ●     ●  ●  ●  ●  ●    ← tất cả cùng y ≈ giống nhau

Ví dụ 2 dòng (ô tô 75AA-37010):
  7  5  A  A          ← dòng trên (y nhỏ hơn)
     3  7  0  1  0    ← dòng dưới (y lớn hơn)
```

---

### 5.4 Ghép Chuỗi Biển Số

**Trường hợp 2 dòng:**

```python
if LP_type == "2":
    y_mean = y_sum / len(center_list)  # ngưỡng phân chia trên/dưới

    for c in center_list:
        if c[1] > y_mean:
            line_2.append(c)   # dòng dưới (y lớn hơn mean)
        else:
            line_1.append(c)   # dòng trên (y nhỏ hơn mean)

    line_1 = sorted(line_1, key=lambda x: x[0])   # sắp theo x
    line_2 = sorted(line_2, key=lambda x: x[0])

    for l1 in line_1:
        license_plate += str(l1[2])
    license_plate += "-"                            # thêm dấu gạch giữa 2 dòng
    for l2 in line_2:
        license_plate += str(l2[2])
```

**Trường hợp 1 dòng:**

```python
else:
    center_list = sorted(center_list, key=lambda x: x[0])   # sắp theo x
    for c in center_list:
        license_plate += str(c[2])
```

---

### 5.5 Validate Regex

```python
PLATE_REGEX = r"^\d{2}([A-Z]\d|[A-Z]{2})-?\d{4,5}$"

if not re.match(PLATE_REGEX, license_plate):
    return "unknown", avg_conf
```

Biển số hợp lệ phải khớp mẫu:

- `\d{2}` → 2 chữ số tỉnh (VD: `75`, `74`, `51`)
- `([A-Z]\d|[A-Z]{2})` → loại xe: `A-Z + số` (VD: `G1`, `F1`) hoặc `2 chữ cái` (VD: `AA`)
- `-?` → dấu gạch tùy chọn
- `\d{4,5}` → số thứ tự 4-5 chữ số

Ví dụ hợp lệ: `75AA-37010`, `74G1-20202`, `51F12345`

---

## 6. Bảng Class Names – `constant/ocr.py`

```python
CLASS_NAMES = [
    "1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","K","L",
    "M","N","P","S","T","U","V","X","Y","O",
    "0","Q","W","R","Y","Z"
]
```

Bảng này ánh xạ **class index** của model `model_ocr.pt` → ký tự thực tế. Thứ tự **phải khớp** với thứ tự lúc train model.

> ⚠️ Nếu OCR ra ký tự sai (VD: số `0` bị đọc thành chữ `O` hoặc ngược lại) → kiểm tra lại thứ tự trong `CLASS_NAMES` so với config train model.

**Lưu ý:** Không có chữ `I` (dễ nhầm với `1`), không có chữ `J`, `Q` ít dùng → bảng 31 ký tự được tối ưu cho biển số Việt Nam.

---

## 7. Tích Hợp Trong `plate_pipeline.py`

### 7.1 Gọi `_read_plate` Trong Vòng Lặp Chính

```python
# Trong PlatePipeline.update(frame, track_id, box, conf)
lp, conf = self._read_plate(crop)

if lp != "unknown":
    self._cache[track_id].append((lp, conf))
    self._recent.append((lp, conf))
    logger.debug("[VOTE] track=%s lp='%s' conf=%.2f", track_id, lp, conf)
```

Mỗi frame có xe → crop → thử 4 deskew → nếu đọc được → lưu vào cache theo `track_id`.

### 7.2 Kết Quả Được Voting Qua `_vote`

Sau khi thu thập đủ mẫu, `_vote` tổng hợp:

```python
# Score = tổng confidence của cùng 1 chuỗi biển số
score_map["75AA-37010"] += 0.92   # lần 1
score_map["75AA-37010"] += 0.88   # lần 2
score_map["75AA-37010"] += 0.95   # lần 3
# → score = 2.75 ≥ threshold 1.5 → PASS Normal path
```

### 7.3 Sơ Đồ Tổng Thể

```
Video Frame
    │
    ▼
YOLO Tracking (model_lp.pt)
    │ bounding box crop
    ▼
_read_plate(crop)
    ├── Thử cc=0, ct=0 → deskew → OCR → "unknown"? tiếp
    ├── Thử cc=0, ct=1 → deskew → OCR → "unknown"? tiếp
    ├── Thử cc=1, ct=0 → deskew → OCR → "75AA-37010" ✓ STOP
    └── Thử cc=1, ct=1 → (không cần thử nữa)
          │
          ▼
    Lưu vào cache[track_id]
          │ (nhiều frame)
          ▼
    _vote(plates) → "75AA-37010" (score ≥ 1.5)
          │
          ▼
    Lưu biển số vào EventManager
```

---

## Tóm Tắt

| Thành phần      | File                                                 | Vai Trò                                 |
| --------------- | ---------------------------------------------------- | --------------------------------------- |
| Bảng ký tự      | `constant/ocr.py`                                    | Ánh xạ class index → ký tự              |
| Tính góc + xoay | `helpers/utils_rotate.py`                            | Canny+Hough → góc nghiêng → xoay        |
| Tăng tương phản | `helpers/utils_rotate.py` → `changeContrast`         | CLAHE trong không gian LAB              |
| Đọc ký tự       | `helpers/ocr.py` → `read_plate`                      | YOLO OCR + ghép chuỗi + validate        |
| Thử 4 cách      | `vehicle_pipeline/plate_pipeline.py` → `_read_plate` | Loop cc×ct, lấy kết quả đầu tiên hợp lệ |
