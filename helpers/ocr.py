import math
from constant.ocr import CLASS_NAMES
import re

PLATE_REGEX = r"^\d{2}([A-Z]\d|[A-Z]{2})-?\d{4,5}$"


# ===== LINEAR =====
def linear_equation(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return 0, y1
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b


def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a * x + b
    return math.isclose(y_pred, y, abs_tol=5)


# ===== READ PLATE (YOLOv8 + CONF) =====
def read_plate(model, img):
    results = model(img, verbose=False)[0]

    if results.boxes is None:
        return "unknown", 0.0

    boxes = results.boxes

    if len(boxes) < 6 or len(boxes) > 12:
        return "unknown", 0.0

    center_list = []
    y_sum = 0

    total_conf = 0
    count = 0

    # ===== LOOP BOX =====
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < 0.5:
            continue

        label = CLASS_NAMES[cls_id]

        x_c = (x1 + x2) / 2
        y_c = (y1 + y2) / 2

        y_sum += y_c
        center_list.append([x_c, y_c, label])

        # ===== TÍNH CONF =====
        total_conf += conf
        count += 1

    if len(center_list) == 0 or count == 0:
        return "unknown", 0.0

    avg_conf = total_conf / count

    # ===== XÁC ĐỊNH 1 DÒNG / 2 DÒNG =====
    LP_type = "1"

    l_point = min(center_list, key=lambda x: x[0])
    r_point = max(center_list, key=lambda x: x[0])

    for ct in center_list:
        if l_point[0] != r_point[0]:
            if not check_point_linear(ct[0], ct[1],
                                      l_point[0], l_point[1],
                                      r_point[0], r_point[1]):
                LP_type = "2"
                break

    y_mean = y_sum / len(center_list)

    # ===== GHÉP KÝ TỰ =====
    line_1 = []
    line_2 = []
    license_plate = ""

    if LP_type == "2":
        for c in center_list:
            if c[1] > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)

        line_1 = sorted(line_1, key=lambda x: x[0])
        line_2 = sorted(line_2, key=lambda x: x[0])

        for l1 in line_1:
            license_plate += str(l1[2])

        license_plate += "-"

        for l2 in line_2:
            license_plate += str(l2[2])

    else:
        center_list = sorted(center_list, key=lambda x: x[0])

        for c in center_list:
            license_plate += str(c[2])

    if not re.match(PLATE_REGEX, license_plate):
        return "unknown", avg_conf

    return license_plate, avg_conf
