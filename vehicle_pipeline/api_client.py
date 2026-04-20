"""
Gửi kết quả event lên BE server.

Dùng requests.Session với HTTPAdapter retry để tự động thử lại
khi server 5xx hoặc timeout mạng.
"""

from __future__ import annotations

import logging
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class ApiClient:

    def __init__(self, config, app_logger: logging.Logger | None = None) -> None:
        self._cfg = config
        self._log = app_logger or logger

        # Session với auto-retry (chỉ retry lỗi server 5xx, không retry 4xx)
        self._session = requests.Session()
        adapter = HTTPAdapter(
            max_retries=Retry(
                total=config.api_retries,
                backoff_factor=0.5,                     # 0.5s, 1s, 2s, ...
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["POST"],
                raise_on_status=False,
            )
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    # Mapping nhãn nội bộ → giá trị vehicleType gửi lên API
    _VEHICLE_TYPE_MAP: dict = {
        "gasoline":  "GAS",
        "electric":  "ELEC",
        "uncertain": "UNCERTAIN",
    }

    def send_vehicle_event(
        self,
        plate: str,
        vehicle_type: str,
        extra: dict | None = None,
    ) -> bool:
        """
        POST event lên server.

        Args:
            plate       : Biển số xe đã được xác nhận (vd: "51G-12345").
            vehicle_type: "gasoline" | "electric" | "uncertain".
            extra       : Thêm các field tuỳ chọn (score, entropy, ...).

        Returns:
            True nếu gửi thành công (HTTP 2xx), False nếu thất bại.
        """
        payload: dict = {
            "licensePlate": plate,
            "timeStamp": int(time.time() * 1000),
            "vehicleType": self._VEHICLE_TYPE_MAP.get(vehicle_type, vehicle_type.upper()),
        }
        if extra:
            payload.update(extra)

        try:
            resp = self._session.post(
                self._cfg.api_url,
                json=payload,
                timeout=self._cfg.api_timeout,
            )
            resp.raise_for_status()
            self._log.info(
                "[API] Sent → plate=%s type=%s  HTTP %d",
                plate, vehicle_type, resp.status_code,
            )
            return True
        except requests.HTTPError as exc:
            self._log.error(
                "[API] HTTP error for plate=%s: %s", plate, exc
            )
        except requests.ConnectionError as exc:
            self._log.error(
                "[API] Connection error for plate=%s: %s", plate, exc
            )
        except requests.Timeout as exc:
            self._log.error(
                "[API] Timeout for plate=%s: %s", plate, exc
            )
        except requests.RequestException as exc:
            self._log.error(
                "[API] Request failed for plate=%s: %s", plate, exc
            )
        return False
