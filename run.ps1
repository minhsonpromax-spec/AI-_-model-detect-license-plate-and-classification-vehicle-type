# ============================================================
#  Script chạy Vehicle Entry Pipeline
#  Cách dùng:
#    .\run.ps1                                  <- dùng mặc định
#    .\run.ps1 -Video "E:\video_ket_qua.mp4"
#    .\run.ps1 -Video "E:\video_ket_qua.mp4" -BypassOcsvm false
#    .\run.ps1 -Video "E:\video_ket_qua.mp4" -BypassOcsvm false
#    .\run.ps1 -Video "E:\video_ket_qua.mp4" -BypassOcsvm false
#    .\run.ps1 -Video "E:\video_ket_qua.mp4" -ApiUrl "http://192.168.1.10/api/events"
# ============================================================
#elec 2,4 lỗi, elec5-hard tốt nhất, gas3,4 không tốt, gas2 tốt
param(
    [string]$Video        = "D:\Wind\KHKT\data test\gas5.mp4",
    [string]$ApiUrl       = "https://aibuildingmanager.online",
    # "true"  = bypass OCSVM (dùng khi test với file mp4)
    # "false" = dùng OCSVM thật (dùng khi chạy với mic thật ở cổng)
    [ValidateSet("true","false")]
    [string]$BypassOcsvm  = "false"
)

# ── Suppress noisy lib output ────────────────────────────────
$env:TF_CPP_MIN_LOG_LEVEL   = "3"
$env:TF_ENABLE_ONEDNN_OPTS  = "0"

# ── Chạy pipeline ────────────────────────────────────────────
python vehicle_pipeline/main.py `
    --video        $Video `
    --api-url      $ApiUrl `
    --bypass-ocsvm $BypassOcsvm

# trước khi chạy với powershell: Set-ExecutionPolicy RemoteSigned
