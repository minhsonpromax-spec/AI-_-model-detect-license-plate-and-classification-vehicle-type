# ============================================================
#  Script chạy Vehicle Entry Pipeline
#  Cách dùng:
#    .\run.ps1                                  <- dùng mặc định
#    .\run.ps1 -Video "D:\videos\test.mp4"
#    .\run.ps1 -Video "D:\videos\test.mp4" -BypassOcsvm false
#    .\run.ps1 -Video "D:\videos\test.mp4" -ApiUrl "http://192.168.1.10/api/events"
# ============================================================

param(
    [string]$Video        = "D:\zalo_cloud\xe6.mp4",
    [string]$ApiUrl       = "http://your-api.com/vehicle-events",
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
