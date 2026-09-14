# seg 학습(seg_yolo11n)이 끝난 뒤 남은 작업을 순서대로 실행한다. GPU 는 한 번에 한 작업만 쓴다.
#   1. 배경 제외 학습 (yolo26s) : bbox_nobg -> seg_nobg  (GPU 전용, train.py)
#   2. 배경 포함/제외 비교      : compare_bg.py
# TensorRT 는 추론(예측) 전용이라 학습 속도와 관계가 없어 파이프라인에서 뺐다 (2026-09-11 결정).
# 1번 학습이 끝나지 않으면 2번은 건너뛴다.
$bbox = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$runs = "$bbox\04_results\baseline_training\runs"
$log  = "$runs\pipeline_log.txt"
$uv   = (Get-Command uv).Source

function Log($m) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" | Out-File $log -Append -Encoding utf8
}
function TrainRunning {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'train\.py' }).Count -gt 0
}
function Step($title, $argList, $logName) {
    Log "시작: $title"
    $p = Start-Process -FilePath $uv -ArgumentList $argList -WorkingDirectory $bbox `
        -RedirectStandardOutput "$runs\$logName.log" `
        -RedirectStandardError  "$runs\$logName.err.log" `
        -WindowStyle Hidden -PassThru -Wait
    Log "끝: $title (종료 코드 $($p.ExitCode), 출력 $runs\$logName.log)"
}

Log '대기 시작 - seg_yolo11n 학습 완료(summary.txt 생성 + 학습 프로세스 종료)를 기다립니다'
$deadline = (Get-Date).AddDays(4)
while (-not ((Test-Path "$runs\seg_yolo11n\summary.txt") -and -not (TrainRunning))) {
    if ((Get-Date) -gt $deadline) {
        Log '4일이 지나 대기를 끝냅니다 (seg 학습이 끝나지 않음). 이후 작업은 시작하지 않았습니다.'
        exit
    }
    Start-Sleep -Seconds 60     # 끝나는 대로 시작하도록 1분마다 확인
}
# seg 학습 프로세스가 GPU 메모리를 완전히 비우도록 30초만 기다린다.
# 기록 통합(wait_and_merge)은 CPU 만 잠깐 쓰므로 학습과 겹쳐도 문제없다.
Log 'seg 학습 완료 확인 - 30초 뒤 yolo26s 배경 제외 학습 시작'
Start-Sleep -Seconds 30
$env:PYTHONIOENCODING = 'utf-8'

Step '1/2 배경 제외 학습 yolo26s (bbox_nobg -> seg_nobg)' @('run','python','02_training/baseline_training/train.py','nobg') 'nobg'

if ((Test-Path "$runs\bbox_yolo26s_nobg\summary.txt") -and (Test-Path "$runs\seg_yolo26s_nobg\summary.txt")) {
    Step '2/2 배경 포함/제외 비교' @('run','python','02_training/baseline_training/compare_bg.py') 'compare'
    Log '모든 단계 완료'
} else {
    Log '배경 제외 학습이 끝나지 않아(summary.txt 없음) 비교를 건너뜁니다. 학습을 이어서 끝낸 뒤 직접 실행하세요.'
}
