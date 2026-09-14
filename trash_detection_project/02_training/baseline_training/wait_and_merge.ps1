# 세그 학습이 끝나면(summary.txt 가 생기면) 끊어진 기록을 자동으로 합친다.
# 작업 세션과 분리된 창 없는 프로세스로 실행한다.
$bbox = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$run  = "$bbox\04_results\baseline_training\runs\seg_yolo11n"
$log  = "$run\merge_watch_log.txt"
$deadline = (Get-Date).AddDays(3)

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 감시 시작 - summary.txt 를 기다립니다" |
    Out-File $log -Append -Encoding utf8

while (-not (Test-Path "$run\summary.txt")) {
    if ((Get-Date) -gt $deadline) {
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 3일이 지나 감시를 끝냅니다 (학습 미완료)" |
            Out-File $log -Append -Encoding utf8
        exit
    }
    Start-Sleep -Seconds 300
}

# 학습 프로세스가 마지막 파일을 다 쓰도록 잠시 기다린다
Start-Sleep -Seconds 120
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] summary.txt 확인 - 통합 실행" |
    Out-File $log -Append -Encoding utf8

$env:PYTHONIOENCODING = 'utf-8'
Set-Location $bbox
& uv run python 02_training/baseline_training/merge_resume_logs.py seg_yolo11n 2>&1 |
    Out-File $log -Append -Encoding utf8
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] 완료 (종료 코드 $LASTEXITCODE)" |
    Out-File $log -Append -Encoding utf8
