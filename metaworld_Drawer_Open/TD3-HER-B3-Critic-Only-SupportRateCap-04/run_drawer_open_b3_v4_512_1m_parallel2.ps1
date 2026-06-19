$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Seeds = @(1, 2, 3, 4, 5)
$MaxParallel = 2
$Processes = @()

foreach ($Seed in $Seeds) {
    while (($Processes | Where-Object { -not $_.HasExited }).Count -ge $MaxParallel) {
        Start-Sleep -Seconds 20
        $Processes = $Processes | Where-Object { -not $_.HasExited }
    }

    $Args = @(
        "run.py",
        "--seed", "$Seed",
        "--steps", "1000000",
        "--device", "cuda"
    )
    $Process = Start-Process -FilePath "python" -ArgumentList $Args -WorkingDirectory $Root -PassThru -WindowStyle Hidden
    $Processes += $Process
    Write-Host "Started drawer-open B3_v4_512_s1_5 seed $Seed PID $($Process.Id)"
}

while (($Processes | Where-Object { -not $_.HasExited }).Count -gt 0) {
    Start-Sleep -Seconds 30
    $Alive = $Processes | Where-Object { -not $_.HasExited }
    Write-Host "Running: $($Alive.Count)"
}

Write-Host "All drawer-open B3_v4_512_s1_5 runs finished."
