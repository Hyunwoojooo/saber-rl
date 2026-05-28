param(
  [int]$MaxParallel = 4,
  [string]$Device = "cpu",
  [int]$TotalSteps = 1000000
)

$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\SS\Projects\metacog-rl\metacog-rl"
$ExperimentRoot = Join-Path $ProjectRoot "version5.0\metacog-lite-v0\experiment"
$LauncherLog = Join-Path $ExperimentRoot "fetch_push_b3_original_vs_runtime_opt_launcher.log"

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

function Write-LauncherLog {
  param([string]$Message)
  $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
  $line | Tee-Object -FilePath $LauncherLog -Append
}

function New-JobSpec {
  param(
    [string]$GroupName,
    [string]$CodeRoot,
    [string]$OutputRoot,
    [string]$LogRoot,
    [string]$RunName,
    [int]$Seed,
    [string[]]$ExtraArgs
  )

  $script = Join-Path $CodeRoot "train_fetch_push_lite_v0.py"
  $summaryPath = Join-Path (Join-Path $OutputRoot $RunName) "summary.json"
  $outLog = Join-Path $LogRoot "$RunName.out.log"
  $errLog = Join-Path $LogRoot "$RunName.err.log"

  $argsList = @(
    "--run_name", $RunName,
    "--output_root", $OutputRoot,
    "--method", "b3_support_only",
    "--reward_type", "sparse",
    "--total_steps", "$TotalSteps",
    "--start_steps", "10000",
    "--warmup_steps", "5000",
    "--updates_per_step", "1",
    "--batch_size", "256",
    "--buffer_capacity", "300000",
    "--her_ratio", "0.8",
    "--M_B", "32",
    "--M_Q_min", "8",
    "--N_local_min", "8",
    "--local_search_max", "512",
    "--eval_freq", "25000",
    "--eval_episodes", "20",
    "--checkpoint_every_steps", "50000",
    "--log_every_steps", "1000",
    "--N_gap_min", "256",
    "--weighting_mode", "batch_norm",
    "--weight_scope", "actor_critic",
    "--device", $Device,
    "--env_seed", "$Seed",
    "--torch_seed", "$Seed",
    "--numpy_seed", "$Seed",
    "--action_noise_seed", "$Seed",
    "--weight_seed", "$(20000 + $Seed)",
    "--eval_seed", "$(10000 + $Seed)",
    "--no_save_step_checkpoints"
  ) + $ExtraArgs

  [pscustomobject]@{
    GroupName = $GroupName
    CodeRoot = $CodeRoot
    Script = $script
    OutputRoot = $OutputRoot
    LogRoot = $LogRoot
    RunName = $RunName
    Seed = $Seed
    SummaryPath = $summaryPath
    OutLog = $outLog
    ErrLog = $errLog
    ArgsList = $argsList
    Proc = $null
  }
}

function Test-CompletedSummary {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    return $false
  }
  try {
    $summary = Get-Content -Path $Path -Raw | ConvertFrom-Json
    if ($null -ne $summary.total_steps -and [int]$summary.total_steps -ge $TotalSteps) {
      return $true
    }
  } catch {
    return $false
  }
  return $false
}

function ConvertTo-QuotedCommand {
  param([string]$CodeRoot, [string]$Script, [string[]]$ArgsList)

  $quotedScript = "'" + $Script.Replace("'", "''") + "'"
  $quotedCodeRoot = "'" + $CodeRoot.Replace("'", "''") + "'"
  $parts = @("`$env:PYTHONPATH = $quotedCodeRoot;", "python $quotedScript")
  foreach ($arg in $ArgsList) {
    $parts += "'" + $arg.Replace("'", "''") + "'"
  }
  return $parts -join " "
}

"" | Set-Content -Encoding UTF8 $LauncherLog
Write-LauncherLog "FetchPush B3 original vs runtime optimization launcher started"
Write-LauncherLog "ProjectRoot=$ProjectRoot"
Write-LauncherLog "ExperimentRoot=$ExperimentRoot"
Write-LauncherLog "MaxParallel=$MaxParallel"
Write-LauncherLog "TotalSteps=$TotalSteps"
Write-LauncherLog "Device=$Device"
Write-LauncherLog "local_search_max=512"

$groups = @(
  [pscustomobject]@{
    GroupName = "B3-Original"
    CodeRoot = Join-Path $ExperimentRoot "B3-Original\code\src"
    OutputRoot = Join-Path $ExperimentRoot "B3-Original\results"
    LogRoot = Join-Path $ExperimentRoot "B3-Original\logs"
    Prefix = "fetch_push_b3_original"
    ExtraArgs = @()
  },
  [pscustomobject]@{
    GroupName = "B3-Original-runtime-optimization"
    CodeRoot = Join-Path $ExperimentRoot "B3-Original-runtime-optimization\code\src"
    OutputRoot = Join-Path $ExperimentRoot "B3-Original-runtime-optimization\results"
    LogRoot = Join-Path $ExperimentRoot "B3-Original-runtime-optimization\logs"
    Prefix = "fetch_push_b3_runtime_opt"
    ExtraArgs = @("--support_lookup_optimization")
  }
)

$jobs = @()
foreach ($group in $groups) {
  [System.IO.Directory]::CreateDirectory($group.OutputRoot) | Out-Null
  [System.IO.Directory]::CreateDirectory($group.LogRoot) | Out-Null

  foreach ($seed in 1, 2, 3, 4, 5) {
    $runName = "$($group.Prefix)_s${seed}_${TotalSteps}_ls512"
    $jobs += New-JobSpec `
      -GroupName $group.GroupName `
      -CodeRoot $group.CodeRoot `
      -OutputRoot $group.OutputRoot `
      -LogRoot $group.LogRoot `
      -RunName $runName `
      -Seed $seed `
      -ExtraArgs $group.ExtraArgs
  }
}

$pending = @()
$skipped = @()
foreach ($job in $jobs) {
  if (Test-CompletedSummary $job.SummaryPath) {
    $skipped += $job
    Write-LauncherLog "SKIP completed: $($job.RunName)"
  } else {
    $pending += $job
  }
}

$running = @()
$completed = @()
$failed = @()

foreach ($job in $pending) {
  while (($running | Where-Object { -not $_.Proc.HasExited }).Count -ge $MaxParallel) {
    Start-Sleep -Seconds 30
    foreach ($finished in @($running | Where-Object { $_.Proc.HasExited })) {
      if ($finished.Proc.ExitCode -ne 0) {
        $failed += $finished
        Write-LauncherLog "FAILED: $($finished.RunName), exit=$($finished.Proc.ExitCode), stderr=$($finished.ErrLog)"
      } else {
        $completed += $finished
        Write-LauncherLog "DONE: $($finished.RunName)"
      }
    }
    $running = @($running | Where-Object { -not $_.Proc.HasExited })
    if ($failed.Count -gt 0) {
      break
    }
  }

  if ($failed.Count -gt 0) {
    break
  }

  Write-LauncherLog "START: $($job.RunName)"
  Write-LauncherLog "  CodeRoot=$($job.CodeRoot)"
  Write-LauncherLog "  OutputRoot=$($job.OutputRoot)"
  Write-LauncherLog "  OutLog=$($job.OutLog)"
  Write-LauncherLog "  ErrLog=$($job.ErrLog)"

  $command = ConvertTo-QuotedCommand -CodeRoot $job.CodeRoot -Script $job.Script -ArgsList $job.ArgsList
  $proc = Start-Process `
    -FilePath "powershell" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command) `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $job.OutLog `
    -RedirectStandardError $job.ErrLog `
    -WindowStyle Hidden `
    -PassThru

  $job.Proc = $proc
  $running += $job
}

foreach ($job in @($running)) {
  $job.Proc.WaitForExit()
  if ($job.Proc.ExitCode -ne 0) {
    $failed += $job
    Write-LauncherLog "FAILED: $($job.RunName), exit=$($job.Proc.ExitCode), stderr=$($job.ErrLog)"
  } else {
    $completed += $job
    Write-LauncherLog "DONE: $($job.RunName)"
  }
}

$summary = [pscustomobject]@{
  task = "fetch_push_b3_original_vs_runtime_optimization"
  total_steps = $TotalSteps
  local_search_max = 512
  max_parallel = $MaxParallel
  device = $Device
  groups = @($groups | ForEach-Object { $_.GroupName })
  seeds = @(1, 2, 3, 4, 5)
  completed_runs = @($completed | ForEach-Object { $_.RunName })
  skipped_runs = @($skipped | ForEach-Object { $_.RunName })
  failed_runs = @($failed | ForEach-Object { $_.RunName })
  summary_paths = @($jobs | ForEach-Object { $_.SummaryPath })
  log_paths = @($jobs | ForEach-Object { $_.OutLog })
  status = $(if ($failed.Count -gt 0) { "failed" } else { "completed" })
  completed_at = (Get-Date).ToString("s")
}
$summaryPath = Join-Path $ExperimentRoot "fetch_push_b3_original_vs_runtime_opt_main_summary.json"
$summary | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $summaryPath

Write-LauncherLog "Launcher summary written: $summaryPath"
if ($failed.Count -gt 0) {
  Write-LauncherLog "Launcher failed"
  exit 1
}

Write-LauncherLog "Launcher completed"
exit 0
