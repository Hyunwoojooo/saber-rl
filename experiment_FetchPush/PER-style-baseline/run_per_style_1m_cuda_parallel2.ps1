param(
  [int]$MaxParallel = 2,
  [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

Set-Location "C:\Users\SS\Projects\metacog-rl\metacog-rl"

$ExpRoot = "C:\Users\SS\Projects\metacog-rl\metacog-rl\version5.0\metacog-lite-v0\experiment_FetchPush\PER-style-baseline"
$CodeRoot = Join-Path $ExpRoot "code\src"
$Script = Join-Path $CodeRoot "train_fetch_push_lite_v0.py"
$OutputRoot = Join-Path $ExpRoot "results"
$LogRoot = Join-Path $ExpRoot "logs"

New-Item -ItemType Directory -Force -Path $OutputRoot, $LogRoot | Out-Null

$env:PYTHONPATH = $CodeRoot
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$jobs = @()
foreach ($seed in 1..5) {
  $runName = "fetch_push_per_style_s$($seed)_1m_$Device"
  $jobs += [pscustomobject]@{
    Seed = $seed
    RunName = $runName
    OutLog = Join-Path $LogRoot "$runName.out.log"
    ErrLog = Join-Path $LogRoot "$runName.err.log"
    Proc = $null
  }
}

$running = @()
$failed = @()

while ($jobs.Count -gt 0 -or $running.Count -gt 0) {
  foreach ($job in @($running)) {
    if ($job.Proc.HasExited) {
      if ($job.Proc.ExitCode -ne 0) {
        $failed += $job
      }
      $running = @($running | Where-Object { $_.RunName -ne $job.RunName })
    }
  }

  while ($jobs.Count -gt 0 -and $running.Count -lt $MaxParallel) {
    $job = $jobs[0]
    $jobs = @($jobs | Select-Object -Skip 1)
    $seed = [int]$job.Seed
    $evalSeed = 10000 + $seed
    $weightSeed = 20000 + $seed

    $argsList = @(
      $Script,
      "--run_name", $job.RunName,
      "--output_root", $OutputRoot,
      "--method", "per_style_baseline",
      "--reward_type", "sparse",
      "--total_steps", "1000000",
      "--start_steps", "10000",
      "--warmup_steps", "5000",
      "--updates_per_step", "1",
      "--batch_size", "256",
      "--buffer_capacity", "300000",
      "--her_ratio", "0.8",
      "--eval_freq", "10000",
      "--eval_episodes", "20",
      "--checkpoint_every_steps", "50000",
      "--log_every_steps", "1000",
      "--weighting_mode", "batch_norm",
      "--weight_scope", "critic_only",
      "--per_alpha", "0.6",
      "--eps_PER", "1e-6",
      "--U_PER_max", "2.0",
      "--device", $Device,
      "--env_seed", "$seed",
      "--torch_seed", "$seed",
      "--numpy_seed", "$seed",
      "--action_noise_seed", "$seed",
      "--weight_seed", "$weightSeed",
      "--eval_seed", "$evalSeed",
      "--no_save_step_checkpoints"
    )

    $proc = Start-Process "C:\Python312\python.exe" `
      -ArgumentList $argsList `
      -WorkingDirectory "C:\Users\SS\Projects\metacog-rl\metacog-rl" `
      -RedirectStandardOutput $job.OutLog `
      -RedirectStandardError $job.ErrLog `
      -WindowStyle Hidden `
      -PassThru

    $job.Proc = $proc
    $running += $job
  }

  Start-Sleep -Seconds 10
}

if ($failed.Count -gt 0) {
  Write-Error "$($failed.Count) PER-style run(s) failed: $($failed.RunName -join ', ')"
  exit 1
}

Write-Host "All FetchPush PER-style 1M runs completed."

