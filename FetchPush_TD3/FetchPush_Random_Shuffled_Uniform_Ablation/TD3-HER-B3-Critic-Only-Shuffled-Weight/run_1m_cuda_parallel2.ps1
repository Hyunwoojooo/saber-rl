param(
  [int]$MaxParallel = 2,
  [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ExpRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Python312\python.exe"
$Train = Join-Path $ExpRoot "code\src\train_fetch_push_lite_v0.py"
$OutputRoot = Join-Path $ExpRoot "results"
$LogRoot = Join-Path $ExpRoot "logs"
New-Item -ItemType Directory -Force $OutputRoot, $LogRoot | Out-Null

$seeds = New-Object System.Collections.Queue
1..5 | ForEach-Object { $seeds.Enqueue($_) }
$running = @()

function Start-TD3Run([int]$seed) {
  $runName = "td3_her_b3_critic_only_s$($seed)_1m_$Device"
  $outLog = Join-Path $LogRoot "$runName.out.log"
  $errLog = Join-Path $LogRoot "$runName.err.log"
  $args = @(
    $Train,
    "--run_name", $runName,
    "--output_root", $OutputRoot,
    "--method", "b3_support_only",
    "--reward_type", "sparse",
    "--total_steps", "1000000",
    "--start_steps", "10000",
    "--warmup_steps", "5000",
    "--updates_per_step", "1",
    "--batch_size", "256",
    "--buffer_capacity", "300000",
    "--her_ratio", "0.8",
    "--gamma", "0.98",
    "--tau", "0.005",
    "--actor_lr", "0.0003",
    "--critic_lr", "0.0003",
    "--hidden_dim", "256",
    "--noise_std", "0.2",
    "--policy_delay", "2",
    "--target_policy_noise", "0.2",
    "--target_noise_clip", "0.5",
    "--eval_freq", "10000",
    "--eval_episodes", "20",
    "--checkpoint_every_steps", "50000",
    "--log_every_steps", "1000",
    "--weighting_mode", "batch_norm",
    "--weight_scope", "critic_only",
    "--local_search_max", "512",
    "--support_lookup_optimization",
    "--support_scan_mode", "sticky_random_v1",
    "--support_scan_seed", "512",
    "--support_cache_max_entries", "500000",
    "--support_local_scan_cache_max_entries", "200000",
    "--support_numba_scan_enabled",
    "--no_support_batch_query_coalescing",
    "--device", $Device,
    "--env_seed", "$seed",
    "--torch_seed", "$seed",
    "--numpy_seed", "$seed",
    "--action_noise_seed", "$seed",
    "--weight_seed", "$(20000 + $seed)",
    "--eval_seed", "$(10000 + $seed)",
    "--no_save_step_checkpoints"
  )
  $p = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $ExpRoot -RedirectStandardOutput $outLog -RedirectStandardError $errLog -WindowStyle Hidden -PassThru
  [pscustomobject]@{ RunName = $runName; Process = $p; OutLog = $outLog; ErrLog = $errLog }
}

while ($seeds.Count -gt 0 -or $running.Count -gt 0) {
  while ($seeds.Count -gt 0 -and $running.Count -lt $MaxParallel) {
    $job = Start-TD3Run -seed ([int]$seeds.Dequeue())
    Write-Host "START $($job.RunName) pid=$($job.Process.Id)"
    $running += $job
  }
  Start-Sleep -Seconds 20
  foreach ($job in @($running)) {
    if ($job.Process.HasExited) {
      $exitCode = $job.Process.ExitCode
      Write-Host "END $($job.RunName) exit=$exitCode"
      if ($exitCode -ne 0) {
        throw "$($job.RunName) failed. See $($job.ErrLog)"
      }
      $running = @($running | Where-Object { $_.RunName -ne $job.RunName })
    }
  }
}

Write-Host "All TD3-HER-B3-Critic-Only FetchPush runs completed."
