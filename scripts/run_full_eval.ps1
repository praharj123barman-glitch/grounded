# Headless full-evaluation runner for Grounded.
# Fires on a daily schedule; runs the full generation eval once the Gemini daily
# quota has reset (answers on LLM_MODEL, judge on JUDGE_MODEL, so each stays under
# its own 20/day free-tier cap). Idempotent: once it succeeds it writes a marker,
# commits the results, and removes its own scheduled task. If quota is still spent
# it logs and leaves the task to retry tomorrow.

$ErrorActionPreference = "Continue"
$root   = "C:\Users\praha\grounded"
$py     = "$root\.venv\Scripts\python.exe"
$log    = "$root\scripts\eval_run.log"
$marker = "$root\eval\FULL_EVAL_DONE.marker"
Set-Location $root

function Log($m) { "$(Get-Date -Format o)  $m" | Out-File -FilePath $log -Append }

if (Test-Path $marker) { Log "already done, skipping"; exit 0 }
Log "=== full eval run start ==="

# rebuild the index (cheap embeddings), then the full generation eval
& $py -m grounded.ingest --source sample  *>> $log
& $py -m eval.run_evals                   *>> $log
$evalExit = $LASTEXITCODE

# retrieval-only ablation: dense vs hybrid (embeddings only, quota-safe)
"--- ablation: dense ---" | Out-File $log -Append
$env:USE_HYBRID = "false"; & $py -m eval.run_evals --retrieval-only *>> $log
Remove-Item Env:USE_HYBRID -ErrorAction SilentlyContinue

$ok = $false
if ($evalExit -eq 0 -and (Test-Path "$root\eval\results.json")) {
    $raw = Get-Content "$root\eval\results.json" -Raw
    if ($raw -match '"faithfulness":\s*([0-9.]+)' -and [double]$Matches[1] -gt 0) { $ok = $true }
}

if ($ok) {
    Copy-Item "$root\eval\results.json" "$root\eval\baseline.json" -Force
    "## Full evaluation run ($(Get-Date -Format 'yyyy-MM-dd'))`n`n``````json`n" +
        (Get-Content "$root\eval\results.json" -Raw) + "`n```````n" |
        Out-File "$root\eval\RESULTS.md" -Encoding utf8
    "done $(Get-Date -Format o)" | Out-File $marker
    Log "SUCCESS: baseline + RESULTS.md updated"

    git add eval/results.json eval/baseline.json eval/RESULTS.md 2>> $log
    git -c user.name="praharj123barman-glitch" -c user.email="praharj123barman@gmail.com" commit -q -m "eval: full generation run (headless, self-scheduled)" 2>> $log
    git push origin master 2>> $log
    schtasks /delete /tn "GroundedFullEval" /f 2>> $log
    Log "committed, pushed, scheduled task removed"
} else {
    Log "eval did not complete (exit $evalExit); quota likely not reset yet, will retry tomorrow"
}
Log "=== end ==="
