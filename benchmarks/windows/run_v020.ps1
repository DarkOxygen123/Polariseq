# Polariseq 0.2.0 on this Windows machine at 1.3M and 4.06M cells, for the supplement.
#
#   git pull
#   powershell -ExecutionPolicy Bypass -File benchmarks\windows\run_v020.ps1
#
# 1. Rebuilds Polariseq from this checkout (setup.ps1's toolchain) and refuses to go on unless it is 0.2.0.
# 2. Fetches brain1306k.h5ad (21.1 GB) and fetal4062k.h5ad (6.7 GB) from the cluster with this machine's key
#    (-Key, default %USERPROFILE%\.ssh\id_ed25519; host and folder from POLARISEQ_DATA_HOST and POLARISEQ_DATA_DIR) if they are not here yet, and checks their sizes. To copy
#    them some other way, put them in benchmarks\results\validation_data and benchmarks\results\fetal_data.
# 3. Runs benchmarks\relicence\windows_v020.py: timed runs with UMAP (in memory once, out of core three
#    times, per dataset), the full analysis of each dataset with every figure, and the planner's prediction
#    against the measured memory; pushes the results to the branch results/windows-v020-final after each part.
#
# About 2 hours. Plug in the charger, close other programs, keep the lid open. Resumable: run it again and
# finished runs are skipped. Options: -NoPush (keep everything local), -PublishOnly (push what is on record).
#
# Keep this file ASCII-only (see the note in setup.ps1).

param(
    [string]$Key = "",
    [switch]$NoPush,
    [switch]$PublishOnly
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    throw "No .venv here. Run benchmarks\windows\setup.ps1 first."
}
$env:PYTHONUTF8 = "1"

if (-not $PublishOnly) {
    # 1. Rebuild with setup.ps1's toolchain.
    $Tools = Join-Path $Repo ".toolchain"
    $env:CARGO_HOME = Join-Path $Tools "cargo"
    $env:RUSTUP_HOME = Join-Path $Tools "rustup"
    $env:UV_CACHE_DIR = Join-Path $Tools "uv-cache"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools "python"
    $env:UV_PYTHON_PREFERENCE = "only-managed"
    $env:TEMP = Join-Path $Tools "tmp"
    $env:TMP = $env:TEMP
    $env:Path = "$(Join-Path $Tools 'uv');$env:CARGO_HOME\bin;$env:Path"
    $env:VIRTUAL_ENV = Join-Path $Repo ".venv"
    $env:CMAKE = (& $Py -c "import cmake, os; print(os.path.join(cmake.CMAKE_BIN_DIR, 'cmake.exe'))").Trim()
    Write-Host "`n==> rebuilding Polariseq (release)" -ForegroundColor Cyan
    & (Join-Path $Repo ".venv\Scripts\maturin.exe") develop --release --uv
    if ($LASTEXITCODE -ne 0) { throw "build failed (exit $LASTEXITCODE)" }
    $CheckFile = Join-Path $env:TEMP "polariseq_v020_check.py"
    @'
import polariseq
assert polariseq.__version__.startswith('0.2.0'), polariseq.__version__
print('build is polariseq', polariseq.__version__)
'@ | Set-Content -Encoding ASCII $CheckFile
    & $Py $CheckFile
    if ($LASTEXITCODE -ne 0) { throw "this build is not 0.2.0; run git pull first" }

    # 2. The data, from the cluster if not here yet.
    if (-not $Key) {
        $Key = Join-Path $env:USERPROFILE ".ssh\id_ed25519"
    }
    $Remote = $env:POLARISEQ_DATA_HOST      # user@host that holds the prepared files (REPRODUCE.md)
    $RemoteBase = $env:POLARISEQ_DATA_DIR    # the folder on it with validation_data/ and fetal_data/
    $Files = @(
        @{ Local = "benchmarks\results\validation_data\brain1306k.h5ad"; Remote = "validation_data/brain1306k.h5ad"; Bytes = 21094963453 },
        @{ Local = "benchmarks\results\fetal_data\fetal4062k.h5ad"; Remote = "fetal_data/fetal4062k.h5ad"; Bytes = 6724796853 },
        @{ Local = "benchmarks\results\fetal_data\fetal_cells.csv.gz"; Remote = "fetal_data/fetal_cells.csv.gz"; Bytes = 138208425 }
    )
    foreach ($f in $Files) {
        $dest = Join-Path $Repo $f.Local
        if ((Test-Path $dest) -and ((Get-Item $dest).Length -eq $f.Bytes)) {
            Write-Host "have $($f.Local)"
            continue
        }
        New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
        if (-not (Test-Path $Key)) {
            throw "no key at $($Key): pass -Key <path>, or copy $($f.Local) here another way"
        }
        Write-Host "`n==> fetching $($f.Remote) from the cluster" -ForegroundColor Cyan
        & scp -P $(if ($env:POLARISEQ_DATA_PORT) { $env:POLARISEQ_DATA_PORT } else { 22 }) -i $Key -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$($Remote):$RemoteBase/$($f.Remote)" "$dest.part"
        if ($LASTEXITCODE -ne 0) { throw "scp failed (exit $LASTEXITCODE) for $($f.Remote)" }
        Move-Item -Force "$dest.part" $dest
        if ((Get-Item $dest).Length -ne $f.Bytes) { throw "$($f.Local) is not $($f.Bytes) bytes" }
    }
}

# 3. The runs.
$Cli = @("benchmarks\relicence\windows_v020.py")
if ($NoPush) { $Cli += "--no-publish" }
if ($PublishOnly) { $Cli += "--publish-only" }
& $Py @Cli
exit $LASTEXITCODE
