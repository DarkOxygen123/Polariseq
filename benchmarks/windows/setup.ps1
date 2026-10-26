# Polariseq benchmark setup for Windows. Run once, from the cloned folder:
#
#   cd D:\polariseq
#   powershell -ExecutionPolicy Bypass -File benchmarks\windows\setup.ps1
#
# Everything it installs stays inside this folder, in .toolchain\ and .venv\:
# uv, Python 3.14.6 (the macOS campaign's interpreter), the pinned packages,
# CMake and a Rust toolchain; temporary files go to .toolchain\tmp. The one
# exception is the Visual Studio C++ build tools, a system component that
# Windows installs under Program Files; winget installs them only if missing
# (a UAC prompt). Then Polariseq is built in release mode and a one-minute
# check of the benchmark harness runs. Safe to re-run: every step skips what
# is already in place. Next step: benchmarks\windows\run.ps1.
#
# Keep this file ASCII-only. Windows PowerShell 5.1 reads a script without a
# byte-order mark in the ANSI code page, where the UTF-8 bytes of a typographic
# dash contain a quote character and silently break string parsing.

param([switch]$SkipTests)

$ErrorActionPreference = "Stop"
# Invoke-WebRequest in Windows PowerShell 5.1 is many times slower while it
# draws a progress bar, and GitHub refuses the TLS versions it may default to.
$ProgressPreference = "SilentlyContinue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Repo
$PythonVersion = "3.14.6"   # the macOS campaign's interpreter
$CMakeVersion = "4.4.2"     # the macOS build's CMake
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$LogDir = Join-Path $Repo "benchmarks\results\logs"
$Stamp = Get-Date -Format "yyyyMMddTHHmmss"

# ------------------------------------------- everything inside this folder
$Tools = Join-Path $Repo ".toolchain"
$UvDir = Join-Path $Tools "uv"
$Uv = Join-Path $UvDir "uv.exe"
$env:CARGO_HOME = Join-Path $Tools "cargo"
$env:RUSTUP_HOME = Join-Path $Tools "rustup"
$env:UV_CACHE_DIR = Join-Path $Tools "uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools "python"
$env:UV_PYTHON_PREFERENCE = "only-managed"   # never a system Python from elsewhere
$env:TEMP = Join-Path $Tools "tmp"
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = "1"
$env:Path = "$UvDir;$env:CARGO_HOME\bin;$env:Path"
New-Item -ItemType Directory -Force -Path $Tools, $UvDir, $env:TEMP, $LogDir | Out-Null

function Step([string]$Text) { Write-Host "`n==> $Text" -ForegroundColor Cyan }

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Invoke-Checked([string]$What, [scriptblock]$Command) {
    Step $What
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

function Get-File([string]$Url, [string]$Dest) {
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Dest
}

# ---------------------------------------------------------------- location
if ($Repo -match "OneDrive") {
    throw ("The repository is inside OneDrive ($Repo). The campaign writes ~45 GB of data " +
           "that OneDrive would try to upload. Clone it to D:\polariseq instead.")
}
if ($Repo.Length -gt 40) {
    Write-Warning ("The path $Repo is long. The HDF5 build nests deep folders and can hit " +
                   "Windows' 260-character limit; D:\polariseq is short enough.")
}
if (-not (Test-Command "git")) { throw "git is missing; install Git for Windows first." }

# A folder on a second drive is often owned by another account (or by nobody,
# on exFAT), and git then refuses to work in it. The campaign records the git
# commit of every run and pushes its results, so mark this folder as trusted.
function Get-GitProbe {
    $ErrorActionPreference = "Continue"
    return (& git -C $Repo rev-parse --is-inside-work-tree 2>&1 | Out-String)
}
if ((Get-GitProbe) -match "dubious ownership") {
    $Safe = $Repo -replace "\\", "/"
    Step "marking $Safe as a safe git directory (it is owned by another account)"
    & git config --global --add safe.directory $Safe
}

# ------------------------------------------------ C++ build tools (system)
$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
function Test-Msvc {
    if (-not (Test-Path $VsWhere)) { return $false }
    $found = & $VsWhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    return [bool]$found
}
if (-not (Test-Msvc)) {
    if (-not (Test-Command "winget")) {
        throw "winget is missing. Install 'App Installer' from the Microsoft Store, then re-run."
    }
    Step "installing the Visual Studio C++ build tools (system component, ~3 GB, UAC prompt)"
    & winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget `
        --accept-source-agreements --accept-package-agreements `
        --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    if (-not (Test-Msvc)) { throw "The C++ build tools did not install; see the winget output above." }
}

# ------------------------------------------------ uv and Rust (this folder)
if (-not (Test-Path $Uv)) {
    Step "downloading uv into $UvDir"
    $Zip = Join-Path $Tools "uv.zip"
    Get-File "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" $Zip
    Expand-Archive -Path $Zip -DestinationPath $UvDir -Force
    Remove-Item $Zip
}

# After the C++ tools, so rustup-init finds them and does not offer to fetch them itself.
$Rustup = Join-Path $env:CARGO_HOME "bin\rustup.exe"
if (-not (Test-Path $Rustup)) {
    Step "installing Rust into $Tools"
    $Init = Join-Path $Tools "rustup-init.exe"
    Get-File "https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe" $Init
    & $Init -y --no-modify-path --default-toolchain stable --profile default
    if ($LASTEXITCODE -ne 0) { throw "rustup-init failed (exit code $LASTEXITCODE)" }
    Remove-Item $Init
}
Invoke-Checked "Rust stable toolchain" { & $Rustup default stable }

# ---------------------------------------------------------------- python
if (-not (Test-Path $Py)) {
    Step "creating .venv on Python $PythonVersion"
    # A half-made .venv (an interrupted earlier setup) would make uv stop
    # and ask before replacing it.
    if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }
    & $Uv venv .venv --python $PythonVersion
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Python $PythonVersion is not available to uv; using the newest 3.14."
        Invoke-Checked "creating .venv on Python 3.14" { & $Uv venv .venv --python 3.14 }
    }
}
Invoke-Checked "installing the pinned packages (same versions as the macOS campaign)" {
    & $Uv pip install --python $Py -r benchmarks\requirements.txt "cmake==$CMakeVersion"
}
# uv does not compile bytecode, so each library's first import would, inside
# the first measured run (on the Ryzen laptop: 9.5 s for polariseq's first
# import and 26 s for scanpy's). Compile it all now; files already compiled
# are skipped, and the exit code is ignored because some packages ship files
# that are not meant to compile.
Step "compiling Python bytecode"
& $Py -m compileall -q -j 0 (Join-Path $Repo ".venv\Lib\site-packages") | Out-Null
# The HDF5 build finds CMake through this variable: the binary inside .venv,
# not the Python wrapper script in front of it.
$CMakeExe = & $Py -c "import cmake, os; print(os.path.join(cmake.CMAKE_BIN_DIR, 'cmake.exe'))"
if (-not $CMakeExe -or -not (Test-Path "$CMakeExe".Trim())) {
    throw "CMake did not install into .venv (looked for: $CMakeExe)"
}
$env:CMAKE = "$CMakeExe".Trim()

# ----------------------------------------------------------------- build
# maturin refuses to run when both VIRTUAL_ENV and CONDA_PREFIX are set, and a shell
# with a conda environment active has the second; the build uses .venv.
Remove-Item Env:CONDA_PREFIX -ErrorAction SilentlyContinue
Remove-Item Env:CONDA_DEFAULT_ENV -ErrorAction SilentlyContinue
$env:VIRTUAL_ENV = Join-Path $Repo ".venv"
$BuildStart = Get-Date
Invoke-Checked "building Polariseq in release mode (the first build compiles HDF5: 10-20 min)" {
    & (Join-Path $Repo ".venv\Scripts\maturin.exe") develop --release --uv
}
$BuildMinutes = [math]::Round(((Get-Date) - $BuildStart).TotalMinutes, 1)

# ------------------------------------------------------------------ check
Invoke-Checked "benchmark harness check on the 10K test file" {
    & $Py benchmarks\crossplat_run.py --smoke
}

$Pytest = "not run"
if (-not $SkipTests) {
    Step "running the test suite (informational; failures do not stop setup)"
    $PytestLog = Join-Path $LogDir "pytest_windows_$Stamp.log"
    # Start-Process redirects to files itself. Redirecting with 2>&1 would
    # not do: Windows PowerShell 5.1 turns a native program's stderr lines
    # into errors, which abort this script.
    $proc = Start-Process -FilePath $Py -WorkingDirectory $Repo -NoNewWindow -Wait -PassThru `
        -ArgumentList @("-m", "pytest", "-q", "-p", "no:cacheprovider") `
        -RedirectStandardOutput $PytestLog `
        -RedirectStandardError (Join-Path $LogDir "pytest_windows_${Stamp}_stderr.log")
    $Pytest = if ($proc.ExitCode -eq 0) { "passed" } else { "FAILED (exit $($proc.ExitCode))" }
    Get-Content $PytestLog -Tail 8 -Encoding UTF8
    Write-Host "pytest: $Pytest (full log: $PytestLog)"
}

# ---------------------------------------------------------------- record
$Versions = @(
    "setup finished $(Get-Date -Format s) on $env:COMPUTERNAME",
    "repo: $Repo",
    "git: $(git rev-parse --short HEAD)",
    "python: $(& $Py --version)",
    "uv: $(& $Uv --version)",
    "rustc: $(& (Join-Path $env:CARGO_HOME 'bin\rustc.exe') --version)",
    "cmake: $((& $env:CMAKE --version | Select-Object -First 1))",
    "polariseq build: $BuildMinutes min (maturin develop --release)",
    "pytest: $Pytest"
)
$Versions | Set-Content -Encoding ascii (Join-Path $LogDir "setup_windows_$Stamp.log")
$Versions | ForEach-Object { Write-Host "  $_" }

Write-Host "`nSetup complete. Plug in the charger, keep the lid open, then run:" -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File benchmarks\windows\run.ps1" -ForegroundColor Green
