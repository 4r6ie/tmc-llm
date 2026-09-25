param(
    [Parameter(Mandatory=$true)][string]$Command,
    [string]$Version = "",
    [string]$AdapterDir = "",
    [string]$Description = "",
    [string]$MergedDir = "",
    [string]$GgufDir = ""
)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".\src").Path
$env:PYTHONDONTWRITEBYTECODE = "1"
$argsList = @($Command)
if ($Version) { $argsList += @("--version", $Version) }
if ($AdapterDir) { $argsList += @("--adapter-dir", $AdapterDir) }
if ($Description) { $argsList += @("--description", $Description) }
if ($MergedDir) { $argsList += @("--merged-dir", $MergedDir) }
if ($GgufDir) { $argsList += @("--gguf-dir", $GgufDir) }
python -m tmc_llm.versioning @argsList
