# SPDX-License-Identifier: GPL-3.0-or-later
# Package a Windows build the way CI ships it:
#   package.ps1 -Build <build dir> -Name <package name> -Commit <commit> [-Out <dir>] [-Qt <Qt root>]
# Writes <Out>/<Name>/ (the program folder) and <Out>/<Name>-symbols/ (the .pdb files).
# Needs VCToolsRedistDir (an MSVC developer shell) for the C++ runtime DLLs.
param(
    [Parameter(Mandatory)][string]$Build,
    [Parameter(Mandatory)][string]$Name,
    [Parameter(Mandatory)][string]$Commit,
    [string]$Out = 'artifacts',
    [string]$Qt = $env:QT_ROOT_DIR
)
$ErrorActionPreference = 'Stop'
$src = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pkg = Join-Path $Out $Name
$sym = Join-Path $Out "$Name-symbols"
foreach ($d in $pkg, $sym) {
    if (Test-Path $d) { Remove-Item -Recurse -Force $d }
    New-Item -ItemType Directory -Force $d | Out-Null
}
Copy-Item "$Build/bin/eden.exe" "$pkg/n2Den.exe"
if (Test-Path "$Build/bin/eden-cli.exe") { Copy-Item "$Build/bin/eden-cli.exe" "$pkg/n2Den-cli.exe" }
Get-ChildItem "$Build/bin" -Filter *.dll | Where-Object Name -notlike 'Qt6*' | Copy-Item -Destination $pkg
& "$Qt/bin/windeployqt.exe" --release --no-compiler-runtime --no-opengl-sw --no-translations "$pkg/n2Den.exe"
if ($LASTEXITCODE) { throw "windeployqt failed ($LASTEXITCODE)" }
Copy-Item "$env:VCToolsRedistDir/x64/Microsoft.VC143.CRT/*.dll" $pkg
Copy-Item "$src/LICENSE.txt" $pkg
Copy-Item "$src/LICENSES" "$pkg/LICENSES" -Recurse
(Get-Content "$src/dist/testers/README-n2den-windows.txt" -Raw).Replace('@COMMIT@', $Commit) |
    Set-Content -NoNewline "$pkg/README.txt"
Get-ChildItem "$Build/bin" -Filter *.pdb | Copy-Item -Destination $sym
Get-ChildItem $pkg | Select-Object Name, Length
