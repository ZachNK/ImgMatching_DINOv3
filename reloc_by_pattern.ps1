<#
.SYNOPSIS
Move non-raw embed files (png/npy/json) from D: to H: while preserving the weight/alt/rot/type structure.
.NOTES
Skips any file whose name contains "_raw_". Extensions are configurable (default: *.png, *.npy, *.json). Use -Plan to preview moves.
Supports logging with -Log [-Append] [-Show].
#>

param(
    [string]$SourceRoot = "D:\dinov3_exports\dinov3_query_embeds\shinsung_data",
    [string]$DestinationRoot = "H:\dinov3_exports\dinov3_query_embeds\shinsung_data",
    [string[]]$Extensions = @("*.png", "*.npy", "*.json"),
    [string]$ExcludePattern = "_raw_",
    [string]$Code,
    [switch]$List,
    [switch]$Plan,
    [string]$Log,
    [switch]$Append,
    [switch]$Show
)

# Logging helpers
$logBuffer = New-Object System.Collections.Generic.List[string]
function Add-Output([string]$Text, [switch]$ForceHost) {
    $logBuffer.Add($Text)
    if (-not $Log -or $Show -or $ForceHost) {
        Write-Host $Text
    }
}
function Flush-Log {
    if (-not $Log) { return }
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $payload = @($timestamp) + $logBuffer
    if ($Append) {
        Add-Content -Path $Log -Value $payload
    } else {
        Set-Content -Path $Log -Value $payload
    }
}

# Preset indices for -Code (1-based: WGT ALT ROT TYP).
$WGT = @("vits16+", "vitb16", "vith16+", "vitl16", "vitl16sat", "vits16")
$ALT = @("100", "150", "200", "300", "400")
$ROT = @("045", "090", "135", "180")
$TYP = @("DenseFT", "GlobalToken", "PatchGrid", "PatchToken")

if ($List) {
    Add-Output "===== WGT (1-based index) ====="
    for ($i = 0; $i -lt $WGT.Count; $i++) {
        Add-Output ("{0}: {1}" -f ($i+1), $WGT[$i])
    }

    Add-Output "`n===== ALT (1-based index) ====="
    for ($i = 0; $i -lt $ALT.Count; $i++) {
        Add-Output ("{0}: {1}" -f ($i+1), $ALT[$i])
    }

    Add-Output "`n===== ROT (1-based index) ====="
    for ($i = 0; $i -lt $ROT.Count; $i++) {
        Add-Output ("{0}: {1}" -f ($i+1), $ROT[$i])
    }

    Add-Output "`n===== TYP (1-based index) ====="
    for ($i = 0; $i -lt $TYP.Count; $i++) {
        Add-Output ("{0}: {1}" -f ($i+1), $TYP[$i])
    }

    Add-Output "`ne.g. -Code 2441 -> WGT index 2, ALT index 4, ROT index 4, TYP index 1`n"
    Flush-Log
    return
}

$summary = @()
$rawOnlyHit = $false
$moveQueue = New-Object System.Collections.Generic.List[pscustomobject]
$pathSummary = New-Object System.Collections.Generic.List[pscustomobject]

# If -Code is supplied, restrict to that single path.
if ($Code) {
    if ($Code.Length -ne 4 -or ($Code -notmatch '^[0-9]{4}$')) {
        throw "Code must be 4 digits: WGT ALT ROT TYP (1-based)."
    }

    $wgtIndex = [int]$Code.Substring(0,1)
    $altIndex = [int]$Code.Substring(1,1)
    $rotIndex = [int]$Code.Substring(2,1)
    $typIndex = [int]$Code.Substring(3,1)

    if ($wgtIndex -lt 1 -or $wgtIndex -gt $WGT.Count -or
        $altIndex -lt 1 -or $altIndex -gt $ALT.Count -or
        $rotIndex -lt 1 -or $rotIndex -gt $ROT.Count -or
        $typIndex -lt 1 -or $typIndex -gt $TYP.Count) {
        throw "Code indices are out of range. Use -List to see valid values."
    }

    $wName = $WGT[$wgtIndex-1]
    $aName = $ALT[$altIndex-1]
    $rName = $ROT[$rotIndex-1]
    $tName = $TYP[$typIndex-1]

    $wPath = Join-Path $SourceRoot $wName
    $aPath = Join-Path $wPath $aName
    $rPath = Join-Path $aPath $rName
    $tPath = Join-Path $rPath $tName

    if (-not (Test-Path $tPath)) {
        Add-Output "Path missing: $tPath" -ForceHost
        Flush-Log
        return
    }

    $weights = @(Get-Item -Path $wPath)
    $altFilter = $aName
    $rotFilter = $rName
    $typFilter = $tName
} else {
    # Exclude already-moved underscore folders.
    $weights = Get-ChildItem -Path $SourceRoot -Directory |
        Where-Object { $_.Name -notlike "_*" }
    $altFilter = $rotFilter = $typFilter = $null
}

foreach ($w in $weights) {
    $alts = if ($altFilter) { @(Get-Item -Path (Join-Path $w.FullName $altFilter)) } else { Get-ChildItem -Path $w.FullName -Directory }
    foreach ($alt in $alts) {
        $rots = if ($rotFilter) { @(Get-Item -Path (Join-Path $alt.FullName $rotFilter)) } else { Get-ChildItem -Path $alt.FullName -Directory }
        foreach ($rot in $rots) {
            $typs = if ($typFilter) { @(Get-Item -Path (Join-Path $rot.FullName $typFilter)) } else { Get-ChildItem -Path $rot.FullName -Directory }
            foreach ($typ in $typs) {
                # Match files against any provided extension pattern (e.g., *.png, *.npy, *.json).
                $allFiles = @(Get-ChildItem -Path $typ.FullName -File |
                    Where-Object {
                        $name = $_.Name
                        $Extensions | Where-Object { $name -like $_ } | Select-Object -First 1
                    })
                if ($allFiles.Count -eq 0) { continue }

                $filesToMove = @($allFiles | Where-Object { $_.Name -notmatch $ExcludePattern })
                $rawOnly = ($allFiles.Count -gt 0 -and $filesToMove.Count -eq 0)

                if ($rawOnly) { $rawOnlyHit = $true }
                if (-not $filesToMove -and -not $rawOnly) { continue }

                $destPath = Join-Path $DestinationRoot ("_{0}\_{1}\_{2}\_{3}" -f $w.Name, $alt.Name, $rot.Name, $typ.Name)

                if ($filesToMove) {
                    if ($Plan) {
                        $filesToMove | ForEach-Object { Add-Output ("[Plan] {0} -> {1}" -f $_.FullName, $destPath) }
                    } else {
                        $filesToMove | ForEach-Object {
                            $moveQueue.Add([pscustomobject]@{
                                Source = $_.FullName
                                Dest   = $destPath
                            })
                        }
                    }
                }

                $summary += [pscustomobject]@{
                    Weight = $w.Name
                    Alt    = $alt.Name
                    Rot    = $rot.Name
                    Type   = $typ.Name
                    Moved  = if ($filesToMove) { $filesToMove.Count } else { 0 }
                }
                $pathSummary.Add([pscustomobject]@{
                    Weight = $w.Name
                    Alt    = $alt.Name
                    Rot    = $rot.Name
                    Type   = $typ.Name
                    Source = $typ.FullName
                    Dest   = $destPath
                })
            }
        }
    }
}

if ($summary.Count -eq 0) {
    Add-Output "No files moved (path missing or no files)." -ForceHost
} else {
    if (-not $Plan -and $moveQueue.Count -gt 0) {
        $total = $moveQueue.Count
        $idx = 0
        Write-Host ""   # blank line before inline progress bar
        foreach ($item in $moveQueue) {
            $idx++
            if (-not (Test-Path $item.Dest)) {
                New-Item -ItemType Directory -Path $item.Dest -Force | Out-Null
            }
            Move-Item -Path $item.Source -Destination $item.Dest
            $percent = [int](($idx / $total) * 100)
            $barLength = 30
            $filled = [int]($barLength * $percent / 100)
            $bar = ('|' * $filled) + ('-' * ($barLength - $filled))
            Write-Host ("`r[{0}] {1,3}% ({2}/{3})" -f $bar, $percent, $idx, $total) -NoNewline
        }
        Write-Host ""  # finalize progress line
    }

    Add-Output ""
    Add-Output "Move summary:"
    $table = ($summary | Sort-Object Weight, Alt, Rot, Type | Select-Object Weight, Alt, Rot, Type, Moved | Format-Table -AutoSize | Out-String).TrimEnd()
    Add-Output $table
    Add-Output ""

    Add-Output "Source -> Destination:"
    $pathSummary |
        Sort-Object Weight, Alt, Rot, Type |
        ForEach-Object {
            Add-Output ('"{0}" -> "{1}"' -f $_.Source, $_.Dest)
        }
    Add-Output ""

    $totalMoved = ($summary.Moved | Measure-Object -Sum).Sum
    if ($totalMoved -eq 0 -and $rawOnlyHit) {
        Add-Output "(only _raw_ files were found)"
    }
}

Flush-Log
