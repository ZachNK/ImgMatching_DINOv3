# ================== Configuration ==================
param([string]$Code, [switch]$List)
# Base directory (exports root \ embed type root \ datasets root \ weights root)
$base = "D:\dinov3_exports\dinov3_query_embeds\shinsung_data"  

# Indices (1-based) for selecting alt, rot, typ
$wgtIndex = 2
$altIndex = 3
$rotIndex = 3
$typIndex = 1

if ($Code) {
    $wgtIndex = [int]$Code.Substring(0,1)
    $altIndex = [int]$Code.Substring(1,1)
    $rotIndex = [int]$Code.Substring(2,1)
    $typIndex = [int]$Code.Substring(3,1)
}

# Possible values for alt, rot, typ
$WGT = @(
"vit7b16",
"vitb16",
"vith16+",
"vitl16",
"vits16",
"vits16+"
"cxBase",
"cxLarge",
"cxSmall",
"cxTiny"   
"vit7b16sat",
"vitl16sat"
)
$ALT = @("100", "150", "200", "300", "400")
$ROT = @("045", "090", "135", "180")
$TYP = @("DenseFT", "GlobalToken", "PatchGrid", "PatchToken")

if ($List) {
    Write-Host "===== WGT (1-based index) ====="
    for ($i = 0; $i -lt $WGT.Count; $i++) {
        Write-Host ("{0}: {1}" -f ($i+1), $WGT[$i])
    }

    Write-Host "`n===== ALT (1-based index) ====="
    for ($i = 0; $i -lt $ALT.Count; $i++) {
        Write-Host ("{0}: {1}" -f ($i+1), $ALT[$i])
    }

    Write-Host "`n===== ROT (1-based index) ====="
    for ($i = 0; $i -lt $ROT.Count; $i++) {
        Write-Host ("{0}: {1}" -f ($i+1), $ROT[$i])
    }

    Write-Host "`n===== TYP (1-based index) ====="
    for ($i = 0; $i -lt $TYP.Count; $i++) {
        Write-Host ("{0}: {1}" -f ($i+1), $TYP[$i])
    }

    return   # 리스트만 보여주고 나머지 이동 로직은 실행 안 함
}

# ===================== define paths ==================
$wgt = $WGT[$wgtIndex-1]    # wgtIndex = 2 -> "vitb16"
$alt = $ALT[$altIndex-1]    # altIndex = 3 -> "200"
$rot = $ROT[$rotIndex-1]    # rotIndex = 2 -> "090"
$typ = $TYP[$typIndex-1]    # typIndex = 2 -> "GlobalToken"

$wgtDest = "_$wgt"      
$altDest = "_$alt" # "_200"
$rotDest = "_$rot" # "_090"
$typDest = "_$typ" # "_GlobalToken"

$src = "$base\$wgt\$alt\$rot\$typ"
$dest = "$base\$wgtDest\$altDest\$rotDest\$typDest"

# src D:\dinov3_exports\dinov3_query_embeds\shinsung_data\vitb16\200\090\GlobalToken   
# dest D:\dinov3_exports\dinov3_query_embeds\shinsung_data\_vitb16\_200\_090\_GlobalToken

# ===================== define keywords ==================

# multiple keywords to move (OR condition)
$keywords = @("subsample_sub2", "subsample_sub4", "subsample_sub8")

# ===================== Move files =====================

# 1) file count before move
$srcBefore  = (Get-ChildItem -Path $src  -File | Measure-Object).Count
$destBefore = (Get-ChildItem -Path $dest -File | Measure-Object).Count

# 2) keyword OR condition (regex) pattern
$pattern = ($keywords -join "|")    # "보고서|DINOv2|2025" 이런 형태

# 3) move files matching any keyword
Get-ChildItem -Path $src -File |
    Where-Object { $_.Name -match $pattern } |
    Move-Item -Destination $dest

# 4) file count after move
$srcAfter  = (Get-ChildItem -Path $src  -File | Measure-Object).Count
$destAfter = (Get-ChildItem -Path $dest -File | Measure-Object).Count

# 5) print only two lines of results
Write-Host "$src  :  $srcBefore  ->  $srcAfter"
Write-Host "$dest :  $destBefore ->  $destAfter"