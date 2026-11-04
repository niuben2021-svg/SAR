param([string]$OutputDirectory = (Join-Path ([Environment]::GetFolderPath('Desktop')) 'SAR-VSCode-Offline'))
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$kit = Split-Path $PSScriptRoot -Parent
$manifest = Get-Content (Join-Path $kit 'downloads.json') -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
foreach ($item in $manifest.files) {
    $target = Join-Path $OutputDirectory $item.name
    if (!(Test-Path $target) -or (Get-FileHash $target -Algorithm SHA256).Hash.ToLower() -ne $item.sha256) {
        Write-Host "Downloading $($item.name)"
        Invoke-WebRequest -UseBasicParsing -Uri $item.url -OutFile $target
        if ($item.name.EndsWith('.vsix')) {
            $inputStream = [IO.File]::OpenRead($target)
            $first = $inputStream.ReadByte(); $second = $inputStream.ReadByte()
            $inputStream.Dispose()
            if ($first -eq 31 -and $second -eq 139) {
                $inputStream = [IO.File]::OpenRead($target)
                $gzip = New-Object IO.Compression.GZipStream($inputStream, [IO.Compression.CompressionMode]::Decompress)
                $outputStream = [IO.File]::Create("$target.decoded")
                try { $gzip.CopyTo($outputStream) } finally { $outputStream.Dispose(); $gzip.Dispose(); $inputStream.Dispose() }
                Move-Item -Force "$target.decoded" $target
            }
        }
    }
    if ((Get-FileHash $target -Algorithm SHA256).Hash.ToLower() -ne $item.sha256) { throw "SHA256 mismatch: $($item.name)" }
    Write-Host "SHA256 PASS: $($item.name)"
}
$destinationKit = Join-Path $OutputDirectory 'vscode-glm-kit'
if ([IO.Path]::GetFullPath($destinationKit).TrimEnd('\') -ne [IO.Path]::GetFullPath($kit).TrimEnd('\')) {
    New-Item -ItemType Directory -Force -Path $destinationKit | Out-Null
    Copy-Item (Join-Path $kit "*") $destinationKit -Recurse -Force
}
Write-Host "Ready for offline transfer: $OutputDirectory"
