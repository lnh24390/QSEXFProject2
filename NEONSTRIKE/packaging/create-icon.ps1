$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$output = Join-Path $PSScriptRoot 'NeonStrike.ico'
$sizes = @(16, 24, 32, 48, 64, 128, 256)
$images = New-Object System.Collections.Generic.List[byte[]]

foreach ($size in $sizes) {
    $bitmap = New-Object System.Drawing.Bitmap($size, $size, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.Clear([System.Drawing.Color]::Transparent)

    $scale = $size / 256.0
    $graphics.ScaleTransform($scale, $scale)
    $background = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 5, 16, 21))
    $border = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 40, 231, 255), 7)
    $border.LineJoin = [System.Drawing.Drawing2D.LineJoin]::Round
    $graphics.FillRectangle($background, 10, 10, 236, 236)
    $graphics.DrawRectangle($border, 10, 10, 236, 236)

    $gunPath = New-Object System.Drawing.Drawing2D.GraphicsPath
    $points = @(
        (New-Object System.Drawing.PointF(30, 112)), (New-Object System.Drawing.PointF(74, 112)),
        (New-Object System.Drawing.PointF(92, 88)), (New-Object System.Drawing.PointF(165, 88)),
        (New-Object System.Drawing.PointF(183, 101)), (New-Object System.Drawing.PointF(226, 101)),
        (New-Object System.Drawing.PointF(226, 127)), (New-Object System.Drawing.PointF(180, 127)),
        (New-Object System.Drawing.PointF(166, 141)), (New-Object System.Drawing.PointF(132, 141)),
        (New-Object System.Drawing.PointF(124, 194)), (New-Object System.Drawing.PointF(94, 194)),
        (New-Object System.Drawing.PointF(90, 141)), (New-Object System.Drawing.PointF(69, 141)),
        (New-Object System.Drawing.PointF(55, 164)), (New-Object System.Drawing.PointF(32, 164)),
        (New-Object System.Drawing.PointF(45, 127)), (New-Object System.Drawing.PointF(30, 127))
    )
    $gunPath.AddPolygon($points)
    $gunBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 113, 220, 235))
    $shadowBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(220, 4, 15, 19))
    $accentBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 178, 62))
    $redBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 56, 93))
    $graphics.FillPath($gunBrush, $gunPath)
    $graphics.FillPolygon($shadowBrush, @((New-Object System.Drawing.PointF(103, 141)), (New-Object System.Drawing.PointF(132, 141)), (New-Object System.Drawing.PointF(125, 187)), (New-Object System.Drawing.PointF(108, 187))))
    $graphics.FillRectangle($shadowBrush, 77, 101, 91, 12)
    $graphics.FillRectangle($accentBrush, 169, 106, 61, 9)
    $graphics.FillEllipse($redBrush, 51, 113, 17, 17)

    $stream = New-Object System.IO.MemoryStream
    $bitmap.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
    $images.Add($stream.ToArray())
    $stream.Dispose()
    $gunPath.Dispose()
    $gunBrush.Dispose()
    $shadowBrush.Dispose()
    $accentBrush.Dispose()
    $redBrush.Dispose()
    $background.Dispose()
    $border.Dispose()
    $graphics.Dispose()
    $bitmap.Dispose()
}

$file = [System.IO.File]::Open($output, [System.IO.FileMode]::Create)
$writer = New-Object System.IO.BinaryWriter($file)
$writer.Write([uint16]0)
$writer.Write([uint16]1)
$writer.Write([uint16]$sizes.Count)
$offset = 6 + (16 * $sizes.Count)
for ($i = 0; $i -lt $sizes.Count; $i++) {
    $size = $sizes[$i]
    $writer.Write([byte]$(if ($size -eq 256) { 0 } else { $size }))
    $writer.Write([byte]$(if ($size -eq 256) { 0 } else { $size }))
    $writer.Write([byte]0)
    $writer.Write([byte]0)
    $writer.Write([uint16]1)
    $writer.Write([uint16]32)
    $writer.Write([uint32]$images[$i].Length)
    $writer.Write([uint32]$offset)
    $offset += $images[$i].Length
}
foreach ($image in $images) { $writer.Write($image) }
$writer.Dispose()
$file.Dispose()
Write-Output "Created $output"
