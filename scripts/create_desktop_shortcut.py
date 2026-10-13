"""Generate a custom leaf icon and create a Desktop shortcut for the app."""

import math
import os
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageDraw


def generate_icon(icon_path: Path):
    """Draw a clean, modern plant leaf icon and save as .ico."""
    size = 256
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # 1. Soft glowing outer circular badge
    center = size // 2
    radius = 118
    # Dark modern forest green circle background
    draw.ellipse(
        [center - radius, center - radius, center + radius, center + radius],
        fill=(22, 60, 38, 255),
        outline=(52, 168, 83, 255),
        width=5,
    )

    # Inner subtle ring
    draw.ellipse(
        [center - radius + 8, center - radius + 8, center + radius - 8, center + radius - 8],
        outline=(30, 95, 55, 180),
        width=2,
    )

    # 2. Draw leaf shape using bezier-like points
    # Leaf tip at top-right (185, 65), base at bottom-left (75, 185)
    leaf_points = []
    # Right curve
    for t in range(50):
        frac = t / 49.0
        # Curve out towards (195, 175)
        x = (1 - frac) ** 2 * 75 + 2 * (1 - frac) * frac * 200 + frac**2 * 185
        y = (1 - frac) ** 2 * 185 + 2 * (1 - frac) * frac * 175 + frac**2 * 65
        leaf_points.append((x, y))
    # Left curve returning to base
    for t in range(50):
        frac = t / 49.0
        # Curve in towards (65, 80)
        x = (1 - frac) ** 2 * 185 + 2 * (1 - frac) * frac * 70 + frac**2 * 75
        y = (1 - frac) ** 2 * 65 + 2 * (1 - frac) * frac * 80 + frac**2 * 185
        leaf_points.append((x, y))

    # Fill leaf with vibrant emerald green
    draw.polygon(leaf_points, fill=(68, 207, 110, 255))

    # Leaf center vein
    draw.line([(75, 185), (185, 65)], fill=(18, 92, 45, 230), width=4)

    # Side veins
    vein_branches = [
        ((110, 150), (135, 135)),
        ((130, 130), (155, 115)),
        ((150, 110), (170, 95)),
        ((110, 150), (95, 130)),
        ((130, 130), (115, 110)),
        ((150, 110), (135, 90)),
    ]
    for start, end in vein_branches:
        draw.line([start, end], fill=(25, 110, 55, 200), width=3)

    # Small stem at bottom
    draw.line([(75, 185), (55, 205)], fill=(52, 168, 83, 255), width=5)

    # Save multiple resolutions in .ico format
    icon_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        str(icon_path),
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"Icon created successfully: {icon_path}")


def create_shortcut(icon_path: Path):
    project_root = Path(__file__).resolve().parent.parent
    bat_path = project_root / "Launch_App.bat"
    working_dir = project_root

    # Get User Desktop
    desktop_dir = Path(os.environ["USERPROFILE"]) / "Desktop"
    shortcut_path = desktop_dir / "Plant Disease Classifier.lnk"

    ps_script = f"""
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
    $Shortcut.TargetPath = "{bat_path}"
    $Shortcut.WorkingDirectory = "{working_dir}"
    $Shortcut.IconLocation = "{icon_path}"
    $Shortcut.Description = "Plant Disease Classifier - AI Screening App"
    $Shortcut.WindowStyle = 1
    $Shortcut.Save()
    """

    result = subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error creating shortcut: {result.stderr}")
    else:
        print(f"Desktop shortcut created: {shortcut_path}")


if __name__ == "__main__":
    app_icon = Path(__file__).resolve().parent.parent / "desktop_app" / "run" / "app_icon.ico"
    generate_icon(app_icon)
    create_shortcut(app_icon)
