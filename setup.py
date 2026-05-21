"""Setup script for Windows executable build"""
from setuptools import setup
import sys

if sys.platform == 'win32':
    from cx_Freeze import Executable
    import os
    
    # Check if logo exists
    icon_path = "assets/logo.ico"
    if not os.path.exists(icon_path):
        icon_path = None
    
    executables = [
        Executable(
            "main.py",
            base="Win32GUI",  # No console window
            target_name="VocalPitchMonitor.exe",
            icon=icon_path,
            shortcut_name="Vocal Pitch Monitor",
            shortcut_dir="DesktopFolder",
        )
    ]
    
    setup(
        name="Vocal Pitch Monitor",
        version="2.7",
        description="Professional vocal pitch monitoring tool",
        options={
            "build_exe": {
                "packages": ["numpy", "soundfile", "sounddevice", "aubio", 
                            "pyqtgraph", "PyQt6", "pydub"],
                "include_files": [("assets/", "assets/")],
                "optimize": 2,
            }
        },
        executables=executables
    )
else:
    print("This setup script is for Windows only")
