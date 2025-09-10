#!/usr/bin/env python3
"""
Build script for Infinite Rouge executable
"""
import os
import subprocess
import sys

def build_executable():
    """Build the executable using PyInstaller"""
    
    print("🔨 Building Infinite Rouge executable...")
    
    # PyInstaller command with optimized settings
    cmd = [
        "pyinstaller",
        "--onefile",                    # Single executable file
        "--windowed",                   # No console window (remove if you want debug console)
        "--name=InfiniteRouge",         # Executable name
        "--icon=icon.ico",              # Icon file (optional, create if you have one)
        "--add-data=constants.py;.",    # Include constants file
        "--hidden-import=pygame",       # Ensure pygame is included
        "--hidden-import=numpy",        # Ensure numpy is included
        "--hidden-import=requests",     # Ensure requests is included
        "--distpath=dist",              # Output directory
        "--workpath=build",             # Temporary build directory
        "--clean",                      # Clean previous builds
        "main.py"                       # Main script
    ]

    # Bundle models folder if it exists
    if os.path.isdir("models"):
        cmd.insert(-1, "--add-data=models;models")
    
    # Remove icon parameter if icon.ico doesn't exist
    if not os.path.exists("icon.ico"):
        cmd.remove("--icon=icon.ico")
        print("ℹ️  No icon.ico found, building without custom icon")
    
    try:
        # Run PyInstaller
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Build successful!")
        print(f"📦 Executable created: dist/InfiniteRouge.exe")
        print("\nYou can now create a desktop shortcut to dist/InfiniteRouge.exe")
        
    except subprocess.CalledProcessError as e:
        print("❌ Build failed!")
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error output: {e.stderr}")
        return False
    
    return True

if __name__ == "__main__":
    build_executable() 