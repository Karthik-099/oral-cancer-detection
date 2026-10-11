"""
Setup script for building OralAID Windows executable
Run: python setup_windows.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def create_windows_executable():
    """Create Windows executable using PyInstaller"""
    
    print("🚀 Building OralAID Windows Executable...")
    
    # Ensure PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyInstaller"])
    
    # Create spec file for PyInstaller
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('models', 'models'),
    ],
    hiddenimports=[
        'torch',
        'torchvision',
        'segmentation_models_pytorch',
        'pytorch_grad_cam',
        'lime',
        'sklearn',
        'PIL',
        'cv2',
        'numpy',
        'matplotlib',
        'tkinter',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='OralAID',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='static/images/oralaid_icon.ico'
)
"""
    
    # Write spec file
    with open('oralaid.spec', 'w') as f:
        f.write(spec_content)
    
    # Create icon if it doesn't exist
    icon_path = Path('static/images/oralaid_icon.ico')
    if not icon_path.exists():
        icon_path.parent.mkdir(parents=True, exist_ok=True)
        # Create a simple icon (you can replace this with a proper icon)
        print("⚠️  Icon file not found. Please add oralaid_icon.ico to static/images/")
    
    # Build executable
    try:
        print("🔨 Building executable...")
        subprocess.check_call([
            sys.executable, "-m", "PyInstaller", 
            "--clean", 
            "--onefile",
            "--windowed",
            "oralaid.spec"
        ])
        
        print("✅ Build completed successfully!")
        print("📁 Executable location: dist/OralAID.exe")
        
        # Create installer directory
        installer_dir = Path('installer')
        installer_dir.mkdir(exist_ok=True)
        
        # Copy executable
        if Path('dist/OralAID.exe').exists():
            shutil.copy2('dist/OralAID.exe', installer_dir / 'OralAID.exe')
        
        # Create installer script
        create_installer_script(installer_dir)
        
        print("🎉 OralAID Windows executable is ready!")
        print(f"📦 Installation package: {installer_dir.absolute()}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False
    
    return True

def create_installer_script(installer_dir):
    """Create NSIS installer script"""
    
    nsis_script = f"""
; OralAID Installer Script
!define APPNAME "OralAID"
!define COMPANYNAME "OralAID Team"
!define DESCRIPTION "AI-Powered Oral Cancer Detection System"
!define VERSIONMAJOR 1
!define VERSIONMINOR 0
!define VERSIONBUILD 0

!define HELPURL "https://github.com/oralaid/oralaid"
!define UPDATEURL "https://github.com/oralaid/oralaid/releases"
!define ABOUTURL "https://oralaid.com"

!define INSTALLSIZE 500000  ; Estimate in KB

RequestExecutionLevel admin

InstallDir "$PROGRAMFILES\\${{APPNAME}}"

Name "${{APPNAME}}"
Icon "oralaid_icon.ico"
outFile "OralAID_Setup.exe"

!include LogicLib.nsh

page components
page directory
page instfiles

!macro VerifyUserIsAdmin
UserInfo::GetAccountType
pop $0
${{If}} $0 != "admin"
    messageBox mb_iconstop "Administrator rights required!"
    setErrorLevel 740
    quit
${{EndIf}}
!macroend

function .onInit
    setShellVarContext all
    !insertmacro VerifyUserIsAdmin
functionEnd

section "install"
    setOutPath $INSTDIR
    
    file "OralAID.exe"
    file /nonfatal "oralaid_icon.ico"
    
    writeUninstaller "$INSTDIR\\uninstall.exe"
    
    createDirectory "$SMPROGRAMS\\${{APPNAME}}"
    createShortCut "$SMPROGRAMS\\${{APPNAME}}\\${{APPNAME}}.lnk" "$INSTDIR\\OralAID.exe" "" "$INSTDIR\\oralaid_icon.ico"
    createShortCut "$DESKTOP\\${{APPNAME}}.lnk" "$INSTDIR\\OralAID.exe" "" "$INSTDIR\\oralaid_icon.ico"
    
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "DisplayName" "${{APPNAME}} - ${{DESCRIPTION}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "UninstallString" "$\\"$INSTDIR\\uninstall.exe$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "QuietUninstallString" "$\\"$INSTDIR\\uninstall.exe$\\" /S"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "InstallLocation" "$\\"$INSTDIR$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "DisplayIcon" "$\\"$INSTDIR\\oralaid_icon.ico$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "Publisher" "${{COMPANYNAME}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "HelpLink" "${{HELPURL}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "URLUpdateInfo" "${{UPDATEURL}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "URLInfoAbout" "${{ABOUTURL}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "DisplayVersion" "${{VERSIONMAJOR}}.${{VERSIONMINOR}}.${{VERSIONBUILD}}"
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "VersionMajor" ${{VERSIONMAJOR}}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "VersionMinor" ${{VERSIONMINOR}}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "NoModify" 1
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "NoRepair" 1
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}" "EstimatedSize" ${{INSTALLSIZE}}
sectionEnd

section "uninstall"
    delete "$INSTDIR\\OralAID.exe"
    delete "$INSTDIR\\oralaid_icon.ico"
    delete "$INSTDIR\\uninstall.exe"
    
    delete "$SMPROGRAMS\\${{APPNAME}}\\${{APPNAME}}.lnk"
    rmDir "$SMPROGRAMS\\${{APPNAME}}"
    delete "$DESKTOP\\${{APPNAME}}.lnk"
    
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{APPNAME}}"
    
    rmDir $INSTDIR
sectionEnd
"""
    
    # Write NSIS script
    with open(installer_dir / 'installer.nsi', 'w') as f:
        f.write(nsis_script)
    
    # Create README
    readme_content = """
# OralAID Windows Installation Package

## Contents
- OralAID.exe: Main application executable
- installer.nsi: NSIS installer script (optional)

## Installation
1. Double-click OralAID.exe to run the application directly
2. Or use the NSIS installer script to create a proper Windows installer

## System Requirements
- Windows 10 or later (64-bit)
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Graphics card with DirectX 11 support (for GPU acceleration)

## Features
- AI-powered oral cancer detection
- Explainable AI visualizations (GradCAM, LIME)
- Lesion segmentation
- Clinical recommendations
- Offline operation
- Export reports and images

## Support
For technical support, please visit: https://github.com/oralaid/oralaid

## License
This software is provided for clinical research and educational purposes.
Please ensure compliance with local medical device regulations.
"""
    
    with open(installer_dir / 'README.txt', 'w') as f:
        f.write(readme_content)
    
    print(f"📝 Installer files created in {installer_dir}")

def create_batch_scripts():
    """Create convenient batch scripts"""
    
    # Run web app script
    web_script = """@echo off
echo Starting OralAID Web Application...
python app.py
pause
"""
    
    with open('run_web_app.bat', 'w') as f:
        f.write(web_script)
    
    # Build executable script
    build_script = """@echo off
echo Building OralAID Windows Executable...
python setup_windows.py
pause
"""
    
    with open('build_executable.bat', 'w') as f:
        f.write(build_script)
    
    print("📝 Batch scripts created:")
    print("  - run_web_app.bat: Start web application")
    print("  - build_executable.bat: Build Windows executable")

if __name__ == "__main__":
    print("🏗️  OralAID Setup Utility")
    print("=" * 50)
    
    choice = input("Choose option:\n1. Build Windows Executable\n2. Create Batch Scripts\n3. Both\nEnter choice (1-3): ")
    
    if choice in ['1', '3']:
        success = create_windows_executable()
        if not success:
            sys.exit(1)
    
    if choice in ['2', '3']:
        create_batch_scripts()
    
    print("\n✅ Setup completed successfully!")
    print("🎯 Next steps:")
    print("  1. Test the web application: python app.py")
    print("  2. Test the desktop app: python desktop_app.py")
    print("  3. Use the Windows executable from the installer/ directory")