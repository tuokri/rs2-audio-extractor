# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

binaries = [
    (r"C:\Program Files (x86)\Windows Kits\10\Redist\10.0.18362.0\ucrt\DLLs\x64", "."),
    (r".\bin", "bin"),
]

a = Analysis(["extract.py"],
             pathex=[],
             binaries=binaries,
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='extract',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          )
