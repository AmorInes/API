# -*- mode: python ; coding: utf-8 -*-
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = ['scikit-learn']

# Collecter les dépendances pour xgboost
tmp_ret = collect_all('xgboost')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Collecter les dépendances pour lightgbm
tmp_ret = collect_all('lightgbm')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Collecter les dépendances pour json
tmp_ret = collect_all('json')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Inclure le fichier distributed.yaml
datas.append(('c:/users/predict/api/env/lib/site-packages/distributed/distributed.yaml', 'distributed'))


a = Analysis(
    ['main_test.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='API_BOOPER_V2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main_test',
)