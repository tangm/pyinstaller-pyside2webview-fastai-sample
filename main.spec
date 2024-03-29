# -*- mode: python ; coding: utf-8 -*-

# https://github.com/pyinstaller/pyinstaller/issues/4064#issuecomment-471063543
import distutils
if (getattr(distutils, 'distutils_path', None) != None) and distutils.distutils_path.endswith('__init__.py'):
    distutils.distutils_path = os.path.dirname(distutils.distutils_path)

block_cipher = None


a = Analysis(['main.py'],
             pathex=['C:\\Users\\mtan\\projects\\pyinstaller-pyside2webview-sample'],
             binaries=[],
             datas=[('data', 'data')],
             hiddenimports=['PySide2.QtPrintSupport', 'numpy.random.common', 'numpy.random.bounded_integers', 'numpy.random.entropy', 'fastprogress'],
             hookspath=['hooks'],
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
          [],
          exclude_binaries=True,
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='main')
