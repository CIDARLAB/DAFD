# -*- mode: python -*-

block_cipher = None


a = Analysis(['DAFD_GUI.py'],
             pathex=['/Users/CIDARLAB/Desktop/DAFD'],
             binaries=[],
             datas=[('MicroFluidics_Random.csv','.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='DAFD_GUI',
          debug=False,
          strip=False,
          upx=True,
          console=False )
app = BUNDLE(exe,
             name='DAFD_GUI.app',
             icon=None,
             bundle_identifier=None)
