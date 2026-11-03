from setuptools import setup, Extension
import os

openmm_dir = '@OPENMM_DIR@'
header_dir = '@PGM_HEADER_DIR@'
library_dir = '@PGM_LIBRARY_DIR@'

extension = Extension(name='_openmmpgm',
                      sources=['PGMPluginWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMPGM'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), library_dir],
                      runtime_library_dirs=[os.path.join(openmm_dir, 'lib')],
                      extra_compile_args=['-std=c++17'],
                      )

setup(name='openmmpgm',
      version='0.1',
      py_modules=['openmmpgm', 'pgm_prmtop'],
      ext_modules=[extension],
      )
