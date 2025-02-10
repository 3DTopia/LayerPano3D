from setuptools import setup, find_packages, Extension
import pathlib

import numpy
import glob
import os
import shutil
import sys

python_version = str(sys.version_info.major) + str(sys.version_info.minor)

#----include dynamic link libraries & build options----
dll_filepath_list = []
extra_compile_args_ = []
extra_link_args_ = []
dll_src_dir = None

if os.name == 'nt':
    # 3rd-party libs
    opencv_include_dir = "D:/libraries_windows/opencv/opencv-4.5.3-vc14_vc15/build/include/"
    opencv_lib_dir = "D:/libraries_windows/opencv/opencv-4.5.3-vc14_vc15/build/x64/vc15/lib/"
    opencv_lib_files = ['opencv_world453', 'depth_stitch']
    depthmap_align_lib_dir = '../lib/Release/'
    dll_filepath_list.append('../bin/Release/depth_stitch.dll')
    dll_src_dir = "./dll/*.dll"
    dist_dll_ext = "*.dll"
    # generate *.pdb for MSVC debugging
    extra_compile_args_ = ["/Zi"]
    extra_link_args_ = ["/DEBUG", "/OPT:REF", "/OPT:ICF"]
elif os.name == 'Linux' or os.name == 'posix':
    opencv_include_dir = "/usr/include/opencv4/"
    opencv_lib_dir = "/usr/lib/x86_64-linux-gnu/"
    opencv_lib_files = ['opencv_core', 'opencv_imgproc', 'depth_stitch']
    depthmap_align_lib_dir = '../lib/'
    dll_filepath_list.append('../lib/libdepth_stitch.so')
    dll_filepath_list.append('../lib/EigenSolvers.cpython-{}-x86_64-linux-gnu.so'.format(python_version))
    dll_src_dir = "./so/*"
    dist_dll_ext = "*.so*"
    extra_compile_args_ = ["-O3", "-DNDEBUG"]
    extra_link_args_ = ["-Wl,-rpath=$ORIGIN/"]
else:
    msg = "System {} do not suport.".format(os.name)
    raise RuntimeError(msg)

if len(dll_filepath_list) == 0:
    print("Warning: Do not find dependent dynamic library.")

# clean & copy dynamic link libs
dll_tar_dir = "./instaOmniDepth/"
for file in glob.glob(dll_tar_dir + dist_dll_ext):
    os.remove(file)
for file in glob.glob(dll_src_dir):
    dll_filepath_list.append(file)
for item in dll_filepath_list:
    shutil.copy(item, dll_tar_dir)

print("Package adding the following dynamic libraries: \n {}".format(dll_filepath_list))

#----define the extension module----
depthmapAlignExt = Extension('instaOmniDepth.depthmapAlign',
                            sources=['./instaOmniDepth/depthmapAlignModule.cpp'],
                            include_dirs=[numpy.get_include(), '../include/', opencv_include_dir],
                            library_dirs=[opencv_lib_dir, depthmap_align_lib_dir],
                            extra_compile_args=extra_compile_args_,
                            extra_link_args=extra_link_args_,
                            libraries=opencv_lib_files)

#----generate the package----
# Get the long description from the README file
pwd_dir_path = pathlib.Path(__file__).parent.resolve()
long_description = (pwd_dir_path / 'README.md').read_text(encoding='utf-8')
setup(
    name='instaOmniDepth',
    version='0.1.0',
    description='Align sub-image depth maps',
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_dir={'': '.'},
    packages=find_packages(where='.'),
    ext_modules=[depthmapAlignExt],
    #package_dir={'instaOmniDepth': 'depthmapAlignPackage'},
    install_requires=['numpy'],
    package_data={
        'instaOmniDepth': [dist_dll_ext],
    },
)
