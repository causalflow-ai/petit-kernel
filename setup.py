# setup.py
import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # A test-only module may be left behind by an explicit internal build.
        # Never let a stale copy leak into a public wheel.
        for filename in ("_test_ops.so", "_test_ops.pyd"):
            test_module = os.path.join(extdir, filename)
            if os.path.isfile(test_module):
                os.remove(test_module)

        cfg = "Debug" if self.debug else "Release"
        output_name = os.path.basename(self.get_ext_fullpath(ext.name))
        
        # Set Python_EXECUTABLE instead of Python_INCLUDE_DIR etc.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DLIBRARY_OUTPUT_NAME={os.path.splitext(output_name)[0]}",
            "-DWITH_ROCM=ON",
            "-DWITH_PYTHON=ON",
            "-GNinja",
        ]

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "petit_kernels"],
            cwd=build_temp,
        )

setup(
    name="petit_kernel",
    ext_modules=[CMakeExtension("petit_kernel.ops")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
