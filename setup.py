# from future.utils import iteritems
import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


def find_in_path(name, path):
    """Find a file in a search path"""

    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, '
                'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    self.src_extensions.append('.cu')

    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', CUDA['nvcc'])
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile




class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)



CUDA = locate_cuda()
print(CUDA)
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


exts = [Extension('roc_auc_pairwise.utils',
        sources = ['roc_auc_pairwise/utils_cython.pyx', 'src/cpu/utils.cpp'],
        library_dirs = [CUDA['lib64']],
        libraries = ['cudart'],
        language = 'c++',
        runtime_library_dirs = [CUDA['lib64']],
        extra_compile_args= {
            'gcc': ['-fopenmp', '-lstdc++', '-std=c++11' ],
            'nvcc': ['--ptxas-options=-v', '-c',
                '--compiler-options', "'-fPIC'"
                ]
            },
        extra_link_args= ['-fopenmp', '-lstdc++'],
        include_dirs = [numpy_include, CUDA['include'], 'src']),
        
        Extension('roc_auc_pairwise.deltaauc_cpu',
        sources = ['roc_auc_pairwise/deltaauc_cpu_cython.pyx', 'src/cpu/deltaauc.cpp'],
        library_dirs = [CUDA['lib64']],
        libraries = ['cudart'],
        language = 'c++',
        runtime_library_dirs = [CUDA['lib64']],
        extra_compile_args= {
            'gcc': ['-fopenmp', '-lstdc++',  '-std=c++11' ],
            'nvcc': ['--ptxas-options=-v', '-c',
                '--compiler-options', "'-fPIC'"
                ]
            },
        extra_link_args= ['-fopenmp', '-lstdc++'],
        include_dirs = [numpy_include, CUDA['include'], 'src']),
        
        Extension('roc_auc_pairwise.sigmoid_pairwise_cpu',
        sources = ['roc_auc_pairwise/sigmoid_pairwise_cpu_cython.pyx', 'src/cpu/sigmoid_pairwise.cpp'],
        library_dirs = [CUDA['lib64']],
        libraries = ['cudart'],
        language = 'c++',
        runtime_library_dirs = [CUDA['lib64']],
        extra_compile_args= {
            'gcc': ['-fopenmp', '-lstdc++',  '-std=c++11' ],
            'nvcc': ['--ptxas-options=-v', '-c',
                '--compiler-options', "'-fPIC'"
                ]
            },
        extra_link_args= ['-fopenmp', '-lstdc++'],
        include_dirs = [numpy_include, CUDA['include'], 'src']),
        
        Extension('roc_auc_pairwise.sigmoid_pairwise_auc_cpu',
        sources = ['roc_auc_pairwise/sigmoid_pairwise_auc_cpu_cython.pyx', 'src/cpu/sigmoid_pairwise_auc.cpp'],
        library_dirs = [CUDA['lib64']],
        libraries = ['cudart'],
        language = 'c++',
        runtime_library_dirs = [CUDA['lib64']],
        extra_compile_args= {
            'gcc': ['-fopenmp', '-lstdc++',  '-std=c++11' ],
            'nvcc': ['--ptxas-options=-v', '-c',
                '--compiler-options', "'-fPIC'"
                ]
            },
        extra_link_args= ['-fopenmp', '-lstdc++'],
        include_dirs = [numpy_include, CUDA['include'], 'src']),
        
        Extension('roc_auc_pairwise.deltaauc_gpu',
        sources = ['roc_auc_pairwise/deltaauc_gpu_cython.pyx', 'src/cuda/deltaauc_kernels.cu', 'src/cuda/deltaauc.cu'],
        library_dirs = [CUDA['lib64']],
        libraries = ['cudart'],
        language = 'c++',
        runtime_library_dirs = [CUDA['lib64']],
        extra_compile_args= {
            'gcc': ['-fopenmp', '-lstdc++',  '-std=c++11' ],
            'nvcc': ['--ptxas-options=-v', '-c',
                '--compiler-options', "'-fPIC'",  '-std=c++11' 
                ]
            },
        extra_link_args= ['-fopenmp', '-lstdc++'],
        include_dirs = [numpy_include, CUDA['include'], 'src']),
        
        Extension('roc_auc_pairwise.sigmoid_pairwise_gpu',
        sources = ['roc_auc_pairwise/sigmoid_pairwise_gpu_cython.pyx', 'src/cuda/sigmoid_pairwise_kernels.cu', 'src/cuda/sigmoid_pairwise.cu'],
        library_dirs = [CUDA['lib64']],
        libraries = ['cudart'],
        language = 'c++',
        runtime_library_dirs = [CUDA['lib64']],
        extra_compile_args= {
            'gcc': ['-fopenmp', '-lstdc++',  '-std=c++11' ],
            'nvcc': ['--ptxas-options=-v', '-c',
                '--compiler-options', "'-fPIC'",  '-std=c++11' 
                ]
            },
        extra_link_args= ['-fopenmp', '-lstdc++'],
        include_dirs = [numpy_include, CUDA['include'], 'src']),
        
        Extension('roc_auc_pairwise.sigmoid_pairwise_auc_gpu',
        sources = ['roc_auc_pairwise/sigmoid_pairwise_auc_gpu_cython.pyx', 'src/cuda/sigmoid_pairwise_auc_kernels.cu', 'src/cuda/sigmoid_pairwise_auc.cu', 'src/cuda/deltaauc.cu', 'src/cpu/utils.cpp'],
        library_dirs = [CUDA['lib64']],
        libraries = ['cudart'],
        language = 'c++',
        runtime_library_dirs = [CUDA['lib64']],
        extra_compile_args= {
            'gcc': ['-fopenmp', '-lstdc++',  '-std=c++11' ],
            'nvcc': ['--ptxas-options=-v', '-c',
                '--compiler-options', "'-fPIC'",  '-std=c++11' 
                ]
            },
        extra_link_args= ['-fopenmp', '-lstdc++'],
        include_dirs = [numpy_include, CUDA['include'], 'src'])
        ]



with open('./README_pypi.md', 'r') as _r:
    long_description = _r.read()



setup(name = 'roc_auc_pairwise',
      author = 'Dmitry Michaylin',
      long_description=long_description,
      long_description_content_type='text/markdown',
      version = '0.0.1',
      ext_modules = exts,
      cmdclass = {'build_ext': custom_build_ext},
      zip_safe = False)
