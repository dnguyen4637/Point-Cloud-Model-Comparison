extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': [
        '-O3',
        '--allow-unsupported-compiler',
        '-Wno-deprecated-gpu-targets',
        '-Xcompiler', '-fPIC'
    ]
}

