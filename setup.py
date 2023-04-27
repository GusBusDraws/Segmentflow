from setuptools import setup, find_packages

setup(
    name="segmentflow",
    version="0.0.1",
    install_requires=[
        'imageio >= 2.21.0',
        'matplotlib',
        'numpy',
        'numpy-stl',
        'pandas',
        'PyYAML',
        'scikit-image >= 0.19.3',
        'scipy',
    ],
    packages=find_packages(),
)

