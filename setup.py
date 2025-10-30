from setuptools import setup, find_packages

setup(
    name='micrograd',
    version='0.1.0',
    description='A small autograd engine and neural network library',
    author='Micrograd Team',
    author_email='example@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0,<2.0.0',
        'flask>=2.0.0,<3.0.0',
        'torch>=1.9.0,<3.0.0',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
