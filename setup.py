from setuptools import setup, find_packages

setup(
    name='TEAMxRadiosondes',  # Name of package
    version='0.1.0',  # Version number
    description='This project aims to investigate the breaking of orographic gravity waves and the vertical distribution of momentum deposition through region of vertical wind shear.',  # Short description
    long_description=open('README.md').read(),  # Long description from README
    long_description_content_type='text/markdown',  # Description content type
    author='Timothy P. Banyard',  # Author's name
    author_email='tpb38@bath.ac.uk',  # Author's email
    url='https://github.com/tbanyard/TEAMxRadiosondes_Public',  # Project URL
    packages=find_packages(where='src'),  # Automatically find packages in src
    package_dir={'': 'src'},  # Root package directory
    include_package_data=True,  # Include package data specified in MANIFEST.in
    install_requires=[
        # List of project dependencies
        # 'package_name>=version',
        'numpy>=1.18.0',
        'pandas>=1.0.0',
    ],
    extras_require={
        'dev': [
            'xarray==2025.4.0',
            'pandas==2.2.3',
            'numpy==2.0.1',
            'matplotlib==3.10.0',
            # Other development dependencies
        ],
    },
    entry_points={
        'console_scripts': [
            # Command-line interface (CLI) entry points
            # 'command_name=module:function',
        ],
    },
    classifiers=[
        # Classifiers help users find your project by categorizing it
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10, <3.12',  # Python version requirement
)
