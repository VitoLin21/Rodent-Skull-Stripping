from setuptools import setup, find_namespace_packages

setup(
    name='RS2',
    version='1.0',
    description='deep learn based skull stripping tool for rodent fmri',
    author='Yongkang Lin',
    license=' GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007',
    python_requires=">=3.9",
    packages=find_namespace_packages(
        include=['RS2', 'RS2.*']),
    install_requires=[
        "torch==2.0.0",
        'monai>=1.1.0',
        "nibabel",
        "tqdm",
        "einops",
        "numpy",
        "tifffile",
        "batchgenerators",
        "simpleitk",
        "scipy",
        "imageio",
        "pandas",
        "acvl_utils"
    ],
    entry_points={
        'console_scripts': [
            'RS2_predict = RS2.inference.predict:predict_entry_point',  # api available
        ],
    },
    keywords=['deep learning', 'image segmentation',
              'medical image segmentation', 'mouse brain', 'rat brain']
)
