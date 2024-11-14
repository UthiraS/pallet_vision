from setuptools import find_packages, setup
from glob import glob  # Changed from os to glob
import os

package_name = 'pallet'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', glob('models/*.pt')),
        ('share/' + package_name + '/models', glob(package_name + '/final_model_checkpoints/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='uthira',
    maintainer_email='uthiralakshmi6@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publish_test_images = pallet.publishImage:main',
            'infer = pallet.inference:main'
        ],
    },
)