from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'myaction'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'action'), glob('action/*.action')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='trace',
    maintainer_email='traceglarue@gmail.com',
    description='Simple countdown action demonstration',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'countdown_server = myaction.countdown_server:main',
            'countdown_client = myaction.countdown_client:main',
        ],
    },
)
