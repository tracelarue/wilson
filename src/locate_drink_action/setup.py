from setuptools import setup
from glob import glob
import os

package_name = 'locate_drink_action'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='trace',
    maintainer_email='trace@example.com',
    description='ROS2 action server for locating drinks using Gemini AI and positioning the robot optimally',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'locate_drink_action_server = locate_drink_action.locate_drink_action_server:main',
        ],
    },
)
