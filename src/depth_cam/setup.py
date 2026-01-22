from setuptools import setup

package_name = 'depth_cam'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='arducam',
    author_email='dennis@arducam.com',
    maintainer='arducam',
    maintainer_email='dennis@arducam.com',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='Arducam Tof Camera Examples.',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tof_pointcloud = depth_cam.tof_pointcloud:main',
            'depth_field = depth_cam.depth_field:main',
            'sim_depth = depth_cam.sim_depth:main',
        ],
    },
)
