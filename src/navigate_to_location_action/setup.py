from setuptools import setup

package_name = 'navigate_to_location_action'

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
    maintainer='trace',
    maintainer_email='trace@example.com',
    description='Action server for navigating to locations with simple interface',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'navigate_to_location_server = navigate_to_location_action.navigate_to_location_server:main',
        ],
    },
)
