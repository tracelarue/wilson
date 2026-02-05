from setuptools import setup

package_name = 'ir_signal_action'

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
    description='Action server to trigger an IR carrier burst for the mini fridge.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ir_signal_action_server = ir_signal_action.ir_signal_action_server:main',
        ],
    },
)
