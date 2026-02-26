import os
from glob import glob

from setuptools import setup

package_name = "wilson_teleop_gui"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="trace",
    maintainer_email="traceglarue@gmail.com",
    description="Simple Tkinter teleop GUI for differential drive robots.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "teleop_gui_node = wilson_teleop_gui.teleop_gui_node:main",
        ],
    },
)
