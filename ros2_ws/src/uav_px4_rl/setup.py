from glob import glob
from setuptools import find_packages, setup


package_name = "uav_px4_rl"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
    ],
    install_requires=["setuptools", "numpy", "gymnasium"],
    zip_safe=True,
    maintainer="uav_px4_rl maintainer",
    maintainer_email="maintainer@example.com",
    description="PX4 SITL and Gazebo online PPO environment for multi-wire 3D LiDAR-feature avoidance.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "offboard_smoke_test = uav_px4_rl.offboard_smoke_test:main",
            "gz_harmonic_bridge = uav_px4_rl.gz_harmonic_bridge:main",
        ],
    },
)
