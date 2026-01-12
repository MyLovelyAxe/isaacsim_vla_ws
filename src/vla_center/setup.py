import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'vla_center'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hardli',
    maintainer_email='lijialei829@gmail.com',
    description='Publish image and current joint state, subscribe to target action command',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'send_observation = vla_center.send_observation:main',
            'send_observation_2cam = vla_center.send_observation_2cam:main',
            'get_action = vla_center.get_action:main',
            'get_action_record = vla_center.get_action_record:main',
            'replay_record = vla_center.replay_record:main',
            'estimate_safety = vla_center.estimate_safety:main',
        ],
    },
)
