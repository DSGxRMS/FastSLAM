from setuptools import setup

package_name = 'fslam'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='armaanm',
    maintainer_email='armaanmahajanbg@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fslam_jcbb = fslam.fslam_node_jcbb:main',
            'fslam_ijcbb = fslam.fslam_node_ijcbb:main',
            'fslam_icp = fslam.fslam_node_icp:main',
            'fslam_hung = fslam.fslam_node_hung:main',
            'plot = fslam.fslam_view:main',
            'fslam_pred = fslam.fslam_pred:main',
            'map = fslam.mapper:main',
            'map_plot = fslam.plot_map:main'
        ],
    },
)
