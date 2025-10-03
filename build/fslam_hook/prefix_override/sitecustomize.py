import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/armaanm/eufs_dev/FastSLAM/install/fslam_hook'
