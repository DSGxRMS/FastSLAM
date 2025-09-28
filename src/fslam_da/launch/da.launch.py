from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='fslam_da',
            executable='da_node',
            name='fslam_da',
            output='screen',
            parameters=[
                # ---- Gating / solver ----
                {'map_crop_radius': 15.0},
                {'chi2_indiv': 9.2103},    # dof=2 @ 99%
                {'joint_sig': 0.99},
                {'meas_floor_xy': 0.05},
                {'color_lambda': 0.30},
                {'enforce_side_rule': True},
                {'orange_gate_radius': 12.0},
                {'allow_unmatched': True},

                # ---- Health checks (frame-local nudges) ----
                {'enable_health_checks': True},
                {'candidate_median_limit': 3},
                {'p95_mahal_limit': 12.0},
                {'wrong_side_rate_limit': 0.05},
                {'color_confusion_rate_limit': 0.20},
                {'chi2_relax_step': 2.6},
                {'crop_shrink_factor_flagged': 0.8},
                {'drop_color_cost_on_confusion': True},

                # ---- Viz ----
                {'show_gates': False},
                {'show_text_badge': True},

                # ---- Frames / topics ----
                {'detections_topic': '/perception/cones'},     # change if needed
                {'detections_frame': 'base'},                  # 'base' or 'world'
                {'map_topic': '/ground_truth/cones'},          # source for map init (one-shot)
                {'map_init_mode': 'from_topic_once'},          # 'from_topic_once' or 'none'
                {'odom_topic': '/ground_truth/odom'},

                # ---- Start / track ----
                {'start_center_world': [0.0, 0.0]},
            ]
        )
    ])
