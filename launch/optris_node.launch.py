#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource



def launch_setup(context, *args, **kwargs):
    """Launch setup for RGBT with multiple cameras and thermal imaging"""
    nodes = []
    optris_calib_path = os.path.join(
        get_package_share_directory("optris_drivers2"),
        "config/"
    )
    # OptrisXi thermal camera - imager node
    optris_imager_node = Node(
        package='optris_drivers2',
        executable='optris_imager_node',
        name='optris_imager',
        parameters=[{
            'xml_config_file': optris_calib_path + 'xi80_2.xml',
        }],
        output='screen'
    )
    nodes.append(optris_imager_node)

    # OptrisXi thermal camera - color convert node
    optris_colorconvert_node = Node(
        package='optris_drivers2',
        executable='optris_colorconvert_node',
        name='optris_colorconvert',
        output='screen'
    )
    nodes.append(optris_colorconvert_node)

    return nodes

def generate_launch_description():
    declared_arguments = []

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
