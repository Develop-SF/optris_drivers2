#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    """Launch setup for Optris thermal camera with supervisor script"""
    
    # Get launch arguments
    xml_config_file = LaunchConfiguration('xml_config_file').perform(context)
    namespace = LaunchConfiguration('namespace').perform(context)
    
    # If no config file specified, use auto-generated path
    if not xml_config_file or xml_config_file == '':
        xml_config_file = '/tmp/optris_auto.xml'
    
    # Build namespace argument if provided
    namespace_arg = f'--ros-args -r __ns:={namespace}' if namespace and namespace != '' else ''
    
    # Get path to supervisor script
    optris_pkg_dir = get_package_share_directory('optris_drivers2')
    supervisor_script = os.path.join(optris_pkg_dir, 'scripts', 'optris_supervisor.sh')
    
    # Launch the supervisor script
    supervisor_process = ExecuteProcess(
        cmd=['bash', supervisor_script, xml_config_file, namespace_arg],
        name='optris_supervisor',
        output='screen',
        shell=False,
    )
    
    return [supervisor_process]

def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            'xml_config_file',
            default_value='',
            description='Path to Optris XML configuration file (will be auto-generated if empty)'
        ),
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Namespace for the Optris nodes'
        ),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
