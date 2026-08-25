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
    
    # Get config template argument
    config_template = LaunchConfiguration('config_template').perform(context)

    # Extra --ros-args for the imager node (e.g. emissivity / temperature range params)
    imager_args = LaunchConfiguration('imager_args').perform(context)
    
    # Get path to supervisor script
    optris_pkg_dir = get_package_share_directory('optris_drivers2')
    supervisor_script = os.path.join(optris_pkg_dir, 'scripts', 'optris_supervisor.sh')
    
    # Launch the supervisor script
    supervisor_process = ExecuteProcess(
        # (an empty string element would break ExecuteProcess, so only append when set)
        cmd=['bash', supervisor_script, xml_config_file, namespace_arg, config_template] + ([imager_args] if imager_args else []),
        name='optris_supervisor',
        output='screen',
        shell=False,
    )
    
    actions = [supervisor_process]

    # Optional automatic 0..250 <-> 150..900 range switching (see scripts/optris_range_switcher.py)
    if LaunchConfiguration('auto_range').perform(context).lower() in ('1', 'true', 'yes'):
        switcher_script = os.path.join(optris_pkg_dir, 'scripts', 'optris_range_switcher.py')
        switcher_args = LaunchConfiguration('switcher_args').perform(context).split()
        actions.append(ExecuteProcess(
            cmd=['python3', switcher_script, '--ns', namespace] + switcher_args,
            name='optris_range_switcher',
            output='screen',
            shell=False,
        ))

    return actions

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
        DeclareLaunchArgument(
            'config_template',
            default_value='',
            description='Path to existing XML config file to use as a template'
        ),
        DeclareLaunchArgument(
            'auto_range',
            default_value='false',
            description='Start optris_range_switcher.py (auto 0..250 <-> 150..900 switching)'
        ),
        DeclareLaunchArgument(
            'switcher_args',
            default_value='',
            description='Extra CLI args for optris_range_switcher.py, e.g. "--roi 40,40,25 --up-temp 240 --down-temp 200"'
        ),
        DeclareLaunchArgument(
            'imager_args',
            default_value='',
            description='Extra --ros-args for optris_imager_node, e.g. "-p emissivity:=0.9 -p temperature_range_min:=150 -p temperature_range_max:=900"'
        ),
    ]

    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
