cmake_minimum_required(VERSION 3.8)
project(gello_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(kdl_parser REQUIRED)

# Python package setup  
ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR src/${PROJECT_NAME}
)

# Install Python scripts
install(PROGRAMS
  scripts/run_agent_node.py
  scripts/run_agent_component.py
  scripts/run_gello.py
  scripts/run_GUI_component.py
  scripts/train.py
  scripts/policy_config.py
  scripts/gello_get_offset.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch" PATTERN "*.launch.py"
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Python modules are installed by ament_python_install_package

# Register Python component for ComponentManager
# Note: Python components are registered differently from C++ components
# The component discovery is done through Python entry points

if(BUILD_TESTING)
  # 既存コードは ament_lint に合わせていないので、lint は使わず pytest だけ実行する
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_pose_command_guard test/test_pose_command_guard.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(test_touch_force_frame test/test_touch_force_frame.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(test_freshness test/test_freshness.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(test_mixed_length_episodes test/test_mixed_length_episodes.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(test_stop_transitions test/test_stop_transitions.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(test_save_episode test/test_save_episode.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  ament_add_pytest_test(test_deadline_call test/test_deadline_call.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
endif()

ament_package()