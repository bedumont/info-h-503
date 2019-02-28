# Install script for directory: D:/Users/INFO-H-503/Desktop/info-h-503/Project/stereo-guided-filter_1.0/third_party/libpng-1.6.12

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/Project")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Users/INFO-H-503/Desktop/info-h-503/Project/stereo-guided-filter_1.0/build/third_party/libpng-1.6.12/Release/libpng16_static.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Users/INFO-H-503/Desktop/info-h-503/Project/stereo-guided-filter_1.0/build/third_party/libpng-1.6.12/Debug/libpng16_staticd.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Users/INFO-H-503/Desktop/info-h-503/Project/stereo-guided-filter_1.0/build/third_party/libpng-1.6.12/MinSizeRel/libpng16_static.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Users/INFO-H-503/Desktop/info-h-503/Project/stereo-guided-filter_1.0/build/third_party/libpng-1.6.12/RelWithDebInfo/libpng16_static.lib")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "D:/Users/INFO-H-503/Desktop/info-h-503/Project/stereo-guided-filter_1.0/third_party/libpng-1.6.12/png.h"
    "D:/Users/INFO-H-503/Desktop/info-h-503/Project/stereo-guided-filter_1.0/third_party/libpng-1.6.12/pngconf.h"
    "D:/Users/INFO-H-503/Desktop/info-h-503/Project/stereo-guided-filter_1.0/third_party/libpng-1.6.12/pnglibconf.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/libpng16" TYPE FILE FILES
    "D:/Users/INFO-H-503/Desktop/info-h-503/Project/stereo-guided-filter_1.0/third_party/libpng-1.6.12/png.h"
    "D:/Users/INFO-H-503/Desktop/info-h-503/Project/stereo-guided-filter_1.0/third_party/libpng-1.6.12/pngconf.h"
    "D:/Users/INFO-H-503/Desktop/info-h-503/Project/stereo-guided-filter_1.0/third_party/libpng-1.6.12/pnglibconf.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man3" TYPE FILE FILES
    "D:/Users/INFO-H-503/Desktop/info-h-503/Project/stereo-guided-filter_1.0/third_party/libpng-1.6.12/libpng.3"
    "D:/Users/INFO-H-503/Desktop/info-h-503/Project/stereo-guided-filter_1.0/third_party/libpng-1.6.12/libpngpf.3"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man5" TYPE FILE FILES "D:/Users/INFO-H-503/Desktop/info-h-503/Project/stereo-guided-filter_1.0/third_party/libpng-1.6.12/png.5")
endif()

