cmake_minimum_required(VERSION 3.14)
project(FedAvgServer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.14.3
)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)

FetchContent_MakeAvailable(httplib nlohmann_json)

add_executable(server 
    main.cpp
    FedAvgServer.cpp
)

target_link_libraries(server PRIVATE 
    httplib
    nlohmann_json::nlohmann_json
)

if(WIN32)
    target_link_libraries(server PRIVATE ws2_32)
endif()
