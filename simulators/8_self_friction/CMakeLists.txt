add_executable(8_self_friction)

target_compile_options(8_self_friction PRIVATE -g)
set_target_properties(8_self_friction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(8_self_friction PRIVATE muda cusolver cublas cusparse)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

include_directories(include)

file(GLOB_RECURSE 8_self_friction_CU_SOURCE CONFIGURE_DEPENDS "src/*.cu")
target_sources(8_self_friction PRIVATE ${8_self_friction_CU_SOURCE})

file(GLOB_RECURSE 8_self_friction_CPP_SOURCE CONFIGURE_DEPENDS "src/*.cpp")
target_sources(8_self_friction PRIVATE ${8_self_friction_CPP_SOURCE})

target_link_libraries(8_self_friction PRIVATE sfml-graphics)
