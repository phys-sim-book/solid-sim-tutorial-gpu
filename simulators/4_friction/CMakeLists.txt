add_executable(4_friction_debug)

target_compile_options(4_friction_debug PRIVATE -g)
set_target_properties(4_friction_debug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(4_friction_debug PRIVATE muda cusolver cublas cusparse )

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

include_directories(include)

file(GLOB_RECURSE 4_friction_debug_CU_SOURCE CONFIGURE_DEPENDS "src/*.cu")
target_sources(4_friction_debug PRIVATE ${4_friction_debug_CU_SOURCE})

file(GLOB_RECURSE 4_friction_debug_CPP_SOURCE CONFIGURE_DEPENDS "src/*.cpp")
target_sources(4_friction_debug PRIVATE ${4_friction_debug_CPP_SOURCE})

target_link_libraries(4_friction_debug PRIVATE sfml-graphics)
