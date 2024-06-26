add_executable(4_friction)

target_compile_options(4_friction PRIVATE -g)
set_target_properties(4_friction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(4_friction PRIVATE muda cusolver cublas cusparse )

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

include_directories(include)

file(GLOB_RECURSE 4_friction_CU_SOURCE CONFIGURE_DEPENDS "src/*.cu")
target_sources(4_friction PRIVATE ${4_friction_CU_SOURCE})

file(GLOB_RECURSE 4_friction_CPP_SOURCE CONFIGURE_DEPENDS "src/*.cpp")
target_sources(4_friction PRIVATE ${4_friction_CPP_SOURCE})

target_link_libraries(4_friction PRIVATE sfml-graphics)
