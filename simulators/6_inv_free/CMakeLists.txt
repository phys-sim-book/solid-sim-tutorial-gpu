add_executable(6_inv_free)

target_compile_options(6_inv_free PRIVATE -g)
set_target_properties(6_inv_free PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(6_inv_free PRIVATE muda cusolver cublas cusparse)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

include_directories(include)

file(GLOB_RECURSE 6_inv_free_CU_SOURCE CONFIGURE_DEPENDS "src/*.cu")
target_sources(6_inv_free PRIVATE ${6_inv_free_CU_SOURCE})

file(GLOB_RECURSE 6_inv_free_CPP_SOURCE CONFIGURE_DEPENDS "src/*.cpp")
target_sources(6_inv_free PRIVATE ${6_inv_free_CPP_SOURCE})

target_link_libraries(6_inv_free PRIVATE sfml-graphics)
