add_executable(2_dirichlet)

target_compile_options(2_dirichlet PRIVATE -g)
set_target_properties(2_dirichlet PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(2_dirichlet PRIVATE muda cusolver cublas cusparse )

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

include_directories(include)

file(GLOB_RECURSE 2_dirichlet_CU_SOURCE CONFIGURE_DEPENDS "src/*.cu")
target_sources(2_dirichlet PRIVATE ${2_dirichlet_CU_SOURCE})

file(GLOB_RECURSE 2_dirichlet_CPP_SOURCE CONFIGURE_DEPENDS "src/*.cpp")
target_sources(2_dirichlet PRIVATE ${2_dirichlet_CPP_SOURCE})

target_link_libraries(2_dirichlet PRIVATE sfml-graphics)
