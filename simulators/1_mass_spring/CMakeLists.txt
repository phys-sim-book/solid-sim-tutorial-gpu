add_executable(1_mass_spring)
set_target_properties(1_mass_spring PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(1_mass_spring PRIVATE muda)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

include_directories(include)

file(GLOB_RECURSE 1_mass_spring_CU_SOURCE CONFIGURE_DEPENDS "src/*.cu")
target_sources(1_mass_spring PRIVATE ${1_mass_spring_CU_SOURCE})

file(GLOB_RECURSE 1_mass_spring_CPP_SOURCE CONFIGURE_DEPENDS "src/*.cpp")
target_sources(1_mass_spring PRIVATE ${1_mass_spring_CPP_SOURCE})

target_link_libraries(1_mass_spring PRIVATE sfml-graphics)
