add_executable(7_self_contact)

target_compile_options(7_self_contact PRIVATE -g)
set_target_properties(7_self_contact PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(7_self_contact PRIVATE muda cusolver cublas cusparse)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

include_directories(include)

file(GLOB_RECURSE 7_self_contact_CU_SOURCE CONFIGURE_DEPENDS "src/*.cu")
target_sources(7_self_contact PRIVATE ${7_self_contact_CU_SOURCE})

file(GLOB_RECURSE 7_self_contact_CPP_SOURCE CONFIGURE_DEPENDS "src/*.cpp")
target_sources(7_self_contact PRIVATE ${7_self_contact_CPP_SOURCE})

target_link_libraries(7_self_contact PRIVATE sfml-graphics)
