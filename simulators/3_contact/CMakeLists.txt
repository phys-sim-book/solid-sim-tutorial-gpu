add_executable(3_contact)

target_compile_options(3_contact PRIVATE -g)
set_target_properties(3_contact PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(3_contact PRIVATE muda cusolver cublas cusparse )

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

include_directories(include)

file(GLOB_RECURSE 3_contact_CU_SOURCE CONFIGURE_DEPENDS "src/*.cu")
target_sources(3_contact PRIVATE ${3_contact_CU_SOURCE})

file(GLOB_RECURSE 3_contact_CPP_SOURCE CONFIGURE_DEPENDS "src/*.cpp")
target_sources(3_contact PRIVATE ${3_contact_CPP_SOURCE})

target_link_libraries(3_contact PRIVATE sfml-graphics)
