# Add the basic context tests
add_executable(hipsolver_context_test context_test.cpp)
target_link_libraries(hipsolver_context_test PRIVATE hipsolver hip::host)
target_include_directories(hipsolver_context_test PRIVATE 
                          ${CMAKE_BINARY_DIR}/include 
                          ${CMAKE_SOURCE_DIR}/H4I-HipSOLVER/include)

# Add tests to CTest
add_test(NAME h4i-hipsolver_context_test COMMAND hipsolver_context_test) 