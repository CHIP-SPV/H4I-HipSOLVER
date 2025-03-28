cmake_minimum_required(VERSION 3.15)

# Add basic test
add_executable(solver_basic_test solver_basic_test.cpp)

# Include the generated headers directory
target_include_directories(solver_basic_test 
    PRIVATE 
    ${CMAKE_CURRENT_BINARY_DIR}/../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include)

if(TARGET deviceInternal) # deviceInternal is available when building in-tree
    target_link_libraries(solver_basic_test PRIVATE deviceInternal hipsolver)
else()
    target_link_libraries(solver_basic_test PRIVATE hip::device hipsolver)
endif()

# Make sure the samples build after the headers are generated
add_dependencies(solver_basic_test ROCmHipsolverHeaders)

# Install the sample in bin directory
install(TARGETS solver_basic_test DESTINATION bin)

# Add the test to CTest
add_test(NAME h4i-hipsolver_basic_context_test COMMAND solver_basic_test) 