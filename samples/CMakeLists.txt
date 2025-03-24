cmake_minimum_required(VERSION 3.15)

# Add basic test
add_executable(solver_basic_test solver_basic_test.cpp)
target_link_libraries(solver_basic_test PRIVATE hip::device hipsolver)

# Install the sample in bin directory
install(TARGETS solver_basic_test DESTINATION bin)

# Add the test to CTest
add_test(NAME hipsolver_basic_context_test COMMAND solver_basic_test) 