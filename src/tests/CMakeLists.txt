include_directories(..)

set(GLOB TESTS_H *.h)

set(TESTS_SRC
    gradient.cu
)

cuda_add_library(tests SHARED ${TESTS_SRC} ${TESTS_H})

cuda_add_cublas_to_target(tests)
target_link_libraries(tests dlmg cudnn)


cuda_add_executable(tests.gpu main.cu)
cuda_add_cublas_to_target(tests.gpu)

set(LIBS
    tests
    dlmg
    cudnn
    ${MPI_C_LIBRARIES}
)
target_link_libraries(tests.gpu ${LIBS})

