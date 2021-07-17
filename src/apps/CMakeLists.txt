# Link source files to executables
# add_application(app)

add_executable(app demo.cpp)
default_properties(app)

# Link to all libraries
target_link_libraries(app
    sobfu
    kfusion
    png
    Boost::program_options
    ${Boost_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${VTK_LIBRARIES}
)

add_executable(app2 demo2.cpp)
default_properties(app2)

# Link to all libraries
target_link_libraries(app2
    sobfu
    kfusion
    png
    Boost::program_options
    ${Boost_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${VTK_LIBRARIES}
)
