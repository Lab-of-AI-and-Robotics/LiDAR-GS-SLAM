add_library(registration STATIC
  vel_estimator.cpp
  geometric.cpp
  photometric.cpp
  photometric.cu
)

target_link_libraries(registration
  tools
  Eigen3::Eigen
  ${CUDA_LIBRARIES}
)
target_compile_features(registration PUBLIC)
set_target_properties(registration PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(registration PRIVATE -lineinfo $<$<COMPILE_LANGUAGE:CUDA>: -v -g -lineinfo >)
