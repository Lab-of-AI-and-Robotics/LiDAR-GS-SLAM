add_library(tools STATIC
  mad_tree.cpp
  image_set.cpp
  image_set.cu
  sum_reduce.cu
)
target_link_libraries(tools Eigen3::Eigen ${CUDA_LIBRARIES})
target_compile_features(tools PUBLIC)
set_target_properties(tools PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(tools PRIVATE -lineinfo $<$<COMPILE_LANGUAGE:CUDA>: -v -g -lineinfo >)
