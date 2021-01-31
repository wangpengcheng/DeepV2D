
#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_ROIPOOLING_OP_GPU_H_
#define TENSORFLOW_USER_OPS_ROIPOOLING_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

/**
 * @brief  向前计算的映射
 * @param  input            输入数据
 * @param  coords           三维坐标
 * @param  dim              数据维度
 * @param  top              输出数据
 * @param  d                device设置
 * @return true             执行成功
 * @return false            执行失败
 */
bool BackProjectForwardLauncher(const float* input, const float* coords,
  const int dim[6], float *top, const Eigen::GpuDevice& d);
/**
 * @brief  反向回调内存管理
 * @param  grad             梯度
 * @param  input            输入数据
 * @param  coords           三维空间坐标
 * @param  dim              综合维度
 * @param  inputs_diff      输入差距
 * @param  coords_diff      坐标差距
 * @param  d                设备
 * @return true 
 * @return false 
 */
bool BackProjectBackwardLauncher(const float *grad, 
  const float* input, const float* coords, const int dim[6],
  float* inputs_diff, float* coords_diff, const Eigen::GpuDevice& d);


}  // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_GPU_H_
