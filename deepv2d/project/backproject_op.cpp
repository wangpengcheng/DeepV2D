#include <torch/torch.h>

#include <vector>
#include <vector>

// CUDA forward declarations
/**
 * @brief  
 * @param  input            My Param doc
 * @param  coords           My Param doc
 * @return std::vector<at::Tensor> 
 */
at::Tensor  back_project_forward(
    at::Tensor input,
    at::Tensor coords
    );

/**
 * @brief  基础向计算函数
 * @param  grad_h           My Param doc
 * @param  grad_cell        My Param doc
 * @param  new_cell         My Param doc
 * @param  input_gate       My Param doc
 * @param  output_gate      My Param doc
 * @param  candidate_cell   My Param doc
 * @param  X                My Param doc
 * @param  gate_weights     My Param doc
 * @param  weights          My Param doc
 * @return std::vector<at::Tensor> 
 */
std::vector<at::Tensor> back_project_backward(
    at::Tensor input,
    at::Tensor coords,
    at::Tensor grad
    );

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
/**
 * @brief  前向计算函数
 * @param  input            My Param doc
 * @param  weights          My Param doc
 * @param  bias             My Param doc
 * @param  old_h            My Param doc
 * @param  old_cell         My Param doc
 * @return std::vector<at::Tensor> 
 */
at::Tensor  backproject_forward(
    at::Tensor input,
    at::Tensor coords
    ) {
   //   std::cout<< "hello word"<<std::endl; 
   CHECK_INPUT(input);
   CHECK_INPUT(coords);
  return back_project_forward(input, coords);
}


std::vector<at::Tensor> backproject_backward(
    at::Tensor input,
    at::Tensor coords,
    at::Tensor grad
    ) {
  return back_project_backward(
      input,
      coords,
      grad
      );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &backproject_forward, "backproject forward (CUDA)");
  m.def("backward", &backproject_backward, "backproject backward (CUDA)");
}

