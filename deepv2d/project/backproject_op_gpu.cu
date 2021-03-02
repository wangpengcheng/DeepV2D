#define EIGEN_USE_GPU
#include <ATen/ATen.h>

#include <cuda.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>
#include <cfloat>

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

using std::max;
using std::min;

template <typename scalar_t>
__global__ void back_project_forward_kernel(
    const int nthreads,
    const scalar_t *__restrict__ input,
    const scalar_t *__restrict__ coords,
    scalar_t *__restrict__ top_data,
    int b,
    int h,
    int w,
    int s,
    int f,
    int c)
{
    int dims[6];
    dims[0] = b;
    dims[1] = h;
    dims[2] = w;
    dims[3] = s;
    dims[4] = f;
    dims[5] = c;
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        //
        // dims: [B[0], H[1], W[2], S[3], F[4], C]
        // input: [B[0], H[1], W[2], F[4], C[5]]
        // coords: [B[0], H[1], W, S, F, 2]
        // 获取当前目标索引
        int n = index;
        int f = n % dims[4];
        n /= dims[4];
        int k = n % dims[3];
        n /= dims[3];
        int w = n % dims[2];
        n /= dims[2];
        int h = n % dims[1];
        n /= dims[1];
        // 计算中间维度
        scalar_t x = coords[2 * index];
        scalar_t y = coords[2 * index + 1];
        
        if (x > 0 && y > 0 && x < dims[2] - 1 && y < dims[1] - 1)
        {
            // 计算对应的索引
            int x0 = static_cast<int>(floor(x));
            int x1 = static_cast<int>(ceil(x));
            int y0 = static_cast<int>(floor(y));
            int y1 = static_cast<int>(ceil(y));
            // 计算差值
            scalar_t dx = x - static_cast<scalar_t>(x0);
            scalar_t dy = y - static_cast<scalar_t>(y0);
            // 计算权重梯度
            scalar_t w00 = (1 - dy) * (1 - dx);
            scalar_t w01 = (1 - dy) * dx;
            scalar_t w10 = dy * (1 - dx);
            scalar_t w11 = dy * dx;
            // 再次计算偏移
            int offset = (n * dims[1] * dims[2] * dims[4] + f) * dims[5];
            int idx00 = offset + dims[4] * dims[5] * (y0 * dims[2] + x0);
            int idx01 = offset + dims[4] * dims[5] * (y0 * dims[2] + x1);
            int idx10 = offset + dims[4] * dims[5] * (y1 * dims[2] + x0);
            int idx11 = offset + dims[4] * dims[5] * (y1 * dims[2] + x1);
            // 计算最终值
            const scalar_t *im00 = input + idx00;
            const scalar_t *im01 = input + idx01;
            const scalar_t *im10 = input + idx10;
            const scalar_t *im11 = input + idx11;
            // 计算顶部数据
            scalar_t *top = top_data + index * dims[5];
            // 计算top
            for (int c = 0; c < dims[5]; c++)
            {
                *top = (*im00) * w00 + (*im01) * w01 + (*im10) * w10 + (*im11) * w11;
                
                im00++;
                im01++;
                im10++;
                im11++;
                top++;
            }

        }
    }
}

at::Tensor back_project_forward(
    at::Tensor input,
    at::Tensor coords)
{
    // 获取维度相关数据
    int b = coords.size(0);
    int h = coords.size(1);
    int w = coords.size(2);
    int s = coords.size(3);
    int f = coords.size(4);
    int c = input.size(4);
    int dims[6];
    dims[0] = b;
    dims[1] = h;
    dims[2] = w;
    dims[3] = s;
    dims[4] = f;
    dims[5] = c;
    // 设置输出 创建输出,注意这里设置为0阶段矩阵
    auto out = at::zeros({b, h, w, s, f, c});

    //定义线程参数
    const int kblock = 128;
    const int nthreads = dims[0] * dims[1] * dims[2] * dims[3] * dims[4];
    // 执行函数
    AT_DISPATCH_FLOATING_TYPES(out.type(), "back_project_forward", ([&] {
                                   back_project_forward_kernel<scalar_t><<<(nthreads + kblock - 1) / kblock, kblock>>>(
                                       nthreads,
                                       input.data<scalar_t>(),
                                       coords.data<scalar_t>(),
                                       out.data<scalar_t>(),
                                       b, h, w, s, f, c);
                               }));
    return out;
}

template <typename scalar_t>
__global__ void back_project_backword_kernel(
    const int nthreads,
    const scalar_t *top_diff,
    const scalar_t *input,
    const scalar_t *coords,
    int b,
    int h,
    int w,
    int s,
    int f,
    int c,
    scalar_t *input_diff,
    scalar_t *coords_diff)
{
    // 记录维度数据
    int dims[6];
    dims[0] = b;
    dims[1] = h;
    dims[2] = w;
    dims[3] = s;
    dims[4] = f;
    dims[5] = c;
    // 进行循环操作
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {

        // dims: [B[0], H[1], W[2], S[3], F[4], C]
        // input: [B[0], H[1], W[2], F[4], C[5]]
        // coords: [B[0], H[1], W, S, F, 2]

        int n = index;
        int f = n % dims[4];
        n /= dims[4];
        int k = n % dims[3];
        n /= dims[3];
        int w = n % dims[2];
        n /= dims[2];
        int h = n % dims[1];
        n /= dims[1];

        scalar_t x = coords[2 * index];
        scalar_t y = coords[2 * index + 1];

        if (x > 0 && y > 0 && x < dims[2] - 1 && y < dims[1] - 1)
        {
            int x0 = static_cast<int>(floor(x));
            int x1 = static_cast<int>(ceil(x));
            int y0 = static_cast<int>(floor(y));
            int y1 = static_cast<int>(ceil(y));

            scalar_t dx = x - static_cast<scalar_t>(x0);
            scalar_t dy = y - static_cast<scalar_t>(y0);

            scalar_t wx0 = 1 - dx;
            scalar_t wx1 = dx;
            scalar_t wy0 = 1 - dy;
            scalar_t wy1 = dy;

            scalar_t w00 = (1 - dy) * (1 - dx);
            scalar_t w01 = (1 - dy) * dx;
            scalar_t w10 = dy * (1 - dx);
            scalar_t w11 = dy * dx;

            int offset = (n * dims[1] * dims[2] * dims[4] + f) * dims[5];
            int idx00 = offset + dims[4] * dims[5] * (y0 * dims[2] + x0);
            int idx01 = offset + dims[4] * dims[5] * (y0 * dims[2] + x1);
            int idx10 = offset + dims[4] * dims[5] * (y1 * dims[2] + x0);
            int idx11 = offset + dims[4] * dims[5] * (y1 * dims[2] + x1);

            const scalar_t *im00 = input + idx00;
            const scalar_t *im01 = input + idx01;
            const scalar_t *im10 = input + idx10;
            const scalar_t *im11 = input + idx11;

            const scalar_t *grad = top_diff + index * dims[5];
            scalar_t *im00_grad = input_diff + idx00;
            scalar_t *im01_grad = input_diff + idx01;
            scalar_t *im10_grad = input_diff + idx10;
            scalar_t *im11_grad = input_diff + idx11;

            scalar_t gx = 0;
            scalar_t gy = 0;
            for (int c = 0; c < dims[5]; c++)
            {
                scalar_t g = *grad;
                atomicAdd(im00_grad, g * w00);
                atomicAdd(im01_grad, g * w01);
                atomicAdd(im10_grad, g * w10);
                atomicAdd(im11_grad, g * w11);

                gx += g * (wy0 * (*im01 - *im00) + wy1 * (*im11 - *im10));
                gy += g * (wx0 * (*im10 - *im00) + wx1 * (*im11 - *im01));

                grad++;
                im00++;
                im00_grad++;
                im01++;
                im01_grad++;
                im10++;
                im10_grad++;
                im11++;
                im11_grad++;
            }

            coords_diff[2 * index] = gx;
            coords_diff[2 * index + 1] = gy;
        }
    }
}

std::vector<torch::Tensor> back_project_backward(
    at::Tensor input,
    at::Tensor coords,
    at::Tensor grad)
{
    // 准备数据
    // 获取输入维度
    int b = coords.size(0);
    int h = coords.size(1);
    int w = coords.size(2);
    int s = coords.size(3);
    int f = coords.size(4);
    int c = input.size(4);

    // 设置输出梯度
    auto inputs_grad = at::zeros({b, h, w, f, c});
    // 设置坐标梯度
    auto coords_grad = at::ones({b, h, w, s, f, 2});
    const int kblock = 128;

    const int nthreads = b * h * w * s * f;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "back_project_backward", ([&] {
                                   back_project_backword_kernel<scalar_t><<<(nthreads + kblock - 1) / kblock, kblock>>>(
                                       nthreads,
                                       grad.data<scalar_t>(),
                                       input.data<scalar_t>(),
                                       coords.data<scalar_t>(),
                                       b, h, w, s, f, c,
                                       inputs_grad.data<scalar_t>(),
                                       coords_grad.data<scalar_t>());
                               }));

    return {inputs_grad, coords_grad};
}
