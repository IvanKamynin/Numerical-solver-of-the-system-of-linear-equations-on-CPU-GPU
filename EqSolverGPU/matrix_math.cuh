#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "basic_types.cuh"
#include "const.cuh"

template<class T, UI1 widthBlock = cDimBlocksX, UI1 heightBlock = cDimBlocksY, UI1 N = cNumRows>
__global__ void matrixTransposeCUDA(const T* const __restrict__ a, T* __restrict__ b)
{
	const UI1 tx = threadIdx.x;
	const UI1 ty = threadIdx.y;

	const UI1 bx = blockIdx.x;
	const UI1 by = blockIdx.y;

	// Coalesced acces to the memory of the array, without bank conflicts
	__shared__ T as[widthBlock][widthBlock + 1];

	UI1 x = bx * widthBlock + tx;
	UI1 y = by * widthBlock + ty;

#pragma unroll
	for (UI1 i = 0; i < widthBlock; i += heightBlock)
		as[ty + i][tx] = a[(y + i) * N + x];

	__syncthreads();

	x = by * widthBlock + tx;
	y = bx * widthBlock + ty;

#pragma unroll
	for (UI1 i = 0; i < widthBlock; i += heightBlock)
		b[(y + i) * N + x] = as[tx][ty + i];
}

template<class T, UI1 widthBlock = cDimBlocksX, UI1 N = cNumRows>
__global__ void fastPointIterationStep(const T* const __restrict__ a, const T* const __restrict__ b, const T* const __restrict__ d, T* x, T* xNew, T* eNew)
{
	const UI1 nb	= gridDim.x;
	const UI1 bx	= blockIdx.x;
	const UI1 tx	= threadIdx.x;

	const UI1 tid	= tx + bx * widthBlock;

	__shared__ T xs[widthBlock];
	__shared__ T bs[widthBlock];
	__shared__ T ds[widthBlock];
	__shared__ T ws[widthBlock];

	bs[tx] = b[tid];
	ds[tx] = d[tid];
	ws[tx] = x[tid];

	register T sum = static_cast<T>(0);

#pragma unroll
	for (UI1 i = 0; i < nb; ++i)
	{
		xs[tx] = x[tx + i * widthBlock];

		__syncthreads();

#pragma unroll
		for (UI1 e = 0; e < widthBlock; ++e)
			sum += a[tid + (e + widthBlock * i) * N] * xs[e];

		__syncthreads();

	}

	T xn = (bs[tx] - sum) / ds[tx];

	eNew[tid] = fabs(xn);
	xNew[tid] = xn + ws[tx];
}