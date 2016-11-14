#pragma once
#include "struct_matrix.cuh"
#include "matrix_math.cuh"

#include "defines.cuh"

#include <omp.h>

enum eTypeSolver
{
	eSolverCPU,
	eSolverGPU
};

template<class T>
class CEquationSolverGPU
{
public:
	void solveByFastPointCPU(CMatrixCPU<T>& aCpu, CMatrixCPU<T>& bCpu, CMatrixCPU<T>& xCpu)
	{
#ifdef DUMP_MATRIX
		aCpu.writeMatrixToFile("aCpuDump.dmp");
		bCpu.writeMatrixToFile("bCpuDump.dmp");
#endif

		CMatrixCPU<T> eCpu(1, cNumColumns);
		CMatrixCPU<T> dCpu(1, cNumColumns);

		CMatrixCPU<T> xNwCpu(1, cNumColumns);

		for (UI1 i = 0; i < cNumColumns; ++i)
			dCpu.getElem(0, i) = aCpu.getElem(i, i);

	labelCpu:
		for (UI1 it = 0; it < cNumIterations; ++it)
		{
#pragma omp parallel for
			for (int i_1 = 0; i_1 < static_cast<int>(cNumRows); ++i_1)
			{
				T sum = static_cast<T>(0);
				for (UI1 i_2 = 0; i_2 < cNumColumns; ++i_2)
					sum += aCpu.getElem(i_1, i_2) * xCpu.getElem(0, i_2);

				T xn = (bCpu.getElem(0, i_1) - sum) / dCpu.getElem(0, i_1);
				eCpu.getElem(0, i_1) = fabs(xn);

				xNwCpu.getElem(0, i_1) = xCpu.getElem(0, i_1) + xn;
			}

			xCpu.swap(xNwCpu);
		}

		for (UI1 i = 0, numColumns = eCpu.getNumColumns(); i < numColumns; ++i)
		{
			if (eCpu.getElem(0, i) > cEpsSolver)
				goto labelCpu;
		}

#ifdef DUMP_MATRIX
		xCpu.writeMatrixToFile("xCpuDump.dmp");
#endif
	}

	void solveByFastPointGPU(CMatrixCPU<T>& aCpu, CMatrixCPU<T>& bCpu, CMatrixCPU<T>& xCpu)
	{
#ifdef DUMP_MATRIX
		aCpu.writeMatrixToFile("aGpuDump.dmp");
		bCpu.writeMatrixToFile("bGpuDump.dmp");
#endif

		CMatrixCPU<T> eCpu(1, cNumColumns);
		CMatrixCPU<T> dCpu(1, cNumColumns);

		for (UI1 i = 0; i < cNumColumns; ++i)
			dCpu.getElem(0, i) = aCpu.getElem(i, i);

		CMatrixCUDA<T> aCuda(cNumRows, cNumColumns, aCpu);
		CMatrixCUDA<T> bCuda(1, cNumColumns, bCpu);
		CMatrixCUDA<T> dCuda(1, cNumColumns, dCpu);
		CMatrixCUDA<T> xCuda(1, cNumColumns, xCpu);

		CMatrixCUDA<T> epsCuda(1, cNumColumns);
		CMatrixCUDA<T> xNwCuda(1, cNumColumns);
		CMatrixCUDA<T> aTrCuda(cNumRows, cNumColumns);

		{
			dim3 dimG(cNumRows / cDimBlocksX, cNumColumns / cDimBlocksX);
			dim3 dimB(cDimBlocksX, cDimBlocksY);
			//Transpose matrix A for quickly multiplying matrix on vector
			matrixTransposeCUDA<T> << <dimG, dimB >> >(aCuda.getConstArray(), aTrCuda.getArray());
		}

		int q = 0;
		{
			const UI1 size = eCpu.getSize() * sizeof(T);

			dim3 dimG(cNumRows / cDimBlocksX);
			dim3 dimB(cDimBlocksX);
		labelGpu:
			for (UI1 i = 0; i < cNumIterations; ++i)
			{
				fastPointIterationStep<T> << <dimG, dimB >> >(aTrCuda.getConstArray(), bCuda.getConstArray(), dCuda.getConstArray(), xCuda.getArray(), xNwCuda.getArray(), epsCuda.getArray());
				//Waiting for threads work
				cudaThreadSynchronize();
				xNwCuda.swap(xCuda);
			}

			++q;
			cudaMemcpy(eCpu.getArray(), epsCuda.getConstArray(), size, cudaMemcpyDeviceToHost);

			for (UI1 i = 0, numColumns = eCpu.getNumColumns(); i < numColumns; ++i)
			{
				if (eCpu.getElem(0, i) > cEpsSolver)
					goto labelGpu;
			}

			cudaMemcpy(xCpu.getArray(), xCuda.getConstArray(), size, cudaMemcpyDeviceToHost);

			printf("Q : %d\n", q);
#ifdef DUMP_MATRIX
			xCpu.writeMatrixToFile("xGpuDump.dmp");
#endif
		}
	}
};