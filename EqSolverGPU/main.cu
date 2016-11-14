#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "struct_solver.cuh"
#include "struct_matrix.cuh"

#include "utils.cuh"
#include "time.h"

int main()
{
	CMatrixCPU<D1> a(cNumRows, cNumColumns);
	CMatrixCPU<D1> b(1, cNumColumns);
	CMatrixCPU<D1> d(1, cNumColumns);

	CMatrixCPU<D1> x_1(1, cNumColumns);
	CMatrixCPU<D1> x_2(1, cNumColumns);


	generate1DMatrixForNumericalSolver(a);
	generate1DMatrixForNumericalSolver(b);

	CEquationSolverGPU<D1> eqSolver;
	{
		clock_t startTime = clock();
		eqSolver.solveByFastPointCPU(a, b, x_1);
		clock_t endTime = clock();
		printf("Time CPU : %d\n", endTime - startTime);
	}
	
	{
		clock_t startTime = clock();
		eqSolver.solveByFastPointGPU(a, b, x_2);
		clock_t endTime = clock();
		printf("Time GPU : %d\n", endTime - startTime);
	}

	for (UI1 i = 0; i < cNumRows; ++i)
	{
		if (fabs(x_1.getElem(0, i) - x_2.getElem(0, i)) > cEpsSolver)
			printf("%lf - %lf\n", x_1.getElem(0, i), x_2.getElem(0, i));
	}

	return 0;
}