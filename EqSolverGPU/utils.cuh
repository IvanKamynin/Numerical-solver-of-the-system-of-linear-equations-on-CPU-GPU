#pragma once
#include "struct_matrix.cuh"

template<class T>
inline T randf(const T& min, const T& max)
{	
	return static_cast<T>(rand()) / static_cast<T>(RAND_MAX) * (max - min) + min;
}

template<typename T>
void generate1DMatrixForNumericalSolver(CMatrixCPU<T>& a)
{
	srand((UI1)time(0));

	const UI1 nRows = a.getNumRows();
	const UI1 nColumns = a.getNumColumns();

	if (nRows != nColumns)
	{
		for (UI1 i_1 = 0; i_1 < nRows; ++i_1)
		{
			for (UI1 i_2 = 0; i_2 < nColumns; ++i_2)
				a.getElem(i_1, i_2) = randf(cMin, cMax);
		}
	}
	else
	{
		for (UI1 i_1 = 0; i_1 < nRows; ++i_1)
		{
			T sum = static_cast<T>(0);
			for (UI1 i_2 = 0; i_2 < nColumns; ++i_2)
			{
				a.getElem(i_1, i_2) = randf(cMin, cMax);
				sum += a.getElem(i_1, i_2);
			}
			a.getElem(i_1, i_1) = sum;
		}
	}
}