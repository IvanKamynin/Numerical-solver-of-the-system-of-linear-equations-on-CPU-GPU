#pragma once
#include "stdio.h"
#include "stdlib.h"

#include "basic_types.cuh"

#include "cuda_runtime.h"

#include <algorithm>

template<class T>
class CMatrix1D
{
public:
	CMatrix1D() : m_nRows(static_cast<UI1>(0)), m_nColumns(static_cast<UI1>(0)), m_size(static_cast<UI1>(0)), m_matrix(nullptr)
	{

	}

	CMatrix1D(UI1 nRows, UI1 nColumns) : m_nRows(nRows), m_nColumns(nColumns), m_size(nRows * nColumns)
	{

	}

	T* getArray()
	{
		return m_matrix;
	}

	void swap(CMatrix1D& a)
	{
		std::swap(m_matrix, a.m_matrix);
	}

	const T* getConstArray() const
	{
		return m_matrix;
	}

	UI1 getNumRows() const
	{
		return m_nRows;
	}

	UI1 getNumColumns() const
	{
		return m_nColumns;
	}

	UI1 getSize() const 
	{
		return m_size;
	}

protected:
	T* m_matrix;

	UI1 m_nRows;
	UI1 m_nColumns;
	UI1 m_size;
};

template<class T>
class CMatrixCPU : public CMatrix1D<T>
{
public:
	CMatrixCPU() : CMatrix1D<T>()
	{

	}

	CMatrixCPU(UI1 nRows, UI1 nColumns) : CMatrix1D<T>(nRows, nColumns)
	{
		m_matrix = static_cast<T*>(calloc(m_size, sizeof(T)));
	}

	~CMatrixCPU()
	{
		free(static_cast<void*>(m_matrix));
	}

	T& getElem(UI1 i_1, UI1 i_2)
	{
		return m_matrix[i_1 * m_nColumns + i_2];
	}

	void writeMatrixToFile(const char* const pName) const
	{
		FILE* pFile = nullptr;

		errno_t err = fopen_s(&pFile, pName, "wb");
		if (err != NULL)
			throw;

		fprintf_s(pFile, "%u %u\n", m_nRows, m_nColumns);

		for (UI1 i = 0; i < m_size; ++i)
			fprintf_s(pFile, "%lf\n", m_matrix[i]);

		fclose(pFile);
	}

	void readMatrixFromFile(const char* const pName) const
	{
		FILE* pFile = nullptr;

		errno_t err = fopen_s(&pFile, pName, "rb");
		if (err != NULL)
			throw;

		fscanf_s(pFile, "%u %u\n", &m_nRows, &m_nColumns);

		m_size = m_nRows * m_nColumns;

		for (UI1 i = 0; i < m_size; ++i)
			fscanf_s(pFile, "%lf\n", &m_matrix[i]);

		fclose(pFile);
	}
};

template<class T>
class CMatrixCUDA : public CMatrix1D<T>
{
public:
	CMatrixCUDA(UI1 nRows, UI1 nColumns) : CMatrix1D<T>(nRows, nColumns)
	{
		cudaMalloc(reinterpret_cast<void**>(&m_matrix), m_size * sizeof(T));
	}

	CMatrixCUDA(UI1 nRows, UI1 nColumns, const CMatrixCPU<T>& a) : CMatrix1D<T>(nRows, nColumns)
	{
		cudaMalloc(reinterpret_cast<void**>(&m_matrix), m_size * sizeof(T));
		cudaMemcpy(static_cast<void*>(m_matrix), a.getConstArray(), m_size * sizeof(T), cudaMemcpyHostToDevice);
	}

	~CMatrixCUDA()
	{
		cudaFree(static_cast<void*>(m_matrix));
	}
};