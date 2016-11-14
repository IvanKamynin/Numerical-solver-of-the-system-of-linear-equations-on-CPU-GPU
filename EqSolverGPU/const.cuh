#include "basic_types.cuh"

const D1 cMin				= static_cast<D1>(1.);
const D1 cMax				= static_cast<D1>(1000.);

const D1 cEpsSolver			= static_cast<D1>(1e-4);

const UI1 cNumRows			= static_cast<UI1>(1024*4);
const UI1 cNumColumns		= static_cast<UI1>(1024*4);

const UI1 cDimBlocksX		= static_cast<UI1>(32);
const UI1 cDimBlocksY		= static_cast<UI1>(8);

const UI1 cNumIterations	= static_cast<UI1>(16);