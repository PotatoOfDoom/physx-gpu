// SPDX-FileCopyrightText: Copyright (c) 2008-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef __CUDA_KERNEL_WRANGLER__
#define __CUDA_KERNEL_WRANGLER__

#include "foundation/PxPreprocessor.h"

// Make this header is safe for inclusion in headers that are shared with device code.
#if !PX_CUDA_COMPILER

#include "foundation/PxUserAllocated.h"
#include "foundation/PxArray.h"

#if PX_LINUX && PX_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
#endif
// The only cuda.h symbol this header needs is CUfunction, which PxCudaTypes.h
// defines identically (typedef struct CUfunc_st*) when the CUDA toolkit is
// absent, and defers to the real cuda.h typedef when CUDA_VERSION is already
// defined.  Including PxCudaTypes.h instead of <cuda.h> lets this header — and
// therefore the KernelWrangler — build in a Vulkan/no-CUDA-toolkit config while
// staying byte-for-byte identical in a real CUDA build.  The .cpp still binds
// the CUDA driver API (cuModuleGetFunction, ...) at link time; only header
// compilation is decoupled here.
#include "cudamanager/PxCudaTypes.h"
#if PX_LINUX && PX_CLANG
#pragma clang diagnostic pop
#endif

namespace physx
{
	class PxCudaContextManager;
	class PxCudaContext;

class KernelWrangler : public PxUserAllocated
{
	PX_NOCOPY(KernelWrangler)
public:
	KernelWrangler(PxCudaContextManager& cudaContextManager, PxErrorCallback& errorCallback, const char** funcNames, uint16_t numFuncs);
	virtual ~KernelWrangler() {}

	PX_FORCE_INLINE	CUfunction getCuFunction(uint16_t funcIndex) const
	{
		CUfunction func = mCuFunctions[ funcIndex ];
		PX_ASSERT(func);
		return func;
	}

	const char* getCuFunctionName(uint16_t funcIndex) const;

	PX_FORCE_INLINE	bool hadError() const { return mError; }

protected:
	bool					mError;
	const char**			mKernelNames;
	PxArray<CUfunction>		mCuFunctions;
	PxCudaContextManager&	mCudaContextManager;
	PxCudaContext*			mCudaContext;
	PxErrorCallback&		mErrorCallback;
};

}

#endif

#endif
