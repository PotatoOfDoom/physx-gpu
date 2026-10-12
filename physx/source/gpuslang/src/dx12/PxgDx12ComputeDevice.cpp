// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

// Direct3D 12 compute backend for the Slang port of the PhysX GPU kernels.
//
// Binding model (matches PxgSlangComputeDevice's contract and the Vulkan
// backend):
//   - root parameter 0: 32-bit root constants (the Slang push-constant block,
//     emitted as cbuffer b0 by slangc -target dxil)
//   - root parameter 1: one descriptor table of UAVs, u0..u(N-1), matching
//     the shader's storage buffers in declaration order
//   - buffers live on UPLOAD-compatible heaps for simplicity (custom heap
//     with WRITE_BACK, or readback staging on discrete adapters); production
//     integrations would use DEFAULT heaps plus staging rings.
//
// Kernels are compiled offline: slangc -target dxil (via HLSL/DXC).  On
// non-Windows platforms this translation unit compiles to nothing; use the
// Vulkan backend there.

#if defined(_WIN32)

#include "PxgSlangComputeDevice.h"

#include <d3d12.h>
#include <dxgi1_4.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

namespace physx
{
namespace gpuslang
{

#define PXG_DX_CHECK(expr)                                                                  \
    do                                                                                      \
    {                                                                                       \
        HRESULT pxgHr_ = (expr);                                                            \
        if (FAILED(pxgHr_))                                                                 \
        {                                                                                   \
            fprintf(stderr, "[gpuslang/dx12] %s failed: 0x%08lx (%s:%d)\n", #expr,          \
                    static_cast<unsigned long>(pxgHr_), __FILE__, __LINE__);                \
            assert(false);                                                                  \
        }                                                                                   \
    } while (0)

template <typename T>
class ComPtrLite
{
public:
    ComPtrLite() : mPtr(nullptr) {}
    ~ComPtrLite() { reset(); }
    ComPtrLite(const ComPtrLite&) = delete;
    ComPtrLite& operator=(const ComPtrLite&) = delete;

    T** put() { reset(); return &mPtr; }
    T* get() const { return mPtr; }
    T* operator->() const { return mPtr; }
    explicit operator bool() const { return mPtr != nullptr; }
    void reset()
    {
        if (mPtr)
        {
            mPtr->Release();
            mPtr = nullptr;
        }
    }
    T* detach()
    {
        T* p = mPtr;
        mPtr = nullptr;
        return p;
    }

private:
    T* mPtr;
};

class PxgDx12ComputeDevice : public PxgSlangComputeDevice
{
private:
    struct Pipeline;   // defined with the other private state below (refusePipeline takes one)
    struct Buffer;     // likewise (liveBuffer returns one)

public:
    PxgDx12ComputeDevice() {}

    ~PxgDx12ComputeDevice() override
    {
        waitForGpu();
        for (auto& kv : mBuffers)
            kv.second.resource->Release();
        for (auto& kv : mPipelines)
        {
            if (kv.second.pso)
                kv.second.pso->Release();
            if (kv.second.rootSignature)
                kv.second.rootSignature->Release();
        }
        if (mEvent)
            CloseHandle(mEvent);
    }

    bool init(const PxgSlangDeviceDesc& desc)
    {
        UINT dxgiFlags = 0;
        if (desc.enableValidation)
        {
            ComPtrLite<ID3D12Debug> debug;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(debug.put()))))
            {
                debug->EnableDebugLayer();
                dxgiFlags |= DXGI_CREATE_FACTORY_DEBUG;
            }
        }

        ComPtrLite<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory2(dxgiFlags, IID_PPV_ARGS(factory.put()))))
            return false;

        ComPtrLite<IDXGIAdapter1> adapter;
        if (desc.preferSoftwareDevice)
        {
            // WARP: the D3D12 software rasterizer (counterpart of llvmpipe).
            if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(adapter.put()))))
                return false;
        }
        else
        {
            for (UINT i = 0; factory->EnumAdapters1(i, adapter.put()) != DXGI_ERROR_NOT_FOUND; ++i)
            {
                DXGI_ADAPTER_DESC1 adapterDesc;
                adapter->GetDesc1(&adapterDesc);
                if (adapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                    continue;
                break;
            }
        }

        if (FAILED(D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(mDevice.put()))))
            return false;

        {
            DXGI_ADAPTER_DESC1 adapterDesc;
            adapter->GetDesc1(&adapterDesc);
            char name[256] = {};
            const wchar_t* w = adapterDesc.Description;
            size_t n = 0;
            while (w[n] && n < 255)
            {
                name[n] = char(w[n] < 128 ? w[n] : '?');
                ++n;
            }
            mDeviceName = name;
        }

        D3D12_COMMAND_QUEUE_DESC queueDesc = {};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        PXG_DX_CHECK(mDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(mQueue.put())));

        PXG_DX_CHECK(mDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(mAllocator.put())));
        PXG_DX_CHECK(mDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, mAllocator.get(), nullptr,
                                                IID_PPV_ARGS(mCommandList.put())));
        PXG_DX_CHECK(mCommandList->Close());

        PXG_DX_CHECK(mDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(mFence.put())));
        mEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

        // Shader-visible descriptor heap for UAV tables (ring, reset on flush).
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = kDescriptorHeapSize;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        PXG_DX_CHECK(mDevice->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(mDescriptorHeap.put())));
        mDescriptorSize = mDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        return true;
    }

    const char* getDeviceName() const override { return mDeviceName.c_str(); }
    const char* getBackendName() const override { return "Direct3D 12"; }

    // ---- resources ----

    PxgBufferHandle createBuffer(size_t sizeBytes) override
    {
        if (sizeBytes == 0)
            sizeBytes = 4;
        // UAV-capable buffers require 4-byte multiples.
        sizeBytes = (sizeBytes + 3) & ~size_t(3);

        // Custom heap: CPU write-back + GPU access — the D3D12 analogue of
        // Vulkan HOST_VISIBLE|HOST_COHERENT, valid for UAV usage (UPLOAD
        // heaps are not).
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_CUSTOM;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;

        D3D12_RESOURCE_DESC resDesc = {};
        resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resDesc.Width = sizeBytes;
        resDesc.Height = 1;
        resDesc.DepthOrArraySize = 1;
        resDesc.MipLevels = 1;
        resDesc.SampleDesc.Count = 1;
        resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        Buffer buf = {};
        buf.size = sizeBytes;
        // P9.98 check round 1: checked IN RELEASE BUILDS (PXG_DX_CHECK only asserts), as the
        // Vulkan backend's createBufferInternal is: a failure returns 0 -- the shim's
        // OutOfMemory -- instead of registering a handle with a null resource that
        // upload / download / dispatch then dereference.
        const HRESULT hr = mDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
                                                           D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                           IID_PPV_ARGS(&buf.resource));
        if (FAILED(hr) || !buf.resource)
        {
            std::fprintf(stderr, "[PxgDx12] createBuffer(%zu B): CreateCommittedResource failed with HRESULT "
                                 "0x%08lx; returning 0\n", sizeBytes, static_cast<unsigned long>(hr));
            if (buf.resource)
                buf.resource->Release();
            return 0;
        }

        const PxgBufferHandle handle = mNextBufferHandle++;
        mBuffers[handle] = buf;
        return handle;
    }

    void destroyBuffer(PxgBufferHandle handle) override
    {
        auto it = mBuffers.find(handle);
        if (it == mBuffers.end())
            return;
        waitForGpu();
        it->second.resource->Release();
        mBuffers.erase(it);
    }

    // P9.98 check round 1: the Vulkan backend's transfer contract -- an unknown handle is
    // skipped with a message (operator[] default-inserted a Buffer with a null resource and
    // then called Map on it), a range past the buffer is refused, and a failed Map is
    // checked in Release instead of memcpy'ing through a null pointer.
    Buffer* liveBuffer(PxgBufferHandle handle, size_t offset, size_t bytes, const char* caller)
    {
        const auto it = handle == 0 ? mBuffers.end() : mBuffers.find(handle);
        if (it == mBuffers.end())
        {
            std::fprintf(stderr, "[PxgDx12] %s: buffer handle %u is not a live buffer; refused\n", caller,
                         unsigned(handle));
            return nullptr;
        }
        if (offset > it->second.size || bytes > it->second.size - offset)
        {
            std::fprintf(stderr, "[PxgDx12] %s: %zu B at offset %zu runs past buffer %u's %zu bytes; refused\n",
                         caller, bytes, offset, unsigned(handle), it->second.size);
            return nullptr;
        }
        return &it->second;
    }

    void upload(PxgBufferHandle handle, const void* data, size_t sizeBytes, size_t dstOffset) override
    {
        if (sizeBytes == 0)
            return;
        Buffer* buf = liveBuffer(handle, dstOffset, sizeBytes, "upload");
        if (!buf)
            return;
        void* mapped = nullptr;
        const D3D12_RANGE readRange = { 0, 0 };
        const HRESULT hr = buf->resource->Map(0, &readRange, &mapped);
        if (FAILED(hr) || !mapped)
        {
            std::fprintf(stderr, "[PxgDx12] upload: Map failed with HRESULT 0x%08lx; refused\n",
                         static_cast<unsigned long>(hr));
            return;
        }
        memcpy(static_cast<uint8_t*>(mapped) + dstOffset, data, sizeBytes);
        buf->resource->Unmap(0, nullptr);
    }

    void download(PxgBufferHandle handle, void* data, size_t sizeBytes, size_t srcOffset) override
    {
        if (sizeBytes == 0)
            return;
        Buffer* buf = liveBuffer(handle, srcOffset, sizeBytes, "download");
        if (!buf)
            return;
        void* mapped = nullptr;
        const D3D12_RANGE readRange = { srcOffset, srcOffset + sizeBytes };
        const HRESULT hr = buf->resource->Map(0, &readRange, &mapped);
        if (FAILED(hr) || !mapped)
        {
            std::fprintf(stderr, "[PxgDx12] download: Map failed with HRESULT 0x%08lx; refused\n",
                         static_cast<unsigned long>(hr));
            return;
        }
        memcpy(data, static_cast<uint8_t*>(mapped) + srcOffset, sizeBytes);
        const D3D12_RANGE writtenRange = { 0, 0 };
        buf->resource->Unmap(0, &writtenRange);
    }

    // ---- pipelines ----

    PxgPipelineHandle createPipeline(const void* code, size_t codeSize, const char* entryName,
                                     uint32_t numBuffers, const uint32_t* bufferStrides,
                                     uint32_t pushConstantSize) override
    {
        (void)entryName; // DXIL blobs are single-entry; selection happened at compile time

        Pipeline pipe = {};
        pipe.numBuffers = numBuffers;
        pipe.pushConstantSize = pushConstantSize;
        pipe.bufferStrides.assign(bufferStrides, bufferStrides + numBuffers);

        // Root signature: [0] root constants b0, [1] UAV table u0..uN-1.
        D3D12_DESCRIPTOR_RANGE uavRange = {};
        uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        uavRange.NumDescriptors = numBuffers;
        uavRange.BaseShaderRegister = 0;

        D3D12_ROOT_PARAMETER params[2] = {};
        UINT numParams = 0;

        if (pushConstantSize > 0)
        {
            params[numParams].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            params[numParams].Constants.ShaderRegister = 0;
            params[numParams].Constants.Num32BitValues = (pushConstantSize + 3) / 4;
            params[numParams].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            pipe.rootConstantParamIndex = numParams;
            ++numParams;
        }
        else
        {
            pipe.rootConstantParamIndex = UINT(-1);
        }

        if (numBuffers > 0)
        {
            params[numParams].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            params[numParams].DescriptorTable.NumDescriptorRanges = 1;
            params[numParams].DescriptorTable.pDescriptorRanges = &uavRange;
            params[numParams].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            pipe.uavTableParamIndex = numParams;
            ++numParams;
        }
        else
        {
            pipe.uavTableParamIndex = UINT(-1);
        }

        D3D12_ROOT_SIGNATURE_DESC rootDesc = {};
        rootDesc.NumParameters = numParams;
        rootDesc.pParameters = params;

        // P9.98 check round 0: each HRESULT is checked IN RELEASE BUILDS (PXG_DX_CHECK only
        // asserts) and a failure releases what was made and returns 0, as the Vulkan
        // backend's createPipelineInternal does -- a handle with a null root signature or
        // PSO used to be registered, and dispatch / destroyPipeline then used it.
        ComPtrLite<ID3DBlob> serialized;
        ComPtrLite<ID3DBlob> errors;
        HRESULT hr = D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, serialized.put(),
                                                 errors.put());
        if (FAILED(hr) || !serialized)
            return refusePipeline(pipe, "D3D12SerializeRootSignature", hr);
        hr = mDevice->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                          IID_PPV_ARGS(&pipe.rootSignature));
        if (FAILED(hr) || !pipe.rootSignature)
            return refusePipeline(pipe, "CreateRootSignature", hr);

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = pipe.rootSignature;
        psoDesc.CS.pShaderBytecode = code;
        psoDesc.CS.BytecodeLength = codeSize;
        hr = mDevice->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pipe.pso));
        if (FAILED(hr) || !pipe.pso)
            return refusePipeline(pipe, "CreateComputePipelineState", hr);

        const PxgPipelineHandle handle = mNextPipelineHandle++;
        mPipelines[handle] = pipe;
        return handle;
    }

    PxgPipelineHandle refusePipeline(Pipeline& pipe, const char* what, HRESULT hr)
    {
        std::fprintf(stderr, "[PxgDx12] createPipeline: %s failed with HRESULT 0x%08lx; refused\n", what,
                     static_cast<unsigned long>(hr));
        if (pipe.pso)
            pipe.pso->Release();
        if (pipe.rootSignature)
            pipe.rootSignature->Release();
        pipe.pso = nullptr;
        pipe.rootSignature = nullptr;
        return 0;
    }

    void destroyPipeline(PxgPipelineHandle handle) override
    {
        auto it = mPipelines.find(handle);
        if (it == mPipelines.end())
            return;
        waitForGpu();
        if (it->second.pso)
            it->second.pso->Release();
        if (it->second.rootSignature)
            it->second.rootSignature->Release();
        mPipelines.erase(it);
    }

    // ---- execution ----

    bool dispatch(PxgPipelineHandle pipelineHandle, uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                  const PxgBufferHandle* buffers, uint32_t numBuffers,
                  const void* pushConstants, uint32_t pushConstantSize) override
    {
        // P9.98: an unknown handle is refused, never default-inserted (see the Vulkan
        // backend's dispatchInternal).
        auto found = mPipelines.find(pipelineHandle);
        if (pipelineHandle == 0 || found == mPipelines.end())
        {
            std::fprintf(stderr, "[PxgDx12] dispatch: pipeline handle %u is not a live pipeline; "
                                 "nothing recorded\n", unsigned(pipelineHandle));
            return false;
        }
        Pipeline& pipe = found->second;
        assert(pushConstantSize == pipe.pushConstantSize);
        // P9.98 check round 0: the header's dispatch contract -- a blob larger than the
        // root constants (Num32BitValues = ceil(pipe.pushConstantSize / 4)), a buffer
        // count that is not the pipeline's, or a buffer handle that is 0 / unknown is
        // refused before anything is recorded (they were assert-only, and the buffers
        // operator[] lookups default-inserted a null resource).
        const uint32_t rootConstantBytes =
            pipe.rootConstantParamIndex != UINT(-1) ? (pipe.pushConstantSize + 3u) / 4u * 4u : 0u;
        if (pushConstantSize > rootConstantBytes)
        {
            std::fprintf(stderr, "[PxgDx12] dispatch: a %u-byte push blob does not fit pipeline %u's %u bytes of "
                                 "root constants; nothing recorded\n", pushConstantSize, unsigned(pipelineHandle),
                         rootConstantBytes);
            return false;
        }
        if (numBuffers != pipe.numBuffers || (numBuffers > 0 && !buffers))
        {
            std::fprintf(stderr, "[PxgDx12] dispatch: pipeline %u binds %u buffer(s), the dispatch passed %u; "
                                 "nothing recorded\n", unsigned(pipelineHandle), pipe.numBuffers, numBuffers);
            return false;
        }
        std::vector<const Buffer*> bound(numBuffers, nullptr);
        for (uint32_t i = 0; i < numBuffers; ++i)
        {
            const auto b = buffers[i] == 0 ? mBuffers.end() : mBuffers.find(buffers[i]);
            if (b == mBuffers.end())
            {
                std::fprintf(stderr, "[PxgDx12] dispatch: buffer handle %u is not a live buffer; nothing recorded\n",
                             unsigned(buffers[i]));
                return false;
            }
            bound[i] = &b->second;
        }
        assert(mDescriptorCursor + numBuffers <= kDescriptorHeapSize && "descriptor heap exhausted; flush() more often");

        beginRecording();

        mCommandList->SetPipelineState(pipe.pso);
        mCommandList->SetComputeRootSignature(pipe.rootSignature);

        if (pipe.rootConstantParamIndex != UINT(-1) && pushConstantSize > 0)
        {
            // P9.98 check round 1: SetComputeRoot32BitConstants reads whole 32-bit values,
            // so a blob whose size is not a multiple of 4 is copied into zeroed scratch of
            // the rounded size first -- reading ceil(size / 4) words straight from the
            // caller's blob read up to 3 bytes past it (the Vulkan backend's
            // pushConstantsPadded copies the same way).
            const void* src = pushConstants;
            const uint32_t words = (pushConstantSize + 3) / 4;
            if ((pushConstantSize & 3u) != 0)
            {
                mPushScratch.assign(size_t(words) * 4u, 0u);
                if (pushConstants)
                    memcpy(mPushScratch.data(), pushConstants, pushConstantSize);
                src = mPushScratch.data();
            }
            mCommandList->SetComputeRoot32BitConstants(pipe.rootConstantParamIndex, words, src, 0);
        }

        if (numBuffers > 0)
        {
            D3D12_CPU_DESCRIPTOR_HANDLE cpuBase = mDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
            D3D12_GPU_DESCRIPTOR_HANDLE gpuBase = mDescriptorHeap->GetGPUDescriptorHandleForHeapStart();
            cpuBase.ptr += size_t(mDescriptorCursor) * mDescriptorSize;
            gpuBase.ptr += UINT64(mDescriptorCursor) * mDescriptorSize;

            for (uint32_t i = 0; i < numBuffers; ++i)
            {
                const Buffer& buf = *bound[i];
                const uint32_t stride = pipe.bufferStrides[i];

                // Structured UAV matching the shader's RWStructuredBuffer stride.
                D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
                uavDesc.Format = DXGI_FORMAT_UNKNOWN;
                uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                uavDesc.Buffer.NumElements = UINT(buf.size / stride);
                uavDesc.Buffer.StructureByteStride = stride;

                D3D12_CPU_DESCRIPTOR_HANDLE dst = cpuBase;
                dst.ptr += size_t(i) * mDescriptorSize;
                mDevice->CreateUnorderedAccessView(buf.resource, nullptr, &uavDesc, dst);
            }

            mDescriptorCursor += numBuffers;
            mCommandList->SetComputeRootDescriptorTable(pipe.uavTableParamIndex, gpuBase);
        }

        mCommandList->Dispatch(gridX, gridY, gridZ);
        return true;
    }

    void barrier() override
    {
        if (!mRecording)
            return;
        D3D12_RESOURCE_BARRIER uavBarrier = {};
        uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uavBarrier.UAV.pResource = nullptr; // all UAV accesses
        mCommandList->ResourceBarrier(1, &uavBarrier);
    }

    void flush() override
    {
        if (!mRecording)
            return;

        PXG_DX_CHECK(mCommandList->Close());
        ID3D12CommandList* lists[] = { mCommandList.get() };
        mQueue->ExecuteCommandLists(1, lists);
        waitForGpu();

        PXG_DX_CHECK(mAllocator->Reset());
        mDescriptorCursor = 0;
        mRecording = false;
    }

private:
    static const uint32_t kDescriptorHeapSize = 4096;

    struct Buffer
    {
        ID3D12Resource* resource;
        size_t size;
    };

    struct Pipeline
    {
        ID3D12PipelineState* pso;
        ID3D12RootSignature* rootSignature;
        uint32_t numBuffers;
        uint32_t pushConstantSize;
        UINT rootConstantParamIndex;
        UINT uavTableParamIndex;
        std::vector<uint32_t> bufferStrides;
    };

    void beginRecording()
    {
        if (mRecording)
            return;
        PXG_DX_CHECK(mCommandList->Reset(mAllocator.get(), nullptr));
        ID3D12DescriptorHeap* heaps[] = { mDescriptorHeap.get() };
        mCommandList->SetDescriptorHeaps(1, heaps);
        mRecording = true;
    }

    void waitForGpu()
    {
        if (!mQueue || !mFence)
            return;
        const UINT64 value = ++mFenceValue;
        PXG_DX_CHECK(mQueue->Signal(mFence.get(), value));
        if (mFence->GetCompletedValue() < value)
        {
            PXG_DX_CHECK(mFence->SetEventOnCompletion(value, mEvent));
            WaitForSingleObject(mEvent, INFINITE);
        }
    }

    ComPtrLite<ID3D12Device> mDevice;
    ComPtrLite<ID3D12CommandQueue> mQueue;
    ComPtrLite<ID3D12CommandAllocator> mAllocator;
    ComPtrLite<ID3D12GraphicsCommandList> mCommandList;
    ComPtrLite<ID3D12Fence> mFence;
    ComPtrLite<ID3D12DescriptorHeap> mDescriptorHeap;
    HANDLE mEvent = nullptr;
    UINT64 mFenceValue = 0;
    UINT mDescriptorSize = 0;
    uint32_t mDescriptorCursor = 0;
    bool mRecording = false;

    std::unordered_map<PxgBufferHandle, Buffer> mBuffers;
    std::vector<uint8_t> mPushScratch;   // P9.98 check round 1: rounded root-constant copy
    std::unordered_map<PxgPipelineHandle, Pipeline> mPipelines;
    PxgBufferHandle mNextBufferHandle = 1;
    PxgPipelineHandle mNextPipelineHandle = 1;

    std::string mDeviceName;
};

PxgSlangComputeDevice* PxgCreateDx12ComputeDevice(const PxgSlangDeviceDesc& desc)
{
    PxgDx12ComputeDevice* device = new PxgDx12ComputeDevice();
    if (!device->init(desc))
    {
        delete device;
        return nullptr;
    }
    return device;
}

} // namespace gpuslang
} // namespace physx

#endif // _WIN32
