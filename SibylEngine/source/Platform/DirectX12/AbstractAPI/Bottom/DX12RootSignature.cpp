#include "SIByLpch.h"
#include "DX12RootSignature.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"

namespace SIByL
{
    DX12RootSignature::DX12RootSignature()
        : m_RootSignatureDesc{}
        , m_NumDescriptorsPerTable{ 0 }
        , m_SamplerTableBitMask(0)
        , m_DescriptorTableBitMask(0)
    {}

    DX12RootSignature::DX12RootSignature(
        const CD3DX12_ROOT_SIGNATURE_DESC& rootSignatureDesc)
        : m_RootSignatureDesc{}
        , m_NumDescriptorsPerTable{ 0 }
        , m_SamplerTableBitMask(0)
        , m_DescriptorTableBitMask(0)
    {
        SetRootSignatureDesc(rootSignatureDesc);
    }

    DX12RootSignature::~DX12RootSignature()
    {
        Destroy();
    }

    void DX12RootSignature::Destroy()
    {
        for (UINT i = 0; i < m_RootSignatureDesc.NumParameters; ++i)
        {
            const D3D12_ROOT_PARAMETER& rootParameter = m_RootSignatureDesc.pParameters[i];
            if (rootParameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
            {
                delete[] rootParameter.DescriptorTable.pDescriptorRanges;
            }
        }

        delete[] m_RootSignatureDesc.pParameters;
        m_RootSignatureDesc.pParameters = nullptr;
        m_RootSignatureDesc.NumParameters = 0;

        delete[] m_RootSignatureDesc.pStaticSamplers;
        m_RootSignatureDesc.pStaticSamplers = nullptr;
        m_RootSignatureDesc.NumStaticSamplers = 0;

        m_DescriptorTableBitMask = 0;
        m_SamplerTableBitMask = 0;

        memset(m_NumDescriptorsPerTable, 0, sizeof(m_NumDescriptorsPerTable));
    }

    void DX12RootSignature::SetRootSignatureDesc(
        const CD3DX12_ROOT_SIGNATURE_DESC& rootSignatureDesc)
    {
        // Make sure any previously allocated root signature description is cleaned 
        // up first.
        Destroy();

        auto device = DX12Context::GetDevice();

        UINT numParameters = rootSignatureDesc.NumParameters;
        D3D12_ROOT_PARAMETER* pParameters = numParameters > 0 ? new CD3DX12_ROOT_PARAMETER[numParameters] : nullptr;

        for (UINT i = 0; i < numParameters; ++i)
        {
            const D3D12_ROOT_PARAMETER& rootParameter = rootSignatureDesc.pParameters[i];
            pParameters[i] = rootParameter;

            if (rootParameter.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
            {
                UINT numDescriptorRanges = rootParameter.DescriptorTable.NumDescriptorRanges;
                D3D12_DESCRIPTOR_RANGE* pDescriptorRanges = numDescriptorRanges > 0 ? new D3D12_DESCRIPTOR_RANGE[numDescriptorRanges] : nullptr;

                memcpy(pDescriptorRanges, rootParameter.DescriptorTable.pDescriptorRanges,
                    sizeof(D3D12_DESCRIPTOR_RANGE) * numDescriptorRanges);

                pParameters[i].DescriptorTable.NumDescriptorRanges = numDescriptorRanges;
                pParameters[i].DescriptorTable.pDescriptorRanges = pDescriptorRanges;

                // Set the bit mask depending on the type of descriptor table.
                if (numDescriptorRanges > 0)
                {
                    switch (pDescriptorRanges[0].RangeType)
                    {
                    case D3D12_DESCRIPTOR_RANGE_TYPE_CBV:
                    case D3D12_DESCRIPTOR_RANGE_TYPE_SRV:
                    case D3D12_DESCRIPTOR_RANGE_TYPE_UAV:
                        m_DescriptorTableBitMask |= (1 << i);
                        break;
                    case D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER:
                        m_SamplerTableBitMask |= (1 << i);
                        break;
                    }
                }

                // Count the number of descriptors in the descriptor table.
                for (UINT j = 0; j < numDescriptorRanges; ++j)
                {
                    m_NumDescriptorsPerTable[i] += pDescriptorRanges[j].NumDescriptors;
                }
            }
        }

        m_RootSignatureDesc.NumParameters = numParameters;
        m_RootSignatureDesc.pParameters = pParameters;

        UINT numStaticSamplers = rootSignatureDesc.NumStaticSamplers;
        D3D12_STATIC_SAMPLER_DESC* pStaticSamplers = numStaticSamplers > 0 ? new D3D12_STATIC_SAMPLER_DESC[numStaticSamplers] : nullptr;

        if (pStaticSamplers)
        {
            memcpy(pStaticSamplers, rootSignatureDesc.pStaticSamplers,
                sizeof(D3D12_STATIC_SAMPLER_DESC) * numStaticSamplers);
        }

        m_RootSignatureDesc.NumStaticSamplers = numStaticSamplers;
        m_RootSignatureDesc.pStaticSamplers = pStaticSamplers;

        D3D12_ROOT_SIGNATURE_FLAGS flags = rootSignatureDesc.Flags;
        m_RootSignatureDesc.Flags = flags;

        ComPtr<ID3DBlob> serializedRootSig = nullptr;
        ComPtr<ID3DBlob> errorBlob = nullptr;
        HRESULT hr = D3D12SerializeRootSignature(&rootSignatureDesc,
            D3D_ROOT_SIGNATURE_VERSION_1, &serializedRootSig, &errorBlob);

        if (errorBlob != nullptr)
        {
            OutputDebugStringA((char*)errorBlob->GetBufferPointer());
        }
        DXCall(hr);

        DXCall(DX12Context::GetDevice()->CreateRootSignature(0,
	        serializedRootSig->GetBufferPointer(),
	        serializedRootSig->GetBufferSize(),
	        IID_PPV_ARGS(&m_RootSignature)));
    }

    uint32_t DX12RootSignature::GetDescriptorTableBitMask(D3D12_DESCRIPTOR_HEAP_TYPE descriptorHeapType) const
    {
        uint32_t descriptorTableBitMask = 0;
        switch (descriptorHeapType)
        {
        case D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV:
            descriptorTableBitMask = m_DescriptorTableBitMask;
            break;
        case D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER:
            descriptorTableBitMask = m_SamplerTableBitMask;
            break;
        }

        return descriptorTableBitMask;
    }

    uint32_t DX12RootSignature::GetNumDescriptors(uint32_t rootIndex) const
    {
        assert(rootIndex < 32);
        return m_NumDescriptorsPerTable[rootIndex];
    }
}