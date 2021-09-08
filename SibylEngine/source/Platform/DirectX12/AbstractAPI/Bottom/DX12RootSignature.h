#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Bottom/RootSignature.h"

namespace SIByL
{
    class DX12RootSignature :public RootSignature
    {
    public:
        DX12RootSignature();
        DX12RootSignature(
            const CD3DX12_ROOT_SIGNATURE_DESC& rootSignatureDesc);

        virtual ~DX12RootSignature();

        void Destroy();

        ComPtr<ID3D12RootSignature> GetRootSignature() const
        {
            return m_RootSignature;
        }

        void SetRootSignatureDesc(
            const CD3DX12_ROOT_SIGNATURE_DESC& rootSignatureDesc);

        const CD3DX12_ROOT_SIGNATURE_DESC& GetRootSignatureDesc() const
        {
            return m_RootSignatureDesc;
        }

        uint32_t GetDescriptorTableBitMask(D3D12_DESCRIPTOR_HEAP_TYPE descriptorHeapType) const;
        uint32_t GetNumDescriptors(uint32_t rootIndex) const;

    protected:

    private:
        CD3DX12_ROOT_SIGNATURE_DESC m_RootSignatureDesc;
        ComPtr<ID3D12RootSignature> m_RootSignature;

        // Need to know the number of descriptors per descriptor table.
        // A maximum of 32 descriptor tables are supported (since a 32-bit
        // mask is used to represent the descriptor tables in the root signature.
        uint32_t m_NumDescriptorsPerTable[32];

        // A bit mask that represents the root parameter indices that are 
        // descriptor tables for Samplers.
        uint32_t m_SamplerTableBitMask;
        // A bit mask that represents the root parameter indices that are 
        // CBV, UAV, and SRV descriptor tables.
        uint32_t m_DescriptorTableBitMask;
    };
}