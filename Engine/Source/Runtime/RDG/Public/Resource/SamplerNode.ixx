module;
#include <cstdint>
export module GFX.RDG.SamplerNode;
import GFX.RDG.Common;
import Core.MemoryManager;
import Core.BitFlag;
import RHI.ISampler;

namespace SIByL::GFX::RDG
{
    export struct SamplerNode :public ResourceNode
    {
        SamplerNode() { type = NodeDetailedType::SAMPLER; }

        auto devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void
        {
            if (!(attributes & (uint32_t)NodeAttrbutesFlagBits::PLACEHOLDER))
            {
                sampler = factory->createSampler(desc);
            }
        }

        auto getSampler() noexcept -> RHI::ISampler*
        {
            if (hasBit(attributes, NodeAttrbutesFlagBits::PLACEHOLDER))
                return extSampler;
            else
                return sampler.get();
        }

        RHI::SamplerDesc desc;
        RHI::ISampler* extSampler;
        MemScope<RHI::ISampler> sampler;
    };
}