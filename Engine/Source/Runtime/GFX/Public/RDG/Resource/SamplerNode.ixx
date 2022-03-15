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

        auto getSampler() noexcept -> RHI::ISampler*
        {
            if (hasBit(attributes, NodeAttrbutesFlagBits::PLACEHOLDER))
                return extSampler;
            else
                return sampler.get();
        }

        RHI::ISampler* extSampler;
        MemScope<RHI::ISampler> sampler;
    };
}