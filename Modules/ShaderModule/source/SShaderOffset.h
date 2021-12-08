#pragma once

#include "SShaderTypes.h"

namespace SIByL
{
	namespace Shader
	{
        struct SShaderOffset
        {
            SShaderInt uniformOffset = 0;
            SShaderInt bindingRangeIndex = 0;
            SShaderInt bindingArrayIndex = 0;

            uint32_t getHashCode() const;

            bool operator==(const SShaderOffset& other) const;
            bool operator!=(const SShaderOffset& other) const;
            bool operator<(const SShaderOffset& other) const;
            bool operator<=(const SShaderOffset& other) const;
            bool operator>(const SShaderOffset& other) const;
            bool operator>=(const SShaderOffset& other) const;
        };
	}
}