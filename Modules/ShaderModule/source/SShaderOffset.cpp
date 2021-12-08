#include "SShaderOffset.h"

namespace SIByL
{
    namespace Shader
    {
        uint32_t SShaderOffset::getHashCode() const
        {
            return (uint32_t)(((bindingRangeIndex << 20) + bindingArrayIndex) ^ uniformOffset);
        }

        bool SShaderOffset::operator==(const SShaderOffset& other) const
        {
            return uniformOffset == other.uniformOffset
                && bindingRangeIndex == other.bindingRangeIndex
                && bindingArrayIndex == other.bindingArrayIndex;
        }

        bool SShaderOffset::operator!=(const SShaderOffset& other) const
        {
            return !this->operator==(other);
        }

        bool SShaderOffset::operator<(const SShaderOffset& other) const
        {
            if (bindingRangeIndex < other.bindingRangeIndex)
                return true;
            if (bindingRangeIndex > other.bindingRangeIndex)
                return false;
            if (bindingArrayIndex < other.bindingArrayIndex)
                return true;
            if (bindingArrayIndex > other.bindingArrayIndex)
                return false;
            return uniformOffset < other.uniformOffset;
        }

        bool SShaderOffset::operator<=(const SShaderOffset& other) const { return (*this == other) || (*this) < other; }
        bool SShaderOffset::operator>(const SShaderOffset& other) const { return other < *this; }
        bool SShaderOffset::operator>=(const SShaderOffset& other) const { return other <= *this; }
    }
}