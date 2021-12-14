#pragma once

#include "../Misc/Define.h"

namespace SIByL
{
	namespace Graphic
	{
		struct ShaderOffset
		{
			sINT64 uniformOffset = 0;
			sINT64 bindingRangeIndex = 0;
			sINT64 bindingArrayIndex = 0;

            sINT32 getHashCode() const
            {
                return (sINT32)(((bindingRangeIndex << 20) + bindingArrayIndex) ^ uniformOffset);
            }
            bool operator==(const ShaderOffset& other) const
            {
                return uniformOffset == other.uniformOffset
                    && bindingRangeIndex == other.bindingRangeIndex
                    && bindingArrayIndex == other.bindingArrayIndex;
            }
            bool operator!=(const ShaderOffset& other) const
            {
                return !this->operator==(other);
            }
            bool operator<(const ShaderOffset& other) const
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
            bool operator<=(const ShaderOffset& other) const { return (*this == other) || (*this) < other; }
            bool operator>(const ShaderOffset& other) const { return other < *this; }
            bool operator>=(const ShaderOffset& other) const { return other <= *this; }

		};


	}
}