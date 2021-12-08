#pragma once

#include "SShaderOffset.h"

namespace SIByL
{
	namespace Shader
	{
		struct SShaderCursor
		{
			SShaderOffset m_offset;
			
			void setData(void const* data, size_t size) const;

		};
	}
}