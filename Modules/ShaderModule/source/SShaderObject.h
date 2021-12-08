#pragma once

namespace SIByL
{
	namespace Shader
	{
		class ISShaderObject
		{
		public:
			virtual slang::TypeLayoutReflection* SLANG_MCALL getElementTypeLayout() = 0;
			virtual ShaderObjectContainerType SLANG_MCALL getContainerType() = 0;
			virtual UInt SLANG_MCALL getEntryPointCount() = 0;

		};
	}
}