#pragma once

#include "../../../module/import.h"

namespace SIByL
{
	namespace Graphic
	{
		class IShaderProgram;
		struct ShaderProgramDesc;

		class IDevice
		{
		public:
			virtual ~IDevice() noexcept = default;

		public:
			virtual bool createProgram (const ShaderProgramDesc& desc, Scope<IShaderProgram>& outProgram) noexcept = 0;

			inline Scope<IShaderProgram> createProgram(const ShaderProgramDesc& desc);

		};
	}
}