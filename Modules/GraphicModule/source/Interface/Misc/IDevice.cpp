#include "IDevice.h"

#include "../Shader/IShaderComponent.h"
#include "../Shader/IShaderProgram.h"

namespace SIByL
{
	namespace Graphic
	{
		inline Scope<IShaderProgram> IDevice::createProgram(const ShaderProgramDesc& desc)
		{
			Scope<IShaderProgram> program;
			createProgram(desc, program);
			return std::move(program);
		}

	}
}