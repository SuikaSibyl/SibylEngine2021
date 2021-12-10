#include "IDevice.h"

#include "../Shader/IShaderComponent.h"
#include "../Shader/IShaderProgram.h"

namespace SIByL
{
	namespace Graphic
	{
		inline Ref<IShaderProgram> IDevice::createProgram(const ShaderProgramDesc& desc)
		{
			Ref<IShaderProgram> program;
			createProgram(desc, program);
			return program;
		}

	}
}