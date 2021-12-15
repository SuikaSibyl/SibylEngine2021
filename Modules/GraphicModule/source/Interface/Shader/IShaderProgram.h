#pragma once

#include "../Misc/Define.h"

namespace SIByL
{
	namespace Graphic
	{
		class IShaderComponent;

		class IShaderProgram
		{
		public:
			virtual ~IShaderProgram() noexcept = default;
		};

		struct ShaderProgramDesc
		{
			PipelineType pipelineType;
			IShaderComponent* program;
		};
	}
}