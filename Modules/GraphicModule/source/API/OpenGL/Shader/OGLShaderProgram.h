#pragma once

#include "../../../Interface/Shader/IShaderProgram.h"

namespace SIByL
{
	namespace Graphic
	{
		class OGLShaderProgram :public IShaderProgram
		{
		public:
			virtual ~OGLShaderProgram() noexcept = default;
		};
	}
}