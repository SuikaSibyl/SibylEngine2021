#pragma once

#include "../../../Interface/Shader/IShaderProgram.h"

namespace SIByL
{
	namespace Graphic
	{
		class GLShaderProgram :public IShaderProgram
		{
		public:
			virtual ~GLShaderProgram() noexcept = default;
		};
	}
}