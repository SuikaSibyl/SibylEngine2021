#pragma once

#include "glm/glm.hpp"

namespace SIByL
{
	class Color
	{
	public:
		Color(glm::vec4 color)
			:m_Color(color) {}

	private:
		glm::vec4 m_Color;
	};
}