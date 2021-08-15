#pragma once

#include "Sibyl/Renderer/GraphicContext.h"

struct GLFWwindow;

namespace SIByL
{
	class OpenGLContext :public GraphicContext
	{
	public:
		OpenGLContext(GLFWwindow* windowHandle);

		virtual void Init() override;
		virtual void SwipBuffers() override;

	private:
		GLFWwindow* m_WindowHandle;
	};
}