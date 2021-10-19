#pragma once

namespace SIByL
{
	class RenderTarget;

	class FrameBufferLibrary
	{
	public:
		static void ResizeAll(unsigned int width, unsigned int height);
		static RenderTarget* GetRenderTarget(std::string Identifier);
	};
}