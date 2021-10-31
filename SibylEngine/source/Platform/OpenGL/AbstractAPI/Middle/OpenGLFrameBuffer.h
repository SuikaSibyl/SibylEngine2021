#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"

namespace SIByL
{
	class OpenGLRenderTarget;
	class OpenGLStencilDepth;
	class OpenGLFrameBuffer :public FrameBuffer
	{
	public:
		OpenGLFrameBuffer(const FrameBufferDesc& desc, std::string identifier);

		virtual void Bind() override;
		virtual void Unbind() override;
		virtual void ClearBuffer() override;
		virtual unsigned int CountColorAttachment() override;
		virtual void Resize(uint32_t width, uint32_t height) override;
		virtual void* GetColorAttachment(unsigned int index) override;
		virtual void* GetDepthStencilAttachment() override;
		virtual RenderTarget* GetRenderTarget(unsigned int index) override;

	private:
		void Invalidate();

	private:
		std::vector<Ref<OpenGLRenderTarget>> RenderTargets;
		Ref<OpenGLStencilDepth> DepthStencil;

		unsigned int Width, Height;
		unsigned int m_FrameBufferObject = 0;
	};
}