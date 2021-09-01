#pragma once

#include "Sibyl/Graphic/AbstractAPI/Middle/FrameBuffer.h"

namespace SIByL
{
	class OpenGLFrameBuffer :public FrameBuffer
	{
	public:
		OpenGLFrameBuffer(const FrameBufferDesc& desc);
		virtual ~OpenGLFrameBuffer();

		void Invalidate();

		virtual void Bind() override;
		virtual void Unbind() override;
		virtual unsigned int GetColorAttachment() override;
		virtual void Resize(uint32_t width, uint32_t height) override;
		virtual void ClearBuffer() override;
		virtual void ClearRgba() override;
		virtual void ClearDepthStencil() override;

		// Use As Texture
		virtual Ref<Texture2D> ColorAsTexutre() override;
		virtual Ref<Texture2D> DepthStencilAsTexutre() override;

		virtual const FrameBufferDesc& GetDesc() const override { return m_Desc; }

	private:
		FrameBufferDesc m_Desc;

		unsigned int m_FrameBufferObject = 0;
		unsigned int m_DepthStencilObject = 0;
		unsigned int m_TextureObject = 0;
	};
}