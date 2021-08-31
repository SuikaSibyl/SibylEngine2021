#pragma once

#include "Sibyl/Renderer/FrameBuffer.h"

namespace SIByL
{
	class DX12FrameBuffer :public FrameBuffer
	{
	public:
		DX12FrameBuffer(const FrameBufferDesc& desc);

		virtual void Bind() override {}
		virtual void Unbind() override {}
		virtual unsigned int GetColorAttachment() override { return 0; }
		virtual void Resize(uint32_t width, uint32_t height) override {}
		virtual void ClearBuffer() override {}
		virtual void ClearRgba() override {}
		virtual void ClearDepthStencil() override {}
		virtual const FrameBufferDesc& GetDesc() const override { return m_Desc; }


	private:
		FrameBufferDesc m_Desc;
	};
}