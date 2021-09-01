#pragma once

#include "Sibyl/Graphic/AbstractAPI/Middle/FrameBuffer.h"

namespace SIByL
{
	class DX12FrameBuffer :public FrameBuffer
	{
	public:
		/////////////////////////////////////////////////////////
		///				    	Constructors		          ///
		DX12FrameBuffer(const FrameBufferDesc& desc);
		virtual ~DX12FrameBuffer();

		/////////////////////////////////////////////////////////
		///				        Manipulator     	          ///
		virtual void Bind() override;
		virtual void Unbind() override;
		virtual void Resize(uint32_t width, uint32_t height) override;
		virtual void ClearBuffer() override;
		virtual void ClearRgba() override;
		virtual void ClearDepthStencil() override;

		/////////////////////////////////////////////////////////
		///				     Fetcher / Setter		          ///
		virtual const FrameBufferDesc& GetDesc() const override { return m_Desc; }
		virtual unsigned int GetColorAttachment() override { return 0; }

		/////////////////////////////////////////////////////////
		///				          Caster		              ///
		virtual Ref<Texture2D> ColorAsTexutre() override;
		virtual Ref<Texture2D> DepthStencilAsTexutre() override;

	private:
		FrameBufferDesc m_Desc;

	};
}