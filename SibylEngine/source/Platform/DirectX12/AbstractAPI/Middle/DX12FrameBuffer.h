#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"

namespace SIByL
{
	class DX12DepthStencilResource;
	class DX12RenderTargetResource;

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
		virtual void* GetColorAttachment() override;

		/////////////////////////////////////////////////////////
		///				          Caster		              ///
		virtual Ref<Texture2D> ColorAsTexutre() override;
		virtual Ref<Texture2D> DepthStencilAsTexutre() override;

	private:
		void SetViewportRect(int width, int height);

	private:
		FrameBufferDesc m_Desc;
		Ref<DX12DepthStencilResource> m_DepthStencilResource;
		Ref<DX12RenderTargetResource> m_RenderTargetResource;

		D3D12_VIEWPORT viewPort;
		D3D12_RECT scissorRect;
	};
}