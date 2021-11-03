#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"

namespace SIByL
{
	class DX12DepthStencilResource;
	class DX12RenderTargetResource;

	//class DX12FrameBuffer_v1 :public FrameBuffer_v1
	//{
	//public:
	//	/////////////////////////////////////////////////////////
	//	///				    	Constructors		          ///
	//	DX12FrameBuffer_v1(const FrameBufferDesc_v1& desc);
	//	virtual ~DX12FrameBuffer_v1();

	//	/////////////////////////////////////////////////////////
	//	///				        Manipulator     	          ///
	//	virtual void Bind() override;
	//	virtual void Unbind() override;
	//	virtual void Resize(uint32_t width, uint32_t height) override;
	//	virtual void ClearBuffer() override;
	//	virtual void ClearRgba() override;
	//	virtual void ClearDepthStencil() override;

	//	/////////////////////////////////////////////////////////
	//	///				     Fetcher / Setter		          ///
	//	virtual const FrameBufferDesc_v1& GetDesc() const override { return m_Desc; }
	//	virtual void* GetColorAttachment() override;

	//	/////////////////////////////////////////////////////////
	//	///				          Caster		              ///
	//	virtual Ref<Texture2D> ColorAsTexutre() override;
	//	virtual Ref<Texture2D> DepthStencilAsTexutre() override;

	//private:
	//	void SetViewportRect(int width, int height);

	//private:
	//	FrameBufferDesc_v1 m_Desc;
	//	Ref<DX12DepthStencilResource> m_DepthStencilResource;
	//	Ref<DX12RenderTargetResource> m_RenderTargetResource;

	//	D3D12_VIEWPORT viewPort;
	//	D3D12_RECT scissorRect;


	//	////////////////////////////////////////////////////
	//	//					CUDA Interface				  //
	//	////////////////////////////////////////////////////
	//public:
	//	virtual Ref<PtrCudaTexture> GetPtrCudaTexture() override;
	//	virtual Ref<PtrCudaSurface> GetPtrCudaSurface() override;
	//	virtual void ResizePtrCudaTexuture() override;
	//	virtual void ResizePtrCudaSurface() override;

	//protected:
	//	virtual void CreatePtrCudaTexutre() override;
	//	virtual void CreatePtrCudaSurface() override;

	//};


	class DX12RenderTarget;
	class DX12StencilDepth;
	class DX12FrameBuffer :public FrameBuffer
	{
	public:
		DX12FrameBuffer(const FrameBufferDesc& desc);

		virtual void Bind() override;
		virtual void CustomViewport(unsigned int xmin, unsigned int xmax, unsigned int ymin, unsigned int ymax) override;
		virtual void Unbind() override;
		virtual void ClearBuffer() override;
		virtual unsigned int CountColorAttachment() override;
		virtual void Resize(uint32_t width, uint32_t height) override;
		virtual void* GetColorAttachment(unsigned int index) override;
		virtual void* GetDepthStencilAttachment() override;
		virtual RenderTarget* GetRenderTarget(unsigned int index) override;

	private:
		void SetViewportRect(int width, int height);

	private:
		std::vector<Ref<DX12RenderTarget>> RenderTargets;
		Ref<DX12StencilDepth> DepthStencil;

		unsigned int Width, Height;
		unsigned int m_FrameBufferObject = 0;

		D3D12_VIEWPORT viewPort;
		D3D12_RECT scissorRect;
	};
}