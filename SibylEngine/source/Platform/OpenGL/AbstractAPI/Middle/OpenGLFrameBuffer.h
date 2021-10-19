#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"

namespace SIByL
{
	class OpenGLFrameBuffer_v1 :public FrameBuffer_v1
	{
	public:
		OpenGLFrameBuffer_v1(const FrameBufferDesc_v1& desc);
		virtual ~OpenGLFrameBuffer_v1();

		void Invalidate();

		virtual void Bind() override;
		virtual void Unbind() override;
		virtual void* GetColorAttachment() override;
		virtual void Resize(uint32_t width, uint32_t height) override;
		virtual void ClearBuffer() override;
		virtual void ClearRgba() override;
		virtual void ClearDepthStencil() override;

		// Use As Texture
		virtual Ref<Texture2D> ColorAsTexutre() override;
		virtual Ref<Texture2D> DepthStencilAsTexutre() override;

		virtual const FrameBufferDesc_v1& GetDesc() const override { return m_Desc; }

	private:
		FrameBufferDesc_v1 m_Desc;

		unsigned int m_FrameBufferObject = 0;
		unsigned int m_DepthStencilObject = 0;
		unsigned int m_TextureObject = 0;


		////////////////////////////////////////////////////
		//					CUDA Interface				  //
		////////////////////////////////////////////////////
	public:
		virtual Ref<PtrCudaTexture> GetPtrCudaTexture() override;
		virtual Ref<PtrCudaSurface> GetPtrCudaSurface() override;
		virtual void ResizePtrCudaTexuture() override;
		virtual void ResizePtrCudaSurface() override;

	protected:
		virtual void CreatePtrCudaTexutre() override;
		virtual void CreatePtrCudaSurface() override;
	};

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