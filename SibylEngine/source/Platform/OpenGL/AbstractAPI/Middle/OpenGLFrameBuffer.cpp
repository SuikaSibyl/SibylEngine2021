#include "SIByLpch.h"
#include "OpenGLFrameBuffer.h"

#include "OpenGLFrameBufferTexture.h"
#include "Platform/OpenGL/Common/OpenGLContext.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLTexture.h"

#ifdef SIBYL_PLATFORM_CUDA
#include <CudaModule/source/CudaModule.h>
#endif // SIBYL_PLATFORM_CUDA

namespace SIByL
{
	OpenGLFrameBuffer_v1::OpenGLFrameBuffer_v1(const FrameBufferDesc_v1& desc)
		:m_Desc(desc)
	{
		Invalidate();
	}

	OpenGLFrameBuffer_v1::~OpenGLFrameBuffer_v1()
	{
		glDeleteFramebuffers(1, &m_FrameBufferObject);
		glDeleteTextures(1, &m_TextureObject);
		glDeleteTextures(1, &m_DepthStencilObject);
	}

	void OpenGLFrameBuffer_v1::Invalidate()
	{
		if (m_FrameBufferObject)
		{
			glDeleteFramebuffers(1, &m_FrameBufferObject);
			glDeleteTextures(1, &m_TextureObject);
			glDeleteTextures(1, &m_DepthStencilObject);
		}

		glGenFramebuffers(1, &m_FrameBufferObject);
		glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBufferObject);

		// Create RGBA Texture and Bind the Framebuffer to it
		// -------------------------------------------------------
		glCreateTextures(GL_TEXTURE_2D, 1, &m_TextureObject);
		glBindTexture(GL_TEXTURE_2D, m_TextureObject);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Desc.Width, m_Desc.Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_TextureObject, 0);

		// Create Depth Texture and Bind the Framebuffer to it
		// -------------------------------------------------------
		glCreateTextures(GL_TEXTURE_2D, 1, &m_DepthStencilObject);
		glBindTexture(GL_TEXTURE_2D, m_DepthStencilObject);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, m_Desc.Width, m_Desc.Height, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, m_DepthStencilObject, 0);

		// End Creation
		// -------------------------------------------------------
		SIByL_CORE_ASSERT(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "Frame Buffer is Incomplete");
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void OpenGLFrameBuffer_v1::Bind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBufferObject);
		glViewport(0, 0, m_Desc.Width, m_Desc.Height);
	}

	void OpenGLFrameBuffer_v1::Unbind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void* OpenGLFrameBuffer_v1::GetColorAttachment()
	{
		return (void*)m_TextureObject;
	}

	void OpenGLFrameBuffer_v1::Resize(uint32_t width, uint32_t height)
	{
		if (width <= 0 || height <= 0)
		{
			return;
		}

		m_Desc.Width = width;
		m_Desc.Height = height;
		Invalidate();
		ResizePtrCudaSurface();
	}

	void OpenGLFrameBuffer_v1::ClearBuffer()
	{
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	void OpenGLFrameBuffer_v1::ClearRgba()
	{
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	void OpenGLFrameBuffer_v1::ClearDepthStencil()
	{
		glClear(GL_DEPTH_BUFFER_BIT);
	}

	Ref<Texture2D> OpenGLFrameBuffer_v1::ColorAsTexutre()
	{
		Ref<Texture2D> texture;
		texture.reset(new OpenGLTexture2D(m_TextureObject, m_Desc.Width, 
			m_Desc.Height, m_Desc.Channel,m_Desc.Format));
		return texture;
	}

	Ref<Texture2D> OpenGLFrameBuffer_v1::DepthStencilAsTexutre()
	{
		Ref<Texture2D> texture;
		texture.reset(new OpenGLTexture2D(m_TextureObject, m_Desc.Width,
			m_Desc.Height, m_Desc.Channel, Texture2D::Format::R24G8));
		return texture;
	}

	////////////////////////////////////////////////////
	//					CUDA Interface				  //
	////////////////////////////////////////////////////

	Ref<PtrCudaTexture> OpenGLFrameBuffer_v1::GetPtrCudaTexture()
	{
		if (mPtrCudaTexture == nullptr)
		{

		}

		return mPtrCudaTexture;
	}

	Ref<PtrCudaSurface> OpenGLFrameBuffer_v1::GetPtrCudaSurface()
	{
		if (mPtrCudaSurface == nullptr)
		{
#ifdef SIBYL_PLATFORM_CUDA
			mPtrCudaSurface = CreateRef<PtrCudaSurface>();
			mPtrCudaSurface->RegisterByOpenGLTexture(m_TextureObject, m_Desc.Width, m_Desc.Height);
#endif // SIBYL_PLATFORM_CUDA
		}

		return mPtrCudaSurface;
	}

	void OpenGLFrameBuffer_v1::ResizePtrCudaTexuture()
	{

	}

	void OpenGLFrameBuffer_v1::ResizePtrCudaSurface()
	{
		if (mPtrCudaSurface != nullptr)
		{
#ifdef SIBYL_PLATFORM_CUDA
			mPtrCudaSurface->RegisterByOpenGLTexture(m_TextureObject, m_Desc.Width, m_Desc.Height);
#endif // SIBYL_PLATFORM_CUDA
		}
	}

	void OpenGLFrameBuffer_v1::CreatePtrCudaTexutre()
	{
		if (mPtrCudaTexture != nullptr)
		{

		}
	}

	void OpenGLFrameBuffer_v1::CreatePtrCudaSurface()
	{

	}

	////////////////////////////////////////////////////////////////
	// 					 OpenGL Frame Buffer					  //
	////////////////////////////////////////////////////////////////
	OpenGLFrameBuffer::OpenGLFrameBuffer(const FrameBufferDesc& desc)
		:Width(desc.Width), Height(desc.Height)
	{
		for (FrameBufferTextureFormat format : desc.Formats.Formats)
		{
			if (IsDepthFormat(format))
			{
				DepthStencil = CreateRef<OpenGLStencilDepth>(FrameBufferTextureDesc{ format,desc.Width,desc.Height });
			}
			else
			{
				RenderTargets.push_back(CreateRef<OpenGLRenderTarget>(FrameBufferTextureDesc{ format,desc.Width,desc.Height }));
			}
		}

		Invalidate();
	}

	void OpenGLFrameBuffer::Bind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBufferObject);
		glViewport(0, 0, Width, Height);
	}

	void OpenGLFrameBuffer::Unbind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void OpenGLFrameBuffer::ClearBuffer()
	{
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	unsigned int OpenGLFrameBuffer::CountColorAttachment()
	{
		return RenderTargets.size();
	}

	void OpenGLFrameBuffer::Resize(uint32_t width, uint32_t height)
	{
		if (width <= 0 || height <= 0)
		{
			return;
		}
		Width = width;
		Height = height;
		for (int i = 0; i < RenderTargets.size(); i++)
			RenderTargets[i]->Resize(width, height);
		DepthStencil->Resize(width, height);

		Invalidate();
	}

	void* OpenGLFrameBuffer::GetColorAttachment(unsigned int index)
	{
		return (void*)*RenderTargets[index]->GetPtrTextureObject();

	}

	void* OpenGLFrameBuffer::GetDepthStencilAttachment()
	{
		return (void*)DepthStencil->GetTextureObject();
	}

	void OpenGLFrameBuffer::Invalidate()
	{
		if (m_FrameBufferObject)
		{
			glDeleteFramebuffers(1, &m_FrameBufferObject);
			// Delete all frame buffers
			for (int i = 0; i < RenderTargets.size(); i++)
				RenderTargets[i]->DeleteObject();
			DepthStencil->DeleteObject();
		}

		glGenFramebuffers(1, &m_FrameBufferObject);
		glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBufferObject);

		// Create RGBA Texture and Bind the Framebuffer to it
		// -------------------------------------------------------
		for (int i = 0; i < RenderTargets.size(); i++)
		{
			RenderTargets[i]->Invalid();
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, RenderTargets[i]->GetTextureObject(), 0);
		}

		// Create Depth Texture and Bind the Framebuffer to it
		// -------------------------------------------------------
		DepthStencil->Invalid();
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, DepthStencil->GetTextureObject(), 0);


		// Test multi-render-targets situations
		// -------------------------------------------------------
		if (RenderTargets.size() > 1)
		{
			SIByL_CORE_ASSERT((RenderTargets.size() <= 4), "Render Targets is more than 4");
			GLenum buffers[4] = { GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1 ,GL_COLOR_ATTACHMENT2 ,GL_COLOR_ATTACHMENT3 };
			glDrawBuffers(RenderTargets.size(), buffers);
		}
		else if (RenderTargets.size() == 0)
		{
			glDrawBuffer(GL_NONE);
		}

		// End Creation
		// -------------------------------------------------------
		SIByL_CORE_ASSERT(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "Frame Buffer is Incomplete");
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

}