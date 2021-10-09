#include "SIByLpch.h"
#include "OpenGLFrameBuffer.h"

#include "Platform/OpenGL/Common/OpenGLContext.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLTexture.h"

#ifdef SIBYL_PLATFORM_CUDA
#include <CudaModule/source/CudaModule.h>
#endif // SIBYL_PLATFORM_CUDA

namespace SIByL
{
	OpenGLFrameBuffer::OpenGLFrameBuffer(const FrameBufferDesc& desc)
		:m_Desc(desc)
	{
		Invalidate();
	}

	OpenGLFrameBuffer::~OpenGLFrameBuffer()
	{
		glDeleteFramebuffers(1, &m_FrameBufferObject);
		glDeleteTextures(1, &m_TextureObject);
		glDeleteTextures(1, &m_DepthStencilObject);
	}

	void OpenGLFrameBuffer::Invalidate()
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

	void OpenGLFrameBuffer::Bind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBufferObject);
		glViewport(0, 0, m_Desc.Width, m_Desc.Height);
	}

	void OpenGLFrameBuffer::Unbind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void* OpenGLFrameBuffer::GetColorAttachment()
	{
		return (void*)m_TextureObject;
	}

	void OpenGLFrameBuffer::Resize(uint32_t width, uint32_t height)
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

	void OpenGLFrameBuffer::ClearBuffer()
	{
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	void OpenGLFrameBuffer::ClearRgba()
	{
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	void OpenGLFrameBuffer::ClearDepthStencil()
	{
		glClear(GL_DEPTH_BUFFER_BIT);
	}

	Ref<Texture2D> OpenGLFrameBuffer::ColorAsTexutre()
	{
		Ref<Texture2D> texture;
		texture.reset(new OpenGLTexture2D(m_TextureObject, m_Desc.Width, 
			m_Desc.Height, m_Desc.Channel,m_Desc.Format));
		return texture;
	}

	Ref<Texture2D> OpenGLFrameBuffer::DepthStencilAsTexutre()
	{
		Ref<Texture2D> texture;
		texture.reset(new OpenGLTexture2D(m_TextureObject, m_Desc.Width,
			m_Desc.Height, m_Desc.Channel, Texture2D::Format::R24G8));
		return texture;
	}

	////////////////////////////////////////////////////
	//					CUDA Interface				  //
	////////////////////////////////////////////////////

	Ref<PtrCudaTexture> OpenGLFrameBuffer::GetPtrCudaTexture()
	{
		if (mPtrCudaTexture == nullptr)
		{

		}

		return mPtrCudaTexture;
	}

	Ref<PtrCudaSurface> OpenGLFrameBuffer::GetPtrCudaSurface()
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

	void OpenGLFrameBuffer::ResizePtrCudaTexuture()
	{

	}

	void OpenGLFrameBuffer::ResizePtrCudaSurface()
	{
		if (mPtrCudaSurface != nullptr)
		{
#ifdef SIBYL_PLATFORM_CUDA
			mPtrCudaSurface->RegisterByOpenGLTexture(m_TextureObject, m_Desc.Width, m_Desc.Height);
#endif // SIBYL_PLATFORM_CUDA
		}
	}

	void OpenGLFrameBuffer::CreatePtrCudaTexutre()
	{
		if (mPtrCudaTexture != nullptr)
		{

		}
	}

	void OpenGLFrameBuffer::CreatePtrCudaSurface()
	{

	}
}