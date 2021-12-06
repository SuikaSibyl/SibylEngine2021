#include "SIByLpch.h"
#include "OpenGLFrameBufferTexture.h"

#include "Platform/OpenGL/Common/OpenGLContext.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLTexture.h"

#include "CudaModule/source/GraphicInterop/TextureInterface.h"
#include "CudaModule/source/RayTracer/RayTracerInterface.h"

namespace SIByL
{
	// ==========================================================
	// 	   OpenGL Render Target API
	// ==========================================================
	OpenGLRenderTarget::OpenGLRenderTarget(const FrameBufferTextureDesc& descriptor)
	{
		Descriptor = descriptor;
		switch (Descriptor.Format)
		{
		case FrameBufferTextureFormat::None:
			break;
		case FrameBufferTextureFormat::RGB8:
			GLType = GL_RGBA8;
			break;
		case FrameBufferTextureFormat::RGB16F:
			GLType = GL_RGB16F;
			break;
		case FrameBufferTextureFormat::R16G16F:
			GLType = GL_RG16F;
			break;
		case FrameBufferTextureFormat::DEPTH24STENCIL8:
			GLType = GL_DEPTH24_STENCIL8;
			break;
		default:
			break;
		}
	}

	OpenGLRenderTarget::~OpenGLRenderTarget()
	{
		DeleteObject();
	}

	void OpenGLRenderTarget::Invalid()
	{
		glCreateTextures(GL_TEXTURE_2D, 1, &m_TextureObject);
		glBindTexture(GL_TEXTURE_2D, m_TextureObject);
		
		switch (Descriptor.Format)
		{
		case FrameBufferTextureFormat::RGB8:
			glTexImage2D(GL_TEXTURE_2D, 0, GLType, Descriptor.Width, Descriptor.Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
			break;
		case FrameBufferTextureFormat::R16G16F:
			glTexImage2D(GL_TEXTURE_2D, 0, GLType, Descriptor.Width, Descriptor.Height, 0, GL_RG, GL_FLOAT, 0);
			break;
		case FrameBufferTextureFormat::RGB16F:
			glTexImage2D(GL_TEXTURE_2D, 0, GLType, Descriptor.Width, Descriptor.Height, 0, GL_RGB, GL_FLOAT, 0);
			break;

		}

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		if (ptrCudaSurface)
		{
			InvalidCudaSurface();
		}
	}

	void OpenGLRenderTarget::DeleteObject()
	{
		glDeleteTextures(1, &m_TextureObject);
	}

	void OpenGLRenderTarget::Resize(uint32_t width, uint32_t height)
	{
		Descriptor.Width = width;
		Descriptor.Height = height;
	}

	void OpenGLRenderTarget::SetComputeRenderTarget(unsigned int i, unsigned int miplevel)
	{
		glBindImageTexture(0, m_TextureObject, 0, GL_FALSE, miplevel, GL_WRITE_ONLY, GLType);
	}

	void OpenGLRenderTarget::SetShaderResource(unsigned int i)
	{
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, m_TextureObject);
	}


	void OpenGLRenderTarget::InvalidCudaSurface()
	{
#ifdef SIBYL_PLATFORM_CUDA
		if (!ptrCudaSurface) ptrCudaSurface = CreateScope<PtrCudaSurface>();
		ptrCudaSurface->RegisterByOpenGLTexture(m_TextureObject, Descriptor.Width, Descriptor.Height);
#endif // SIBYL_PLATFORM_CUDA
	}

	// ==========================================================
	// 	   OpenGL Depth Stencil API
	// ==========================================================
	OpenGLStencilDepth::OpenGLStencilDepth(const FrameBufferTextureDesc& descriptor)
	{
		Descriptor = descriptor;
		switch (Descriptor.Format)
		{
		case FrameBufferTextureFormat::None:
			break;
		case FrameBufferTextureFormat::RGB8:
			GLType = GL_RGBA8;
			break;
		case FrameBufferTextureFormat::RGB16F:
			GLType = GL_RGB16;
			break;
		case FrameBufferTextureFormat::DEPTH24STENCIL8:
			GLType = GL_DEPTH24_STENCIL8;
			break;
		default:
			break;
		}
	}

	OpenGLStencilDepth::~OpenGLStencilDepth()
	{
		DeleteObject();
	}

	void OpenGLStencilDepth::Invalid()
	{
		// Create Depth Texture and Bind the Framebuffer to it
		// -------------------------------------------------------
		glCreateTextures(GL_TEXTURE_2D, 1, &m_TextureObject);
		glBindTexture(GL_TEXTURE_2D, m_TextureObject);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, Descriptor.Width, Descriptor.Height, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void OpenGLStencilDepth::DeleteObject()
	{
		glDeleteTextures(1, &m_TextureObject);
	}

	void OpenGLStencilDepth::Resize(uint32_t width, uint32_t height)
	{
		Descriptor.Width = width;
		Descriptor.Height = height;
	}
}