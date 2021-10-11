#include "SIByLpch.h"
#include "OpenGLFrameBufferTexture.h"

#include "Platform/OpenGL/Common/OpenGLContext.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLTexture.h"

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
		case FrameBufferTextureFormat::RGBA16:
			GLType = GL_RGB16;
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

		glTexImage2D(GL_TEXTURE_2D, 0, GLType, Descriptor.Width, Descriptor.Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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

	void OpenGLRenderTarget::SetComputeRenderTarget(unsigned int i)
	{
		glBindImageTexture(0, m_TextureObject, 0, GL_FALSE, 0, GL_WRITE_ONLY, GLType);
	}
	void OpenGLRenderTarget::SetShaderResource(unsigned int i)
	{
		glBindTexture(GL_TEXTURE_2D, m_TextureObject);
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
		case FrameBufferTextureFormat::RGBA16:
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