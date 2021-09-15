#include "SIByLpch.h"
#include "OpenGLCommonResource.h"

#include "Platform/OpenGL/Common/OpenGLContext.h"

namespace SIByL
{
	void OpenGLResource::Delete()
	{
		glDeleteTextures(1, &TextureObject);
	}

	void OpenGLResource::Invalide()
	{
		glCreateTextures(GL_TEXTURE_2D, 1, &TextureObject);
		glBindTexture(GL_TEXTURE_2D, TextureObject);

		switch (Descriptor.Format)
		{
		case TextureFormat::R8G8B8A8:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, Descriptor.Width, Descriptor.Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
			break;
		case TextureFormat::DEPTH24STENCIL8:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, Descriptor.Width, Descriptor.Height, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);
			break;
		default:
			break;
		}

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	OpenGLRenderTargetResource::OpenGLRenderTargetResource(const TextureDesc& desc)
	{
		Descriptor = desc;
		Invalide();
	}

	OpenGLDepthStencilResource::OpenGLDepthStencilResource(const TextureDesc& desc)
	{
		Descriptor = desc;
		Invalide();
	}
}