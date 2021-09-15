#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h"

namespace SIByL
{
	class OpenGLResource
	{
	public:
		virtual ~OpenGLResource() = default;
		const unsigned int GetID() { return TextureObject; }
		virtual void Delete();

	protected:
		void Invalide();

	protected:
		unsigned int TextureObject = 0;
		TextureDesc Descriptor;
	};

	class OpenGLRenderTargetResource :public OpenGLResource
	{
	public:
		OpenGLRenderTargetResource(const TextureDesc& desc);
	};

	class OpenGLDepthStencilResource :public OpenGLResource
	{
	public:
		OpenGLDepthStencilResource(const TextureDesc& desc);
	};
}