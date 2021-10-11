#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBufferTexture.h"

namespace SIByL
{
	class OpenGLRenderTarget :public RenderTarget
	{
	public:
		OpenGLRenderTarget(const FrameBufferTextureDesc& descriptor);
		~OpenGLRenderTarget();

		void Invalid();
		void DeleteObject();
		unsigned int& GetTextureObject() { return m_TextureObject; }
		unsigned int* GetPtrTextureObject() { return &m_TextureObject; }
		unsigned int GetGLType() { return GLType; }
		virtual void Resize(uint32_t width, uint32_t height) override;

		void SetComputeRenderTarget(unsigned int i);
		void SetShaderResource(unsigned int i);

	private:
		unsigned int GLType = 0;
		unsigned int m_TextureObject = 0;
	};

	class OpenGLStencilDepth :public StencilDepth
	{
	public:
		OpenGLStencilDepth(const FrameBufferTextureDesc& descriptor);
		~OpenGLStencilDepth();

		void Invalid();
		void DeleteObject();
		unsigned int GetTextureObject() { return m_TextureObject; }
		unsigned int* GetPtrTextureObject() { return &m_TextureObject; }

		virtual void Resize(uint32_t width, uint32_t height) override;

	private:
		unsigned int GLType = 0;
		unsigned int m_TextureObject = 0;
	};
}