#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h"

namespace SIByL
{
	/////////////////////////////////////////////////////////////
	//					 Frame Buffer Texture		    	   //
	/////////////////////////////////////////////////////////////
	class FrameBufferTexture
	{
	public:
		TextureDesc Descriptor;

	public:
		virtual ~FrameBufferTexture() = default;
		virtual void Resize(uint32_t width, uint32_t height) = 0;
	};

	/////////////////////////////////////////////////////////////
	//					 Render Target Texture		    	   //
	/////////////////////////////////////////////////////////////
	class RenderTarget :public FrameBufferTexture
	{
	public:
		static Ref<RenderTarget> Create(TextureDesc desc);
		virtual ~RenderTarget() = default;
	};

	/////////////////////////////////////////////////////////////
	//					 Stencil Buffer Texture		    	   //
	/////////////////////////////////////////////////////////////
	class StencilDepth :public FrameBufferTexture
	{
	public:
		static Ref<StencilDepth> Create(TextureDesc desc);
		virtual ~StencilDepth() = default;
	};
}