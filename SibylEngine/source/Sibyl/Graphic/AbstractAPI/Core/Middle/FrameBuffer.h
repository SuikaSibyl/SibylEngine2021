#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h"

namespace SIByL
{
	enum class FrameBufferTextureFormat
	{
		None = 0,
		RGBA8, 
		DEPTH24STENCIL8,
	};

	struct FrameBufferDesc
	{
		uint32_t Width;
		uint32_t Height;
		uint32_t Channel;
		Texture2D::Format Format;
		bool SwapChainTarget = false;
	};

	class FrameBuffer
	{
	public:
		static Ref<FrameBuffer> Create(const FrameBufferDesc& desc, const std::string& key);
		virtual ~FrameBuffer() {}

		virtual void Bind() = 0;
		virtual void Unbind() = 0;
		virtual void ClearBuffer() = 0;
		virtual void ClearRgba() = 0;
		virtual void ClearDepthStencil() = 0;
		virtual void Resize(uint32_t width, uint32_t height) = 0;
		virtual void* GetColorAttachment() = 0;

		// Caster
		// -------------------------------------------------
		virtual Ref<Texture2D> ColorAsTexutre() = 0;
		virtual Ref<Texture2D> DepthStencilAsTexutre() = 0;

		// Fetcher
		// -------------------------------------------------
		virtual const FrameBufferDesc& GetDesc() const = 0;
	};
}