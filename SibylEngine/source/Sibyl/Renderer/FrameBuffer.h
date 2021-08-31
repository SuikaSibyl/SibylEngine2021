#pragma once

namespace SIByL
{
	enum class FrameBufferFormat
	{
		R8G8B8A8,
		R32G32B32A32,
	};

	struct FrameBufferDesc
	{
		uint32_t Width;
		uint32_t Height;
		FrameBufferFormat Format;
		bool SwapChainTarget = false;
	};

	class FrameBuffer
	{
	public:
		static Ref<FrameBuffer> Create(const FrameBufferDesc& desc);
		~FrameBuffer() {}

		virtual void Bind() = 0;
		virtual void Unbind() = 0;
		virtual void ClearBuffer() = 0;
		virtual void ClearRgba() = 0;
		virtual void ClearDepthStencil() = 0;
		virtual void Resize(uint32_t width, uint32_t height) = 0;
		virtual unsigned int GetColorAttachment() = 0;

		virtual const FrameBufferDesc& GetDesc() const = 0;

	protected:


	};
}