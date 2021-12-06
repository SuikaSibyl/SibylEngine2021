#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h"

#include "CudaModule/source/GraphicInterop/TextureInterface.h"

namespace SIByL
{
	/////////////////////////////////////////////////////////////
	//					 Frame Buffer Texture		    	   //
	/////////////////////////////////////////////////////////////

	enum class FrameBufferTextureFormat
	{
		None,
		RGB8,
		RGB16F,
		R16G16F,
		DEPTH24STENCIL8,
		DEPTH32F,
	};

	static bool IsDepthFormat(FrameBufferTextureFormat format)
	{
		if (format == FrameBufferTextureFormat::DEPTH24STENCIL8)
			return true;
		return false;
	}

	struct FrameBufferTextureDesc
	{
		FrameBufferTextureFormat Format;
		unsigned int Width, Height;
	};

	struct FrameBufferTexturesFormats
	{
		FrameBufferTexturesFormats() = default;
		FrameBufferTexturesFormats(const std::initializer_list<FrameBufferTextureFormat> formats)
			:Formats(formats) {}

		std::vector<FrameBufferTextureFormat> Formats;
	};

	class FrameBufferTexture
	{
	public:
		FrameBufferTextureDesc Descriptor;

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
		static Ref<RenderTarget> Create(FrameBufferTextureDesc desc);
		virtual ~RenderTarget() = default;

	public:
		const std::string& GetIdentifier() { return Identifier; }
		void SetIdentifier(const std::string& id) { Identifier = id; }

	public:
		PtrCudaSurface* GetCudaSurface() { return ptrCudaSurface.get(); }
		virtual void InvalidCudaSurface() {}

	private:
		std::string Identifier;

	protected:
		Scope<PtrCudaSurface> ptrCudaSurface;

	};

	/////////////////////////////////////////////////////////////
	//					 Stencil Buffer Texture		    	   //
	/////////////////////////////////////////////////////////////
	class StencilDepth :public FrameBufferTexture
	{
	public:
		static Ref<StencilDepth> Create(FrameBufferTextureDesc desc);
		virtual ~StencilDepth() = default;

	public:
		const std::string& GetIdentifier() { return Identifier; }
		void SetIdentifier(const std::string& id) { Identifier = id; }
	private:
		std::string Identifier;
	};
}