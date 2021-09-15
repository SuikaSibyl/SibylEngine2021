#pragma once

namespace SIByL
{
	enum class TextureFormat
	{
		None = 0,
		R8G8B8A8,
		DEPTH24STENCIL8,
	};

	struct TextureDesc
	{
		TextureFormat Format;
		unsigned int Width, Height;
	};

	class Image;

	class Texture
	{
	public:
		virtual ~Texture() = default;
		virtual uint32_t GetWidth() const = 0;
		virtual uint32_t GetHeight() const = 0;

		virtual void Bind(uint32_t slot) const = 0;

	};

	class Texture2D :public Texture
	{
	public:
		enum class Format
		{
			R8G8B8,
			R8G8B8A8,
			R32G32B32A32,
			R24G8,
		};

	public:
		virtual ~Texture2D() = default;
		static Ref<Texture2D> Create(const std::string & path);
		static Ref<Texture2D> Create(Ref<Image> image);

		virtual void RegisterImGui() {}
		virtual void* GetImGuiHandle() = 0;

	public:
		//virtual uint32_t GetImGuiIdentifier() = 0;
	};
}