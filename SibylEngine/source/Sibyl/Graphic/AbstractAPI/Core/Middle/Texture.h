#pragma once

namespace SIByL
{
	class PtrCudaTexture;
	class PtrCudaSurface;

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
		static Ref<Texture2D> Create(Ref<Image> image, const std::string ID);

		virtual void RegisterImGui() {}
		virtual void* GetImGuiHandle() = 0;

		std::string Identifer;

	////////////////////////////////////////////////////
	//					CUDA Interface				  //
	////////////////////////////////////////////////////
	public:
		virtual Ref<PtrCudaTexture> GetPtrCudaTexture() = 0;
		virtual Ref<PtrCudaSurface> GetPtrCudaSurface() = 0;
		virtual void ResizePtrCudaTexuture() = 0;
		virtual void ResizePtrCudaSurface() = 0;

	protected:
		virtual void CreatePtrCudaTexutre() = 0;
		virtual void CreatePtrCudaSurface() = 0;

		Ref<PtrCudaTexture> mPtrCudaTexture = nullptr;
		Ref<PtrCudaSurface> mPtrCudaSurface = nullptr;
	};
}