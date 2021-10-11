#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h"

namespace SIByL
{
	class Image;

	class OpenGLTexture2D :public Texture2D
	{
	public:
		friend class OpenGLFrameBuffer_v1;
		OpenGLTexture2D(const std::string& path);
		OpenGLTexture2D(Ref<Image> image);
		void InitFromImage(Image* img);
		OpenGLTexture2D(const uint32_t& id, const uint32_t& width, 
			const uint32_t& height, const uint32_t& channel, const Format& type);

		uint32_t GetID() { return m_TexID; }

		virtual ~OpenGLTexture2D();

		virtual uint32_t GetWidth() const override;
		virtual uint32_t GetHeight() const override;

		virtual void Bind(uint32_t slot) const override;

		virtual void* GetImGuiHandle() override;

	private:
		uint32_t m_Width;
		uint32_t m_Height;
		uint32_t m_Channel;
		Format	 m_Type;

		uint32_t m_TexID;
		std::string m_Path;

		////////////////////////////////////////////////////
		//					CUDA Interface				  //
		////////////////////////////////////////////////////
	public:
		virtual Ref<PtrCudaTexture> GetPtrCudaTexture() override;
		virtual Ref<PtrCudaSurface> GetPtrCudaSurface() override;
		virtual void ResizePtrCudaTexuture() override;
		virtual void ResizePtrCudaSurface() override;

	protected:
		virtual void CreatePtrCudaTexutre() override;
		virtual void CreatePtrCudaSurface() override;

	};
}