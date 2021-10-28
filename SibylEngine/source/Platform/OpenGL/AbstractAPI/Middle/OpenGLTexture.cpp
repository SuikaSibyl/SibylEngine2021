#include "SIByLpch.h"
#include "OpenGLTexture.h"

#include "Sibyl/Graphic/Core/Texture/Image.h"
#include "glad/glad.h"
#include "GLFW/glfw3.h"

#ifdef SIBYL_PLATFORM_CUDA
#include <CudaModule/source/CudaModule.h>
#endif // SIBYL_PLATFORM_CUDA

namespace SIByL
{
	///////////////////////////////////////////////////////////////////////////
	///                      Constructors / Destructors                     ///
	///////////////////////////////////////////////////////////////////////////
	OpenGLTexture2D::OpenGLTexture2D(Ref<Image> image)
		:m_Path("NONE")
	{
		PROFILE_SCOPE_FUNCTION();
		InitFromImage(image.get());
	}

	// Create from local file path
	// --------------------------------
	OpenGLTexture2D::OpenGLTexture2D(const std::string& path)
		:m_Path(path)
	{
		PROFILE_SCOPE_FUNCTION();
		Image image(m_Path);
		InitFromImage(&image);
	}

	void OpenGLTexture2D::InitFromImage(Image* img)
	{
		m_Width = img->GetWidth();
		m_Height = img->GetHeight();
		m_Channel = img->GetChannel();

		glCreateTextures(GL_TEXTURE_2D, 1, &m_TexID);
		if (img->GetChannel() == 3)
		{
			glTextureStorage2D(m_TexID, 1, GL_RGB8, img->GetWidth(), img->GetHeight());
			
			glTextureSubImage2D(m_TexID, 0, 0, 0, img->GetWidth(), img->GetHeight(),
				GL_RGB, GL_UNSIGNED_BYTE, img->GetData());
			m_Type = Texture2D::Format::R8G8B8;
		}
		else if(img->GetChannel() == 4)
		{
			glTextureStorage2D(m_TexID, 1, GL_RGBA8, img->GetWidth(), img->GetHeight());
			glTextureSubImage2D(m_TexID, 0, 0, 0, (int)m_Width, (int)m_Height,
				GL_RGBA, GL_UNSIGNED_BYTE, img->GetData());
			m_Type = Texture2D::Format::R8G8B8A8;
		}

		glTextureParameteri(m_TexID, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTextureParameteri(m_TexID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glGenerateMipmap(GL_TEXTURE_2D);
	}

	// Create from known texture, for example, from Framebuffer
	// --------------------------------
	OpenGLTexture2D::OpenGLTexture2D(const uint32_t& id, const uint32_t& width,
		const uint32_t& height, const uint32_t& channel, const Format& type)
		:m_TexID(id), m_Width(width), m_Height(height), m_Channel(channel),
		m_Path("NONE"), m_Type(type)
	{
	}

	OpenGLTexture2D::~OpenGLTexture2D()
	{
		glDeleteTextures(1, &m_TexID);
	}

	///////////////////////////////////////////////////////////////////////////
	///                               Fetcher                               ///
	///////////////////////////////////////////////////////////////////////////
	uint32_t OpenGLTexture2D::GetWidth() const
	{
		return uint32_t();
	}

	uint32_t OpenGLTexture2D::GetHeight() const
	{
		return uint32_t();
	}

	///////////////////////////////////////////////////////////////////////////
	///                                   ?                                 ///
	///////////////////////////////////////////////////////////////////////////
	void OpenGLTexture2D::Bind(uint32_t slot) const
	{
		glActiveTexture(GL_TEXTURE0 + slot);
		glBindTexture(GL_TEXTURE_2D, m_TexID);
	}
	
	void* OpenGLTexture2D::GetImGuiHandle()
	{
		return (void*)m_TexID;
	}


	////////////////////////////////////////////////////
	//					CUDA Interface				  //
	////////////////////////////////////////////////////

	Ref<PtrCudaTexture> OpenGLTexture2D::GetPtrCudaTexture()
	{
		if (mPtrCudaTexture == nullptr)
		{

		}

		return mPtrCudaTexture;
	}

	Ref<PtrCudaSurface> OpenGLTexture2D::GetPtrCudaSurface()
	{
		if (mPtrCudaSurface == nullptr)
		{
#ifdef SIBYL_PLATFORM_CUDA
			mPtrCudaSurface = CreateRef<PtrCudaSurface>();
			mPtrCudaSurface->RegisterByOpenGLTexture(m_TexID, m_Width, m_Height);
#endif // SIBYL_PLATFORM_CUDA
		}

		return mPtrCudaSurface;
	}

	void OpenGLTexture2D::ResizePtrCudaTexuture()
	{

	}

	void OpenGLTexture2D::ResizePtrCudaSurface()
	{
		if (mPtrCudaSurface != nullptr)
		{
#ifdef SIBYL_PLATFORM_CUDA
			mPtrCudaSurface->RegisterByOpenGLTexture(m_TexID, m_Width, m_Height);
#endif // SIBYL_PLATFORM_CUDA
		}
	}

	void OpenGLTexture2D::CreatePtrCudaTexutre()
	{
		if (mPtrCudaTexture != nullptr)
		{

		}
	}

	void OpenGLTexture2D::CreatePtrCudaSurface()
	{

	}


	///////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////
	///										Cubemap									///
	///////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////

	// Create from local file path
	// --------------------------------
	OpenGLTextureCubemap::OpenGLTextureCubemap(const std::string& path)
		:m_Path(path)
	{
		PROFILE_SCOPE_FUNCTION();
		InitFromImage(path);
	}

	void OpenGLTextureCubemap::InitFromImage(const std::string& path)
	{
		static std::string tails[6] = {
			"_pos_x.png",
			"_neg_x.png",
			"_pos_y.png",
			"_neg_y.png",
			"_pos_z.png",
			"_neg_z.png"};

		// Generate Texture
		glGenTextures(1, &m_TexID);
		glBindTexture(GL_TEXTURE_CUBE_MAP, m_TexID);

		std::string head = path.substr(0, path.find_last_of("."));
		for (unsigned int i = 0; i < 6; i++)
		{
			Image image(head + tails[i]);
			if (i == 0)
			{
				m_Width = image.GetWidth();
				m_Height = image.GetHeight();
				m_Channel = image.GetChannel();
			}
			if (image.GetChannel() == 3)
			{
				glTexImage2D(
					GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
					0, GL_RGB, image.GetWidth(), image.GetHeight(), 0, GL_RGB, GL_UNSIGNED_BYTE, image.GetData());
			}
			else if (image.GetChannel() == 4)
			{
				glTexImage2D(
					GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
					0, GL_RGBA, image.GetWidth(), image.GetHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, image.GetData());
			}

			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

			// generate mipmaps for the cubemap so OpenGL automatically allocates the required memory.
			glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
		}

	}

	OpenGLTextureCubemap::~OpenGLTextureCubemap()
	{
		glDeleteTextures(1, &m_TexID);
	}

	///////////////////////////////////////////////////////////////////////////
	///                               Fetcher                               ///
	///////////////////////////////////////////////////////////////////////////
	uint32_t OpenGLTextureCubemap::GetWidth() const
	{
		return m_Width;
	}

	uint32_t OpenGLTextureCubemap::GetHeight() const
	{
		return m_Height;
	}

	///////////////////////////////////////////////////////////////////////////
	///                                   ?                                 ///
	///////////////////////////////////////////////////////////////////////////
	void OpenGLTextureCubemap::Bind(uint32_t slot) const
	{
		glActiveTexture(GL_TEXTURE0 + slot);
		glBindTexture(GL_TEXTURE_CUBE_MAP, m_TexID);
	}

	void* OpenGLTextureCubemap::GetImGuiHandle()
	{
		return (void*)m_TexID;
	}

}