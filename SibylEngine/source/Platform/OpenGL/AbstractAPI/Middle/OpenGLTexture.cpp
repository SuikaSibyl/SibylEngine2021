#include "SIByLpch.h"
#include "OpenGLTexture.h"

#include "Sibyl/Graphic/Core/Texture/Image.h"
#include "glad/glad.h"
#include "GLFW/glfw3.h"

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
		glBindTextureUnit(slot, m_TexID);
	}
}