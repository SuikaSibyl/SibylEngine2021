#include "SIByLpch.h"
#include "OpenGLTexture.h"

#include "Sibyl/Graphic/Texture/Image.h"
#include "glad/glad.h"
#include "GLFW/glfw3.h"

namespace SIByL
{
	OpenGLTexture2D::OpenGLTexture2D(const std::string& path)
		:m_Path(path)
	{
		Image image(m_Path);
		m_Width = image.GetWidth();
		m_Height = image.GetHeight();
		m_Channel = image.GetChannel();

		glCreateTextures(GL_TEXTURE_2D, 1, &m_TexID);
		if (image.GetChannel() == 3)
		{
			glTextureStorage2D(m_TexID, 1, GL_RGB8, image.GetWidth(), image.GetHeight());
			
			glTextureSubImage2D(m_TexID, 0, 0, 0, image.GetWidth(), image.GetHeight(),
				GL_RGB, GL_UNSIGNED_BYTE, image.GetData());
			m_Type = Texture2D::Type::R8G8B8;
		}
		else if(image.GetChannel() == 4)
		{
			glTextureStorage2D(m_TexID, 1, GL_RGBA8, image.GetWidth(), image.GetHeight());
			glTextureSubImage2D(m_TexID, 0, 0, 0, (int)m_Width, (int)m_Height,
				GL_RGBA, GL_UNSIGNED_BYTE, image.GetData());
			m_Type = Texture2D::Type::R8G8B8A8;
		}

		glTextureParameteri(m_TexID, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTextureParameteri(m_TexID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glGenerateMipmap(GL_TEXTURE_2D);
	}

	OpenGLTexture2D::~OpenGLTexture2D()
	{
		glDeleteTextures(1, &m_TexID);
	}

	uint32_t OpenGLTexture2D::GetWidth() const
	{
		return uint32_t();
	}

	uint32_t OpenGLTexture2D::GetHeight() const
	{
		return uint32_t();
	}

	void OpenGLTexture2D::Bind(uint32_t slot) const
	{
		glBindTextureUnit(slot, m_TexID);
	}
}