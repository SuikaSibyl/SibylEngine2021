#pragma once

#include "Sibyl/Graphic/Texture/Texture.h"

namespace SIByL
{
	class OpenGLTexture2D :public Texture2D
	{
	public:
		OpenGLTexture2D(const std::string& path);
		virtual ~OpenGLTexture2D();

		virtual uint32_t GetWidth() const override;
		virtual uint32_t GetHeight() const override;

		virtual void Bind(uint32_t slot) const override;

	private:
		uint32_t m_Width;
		uint32_t m_Height;
		uint32_t m_Channel;

		uint32_t m_TexID;
		std::string m_Path;
	};
}