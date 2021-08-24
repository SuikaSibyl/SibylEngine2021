#pragma once

#include "glm/glm.hpp"

namespace SIByL
{
	struct RGBPixel
	{
		glm::vec3 rgb;
		RGBPixel(float r, float g, float b)
		{
			rgb.r = r; rgb.g = g; rgb.b = b;
		}
	};

	struct RGBAPixel
	{
		glm::vec4 rgba;
	};

	class Image
	{
	public:
		enum class Type
		{
			RGB,
			RGBA,
		};

	public:
		Image(std::string path);
		Image(unsigned int width, unsigned int height, unsigned int channels);
		~Image();

		inline int GetWidth() { return m_Width; }
		inline int GetHeight() { return m_Height; }
		inline int GetChannel() { return m_Channel; }
		inline unsigned char* GetData() { return m_Data; }

		void SetPixel(unsigned int x, unsigned int y, RGBPixel color)
		{
			uint32_t stride = 3 * sizeof(unsigned char);
			uint32_t offset = (y * m_Width + x) * m_Channel;
			m_Data[offset + 0] = uint8_t(color.rgb.r * 255);
			m_Data[offset + 1] = uint8_t(color.rgb.g * 255);
			m_Data[offset + 2] = uint8_t(color.rgb.b * 255);
		}

	private:
		int m_Width;
		int m_Height;
		int m_Channel;

		unsigned char* m_Data;
	};
}