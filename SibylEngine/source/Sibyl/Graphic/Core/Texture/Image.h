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
		Image(unsigned int width, unsigned int height, unsigned int channels, glm::vec4 color = { 0,0,0,0 });
		~Image();


		void Clear(glm::vec4 color = { 0,0,0,0 });

		inline int GetWidth() { return m_Width; }
		inline int GetHeight() { return m_Height; }
		inline int GetChannel() { return m_Channel; }
		inline unsigned char* GetData() { return m_Data; }
		inline uint32_t GetBufferSize() { return m_BufferSize; }

		void SetPixel(unsigned int x, unsigned int y, RGBPixel color)
		{
			//SIByL_CORE_ASSERT(m_Type == Type::RGB, "");
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
		Type m_Type;

		unsigned char* m_Data;
		uint32_t m_BufferSize;
	};
}