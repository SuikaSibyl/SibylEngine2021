#include "SIByLpch.h"
#include "Image.h"

#include "stb_image.h"
#include "Sibyl/Graphic/Core/Renderer/Renderer.h"

namespace SIByL
{
	Image::Image(std::string path)
	{
		stbi_set_flip_vertically_on_load(1);
		m_Data = stbi_load(path.c_str(), &m_Width, &m_Height, &m_Channel, 0);
		SIByL_CORE_ASSERT(m_Data, "Image Loaded from Path Falied!");
	}

	Image::Image(unsigned int width, unsigned int height, unsigned int channels, glm::vec4 color)
		:m_Width(width), m_Height(height), m_Channel(channels)
	{
		m_BufferSize = width * height * channels * sizeof(unsigned char);
		m_Data = (unsigned char*)malloc(m_BufferSize);
		Clear(color);
	}

	Image::~Image()
	{
		free(m_Data);
		//delete m_Data;
	}

	void Image::Clear(glm::vec4 color)
	{
		uint32_t colFill = 0;
		colFill += ((uint32_t)(uint8_t)(color.a * 255)) << 24;
		colFill += ((uint32_t)(uint8_t)(color.b * 255)) << 16;
		colFill += ((uint32_t)(uint8_t)(color.g * 255)) << 8;
		colFill += ((uint32_t)(uint8_t)(color.r * 255));

		std::fill((uint32_t*)m_Data, (uint32_t*)(m_Data + m_BufferSize), colFill);
	}
}