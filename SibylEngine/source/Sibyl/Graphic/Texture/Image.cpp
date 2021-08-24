#include "SIByLpch.h"
#include "Image.h"

#include "stb_image.h"
#include "Sibyl/Renderer/Renderer.h"

namespace SIByL
{
	Image::Image(std::string path)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: stbi_set_flip_vertically_on_load(1); break;
		case RasterRenderer::DirectX12: stbi_set_flip_vertically_on_load(0); break;
		}
		m_Data = stbi_load(path.c_str(), &m_Width, &m_Height, &m_Channel, 0);
		SIByL_CORE_ASSERT(m_Data, "Image Loaded from Path Falied!");
	}

	Image::Image(unsigned int width, unsigned int height, unsigned int channels)
	{

	}

	Image::~Image()
	{
		stbi_image_free(m_Data);
		//delete m_Data;
	}
}