#include "SIByLpch.h"
#include "Image.h"

#include "stb_image.h"
#include "Sibyl/Graphic/Core/Renderer/Renderer.h"
#include <Sibyl/Basic/Asset/SCacheAsset.h>

namespace SIByL
{
	class SImageCacheAsset :public SCacheAsset
	{
	public:
		struct ImageHead
		{
			int m_Width;
			int m_Height;
			int m_Channel;
		};

		ImageHead Head;

		SImageCacheAsset(const std::string& assetpath)
			:SCacheAsset(assetpath)
		{

		}

		SImageCacheAsset(const std::string& assetpath,
			int width, int height, int channel, unsigned char* data)
			:SCacheAsset(assetpath)
		{
			Head.m_Width = width;
			Head.m_Height = height;
			Head.m_Channel = channel;
			m_Pixels.resize(sizeof(unsigned char) * width * height * channel);
			memcpy(m_Pixels.data(), data, sizeof(unsigned char) * width * height * channel);
		}

		void LoadData(unsigned char* & data, int* width, int* height, int* channel)
		{
			*width = Head.m_Width;
			*height = Head.m_Height;
			*channel = Head.m_Channel;
			data = (unsigned char*)malloc(sizeof(unsigned char) * Head.m_Width * Head.m_Height * Head.m_Channel);
			memcpy(data, m_Pixels.data(), sizeof(unsigned char) * Head.m_Width * Head.m_Height * Head.m_Channel);
		}

		std::vector<unsigned char> m_Pixels;

		virtual void LoadDataToBuffers() override
		{
			Buffers.resize(1);
			int i = 0;
			Buffers[i].SetExtraHead(Head);
			Buffers[i].LoadFromVector(m_Pixels);
		}

		virtual void LoadDataFromBuffers() override
		{
			int i = 0;
			Buffers[i].LoadExtraHead(Head);
			Buffers[i].LoadToVector(m_Pixels);
		}

	};

	Image::Image(std::string path)
	{
		SImageCacheAsset cache(path);
		if (cache.LoadCache())
		{
			cache.LoadData(m_Data, &m_Width, &m_Height, &m_Channel);
		}
		else
		{
			stbi_set_flip_vertically_on_load(1);

			m_Data = stbi_load(path.c_str(), &m_Width, &m_Height, &m_Channel, 0);
			SImageCacheAsset imageCache(path, m_Width, m_Height, m_Channel, m_Data);
			imageCache.SaveCache();
			SIByL_CORE_ASSERT(m_Data, "Image Loaded from Path Falied!");
		}
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