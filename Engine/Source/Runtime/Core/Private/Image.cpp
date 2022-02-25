module;
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <cstdint>
#include <filesystem>
module Core.Image;
import Core.Log;

namespace SIByL::Core
{
    Image::Image(std::filesystem::path path)
    {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(path.string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        
        if (!pixels) {
            SE_CORE_ERROR("Image :: failed to load texture image!");
        }
        
        width = (uint32_t)texWidth;
        height = (uint32_t)texHeight;
        channels = (uint32_t)texChannels;
        data = (char*)pixels;
    }

    Image::~Image()
    {
        stbi_image_free((stbi_uc*)data);
    }

    auto Image::getWidth() noexcept -> uint32_t 
    {
        return width;
    }
    auto Image::getHeight() noexcept -> uint32_t
    {
        return height;
    }
    auto Image::getChannels() noexcept -> uint32_t
    {
        return channels;
    }
    auto Image::getSize() noexcept -> uint32_t
    {
        return width * height * 4;
    }
    auto Image::getData() noexcept -> char*
    {
        return data;
    }

}