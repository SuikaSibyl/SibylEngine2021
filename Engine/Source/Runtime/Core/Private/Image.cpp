module;
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <cstdint>
#include <filesystem>
module Core.Image;
import Core.Log;
import Core.BitFlag;
import Core.MemoryManager;
import Core.Buffer;
import Core.File;

namespace SIByL::Core
{
    Image::Image(std::filesystem::path path)
    {
        std::filesystem::path asset_path = "./assets";
        path = asset_path / path;
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(path.string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        
        if (!pixels) {
            SE_CORE_ERROR("Image :: failed to load texture image!");
        }
        
        width = (uint32_t)texWidth;
        height = (uint32_t)texHeight;
        channels = (uint32_t)texChannels;
        data = (char*)pixels;
        attributes |= addBit(ImageAttribute::FROM_STB);
    }

    Image::Image(Image&& image)
    {
        attributes = image.attributes;
        width = image.width;
        height = image.height;
        channels = image.channels;
        size = image.size;
        data = image.data;
        image.data = nullptr;
    }

    Image& Image::operator=(Image&& image)
    {
        attributes = image.attributes;
        width = image.width;
        height = image.height;
        channels = image.channels;
        size = image.size;
        data = image.data;
        image.data = nullptr;
        return *this;
    }

    Image::Image(uint32_t width, uint32_t height)
        : width(width)
        , height(height)
    {
        size = width * height * 4 * sizeof(uint8_t);
        data = (char*)MemAlloc(size);
        memset(data, 0, size);
    }

    Image::~Image()
    {
        if (data == 0) return;
        if (hasBit(attributes, ImageAttribute::FROM_STB))
        {
            stbi_image_free((stbi_uc*)data);
        }
        else
        {
            MemFree(data, size);
        }
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

    uint8_t clampColorComponent(float c) {
        int tmp = int(c * 255);
        if (tmp < 0) tmp = 0;
        if (tmp > 255) tmp = 255;
        return (uint8_t)tmp;
    }

    void Image::setPixel(uint32_t x, uint32_t y, const Vec4f& color) 
    {
        uint32_t res = 0;
        res += clampColorComponent(color.z()) << 0;
        res += clampColorComponent(color.y()) << 8;
        res += clampColorComponent(color.x()) << 16;
        res += clampColorComponent(color.w()) << 24;

        ((uint32_t*)data)[y * width + x] = res;
    }

    void Image::savePPM(std::filesystem::path path) const
    {
        // TODO
    }

    auto writeAndMove(uint8_t*& p, uint8_t data)
    {
        p[0] = data;
        p++;
    }

    void Image::saveTGA(std::filesystem::path path) const
    {
        Buffer buffer(size + 18, 1);
        uint8_t* data = (uint8_t*)buffer.getData();

        // misc header information
        for (int i = 0; i < 18; i++) {
            if (i == 2) writeAndMove(data, 2);
            else if (i == 12) writeAndMove(data, width % 256);
            else if (i == 13) writeAndMove(data, width / 256);
            else if (i == 14) writeAndMove(data, height % 256);
            else if (i == 15) writeAndMove(data, height / 256);
            else if (i == 16) writeAndMove(data, 32);
            else if (i == 17) writeAndMove(data, 32 + 8);
            else writeAndMove(data, 0);
        }

        memcpy(data, this->data, size);
        AssetLoader shaderLoader;
        shaderLoader.addSearchPath("./assets/");
        shaderLoader.syncWriteAll(path, buffer);
    }
}