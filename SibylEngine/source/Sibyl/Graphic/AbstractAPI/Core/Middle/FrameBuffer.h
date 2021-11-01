#pragma once

#include "FrameBufferTexture.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h"
#include <glm/glm.hpp>

namespace SIByL
{	
	struct FrameBufferDesc
	{
		FrameBufferTexturesFormats Formats;
		unsigned int Width, Height;
		float ScaleX = 1, ScaleY = 1;
	};

	class FrameBuffer
	{
	public:
		static Ref<FrameBuffer> Create(const FrameBufferDesc& desc, const std::string& key);

		virtual void Bind() = 0;
		virtual void Unbind() = 0;
		virtual void ClearBuffer() = 0;
		virtual void SetClearColor(const glm::vec4& color) { ClearColor = color; };
		virtual unsigned int CountColorAttachment() = 0;
		virtual void Resize(uint32_t width, uint32_t height) = 0;
		virtual void* GetColorAttachment(unsigned int index) = 0;
		virtual void* GetDepthStencilAttachment() = 0;
		virtual RenderTarget* GetRenderTarget(unsigned int index) = 0;

		const std::string& GetIdentifier() { return Identifier; }
		void SetIdentifier(const std::string& id) { Identifier = id; }

		std::pair<float, float> GetScale() const { return std::pair<float, float>{ ScaleX, ScaleY }; }
	protected:
		std::string Identifier;
		glm::vec4 ClearColor = glm::vec4(0, 0, 0, 0);
		float ScaleX = 1, ScaleY = 1;
	};
}