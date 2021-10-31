#include "SIByLpch.h"
#include "FrameBufferLibrary.h"

#include "ResourceLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"

namespace SIByL
{
	void FrameBufferLibrary::ResizeAll(unsigned int width, unsigned int height)
	{
		std::unordered_map<std::string, Ref<FrameBuffer>> frameBuffers = Library<FrameBuffer>::Mapper;
		for (auto iter : frameBuffers)
		{
			auto& [x, y] = iter.second->GetScale();
			iter.second->Resize(width * x, height * y);
		}
	}

	RenderTarget* FrameBufferLibrary::GetRenderTarget(std::string Identifier)
	{
		std::string name = Identifier.substr(0, Identifier.length() - 1);
		unsigned int index = atoi(Identifier.substr(Identifier.length() - 1, 1).c_str());
		Ref<FrameBuffer> frameBuffer = Library<FrameBuffer>::Fetch(name);
		RenderTarget* rendertarget = frameBuffer->GetRenderTarget(index);
		return rendertarget;
	}

}