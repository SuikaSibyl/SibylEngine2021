#include "SIByLpch.h"
#include "FrameBufferLibrary.h"

namespace SIByL
{
	FrameBufferLibrary::FrameBufferMap FrameBufferLibrary::m_Mapper;
	
	void FrameBufferLibrary::Register(const std::string& name, Ref<FrameBuffer>)
	{

	}

	void FrameBufferLibrary::Remove(const std::string& name)
	{

	}

	Ref<FrameBuffer> FrameBufferLibrary::Fetch(const std::string& name)
	{
		return m_Mapper[name];
	}
}