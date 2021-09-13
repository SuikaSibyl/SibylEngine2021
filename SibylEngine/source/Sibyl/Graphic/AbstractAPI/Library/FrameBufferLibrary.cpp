#include "SIByLpch.h"
#include "FrameBufferLibrary.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"

namespace SIByL
{
	FrameBufferLibrary::FrameBufferMap FrameBufferLibrary::m_Mapper;
	
	void FrameBufferLibrary::Register(const std::string& name, Ref<FrameBuffer> buffer)
	{
		if (m_Mapper.find(name) == m_Mapper.end())
		{
			m_Mapper[name] = buffer;
		}
		else
		{
			SIByL_CORE_ERROR("Duplicate Frame Buffer Key!");
		}
	}

	void FrameBufferLibrary::Remove(const std::string& name)
	{
		if (m_Mapper.find(name) != m_Mapper.end())
		{
			m_Mapper.erase(name);
		}
		else
		{
			SIByL_CORE_ERROR("Frame Buffer Key Not Exist!");
		}
	}

	void FrameBufferLibrary::Reset()
	{
		m_Mapper.clear();
	}

	Ref<FrameBuffer> FrameBufferLibrary::Fetch(const std::string& name)
	{
		if (m_Mapper.find(name) != m_Mapper.end())
		{
			return m_Mapper[name];
		}
		else
		{
			SIByL_CORE_ERROR("Frame Buffer Key Not Exist!");
			return nullptr;
		}
	}
}