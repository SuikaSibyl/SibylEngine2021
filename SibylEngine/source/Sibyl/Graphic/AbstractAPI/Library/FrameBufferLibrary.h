#pragma once

namespace SIByL
{
	class FrameBuffer_v1;

	class FrameBufferLibrary
	{
	public:
		static void Register(const std::string& name, Ref<FrameBuffer_v1>);
		static void Remove(const std::string& name);
		static void Reset();
		static Ref<FrameBuffer_v1> Fetch(const std::string& name);

	private:
		using FrameBufferMap = std::unordered_map<std::string, Ref<FrameBuffer_v1>>;
		static FrameBufferMap m_Mapper;
	};
}