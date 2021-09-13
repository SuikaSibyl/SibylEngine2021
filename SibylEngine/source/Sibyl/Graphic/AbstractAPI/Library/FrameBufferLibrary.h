#pragma once

namespace SIByL
{
	class FrameBuffer;

	class FrameBufferLibrary
	{
	public:
		static void Register(const std::string& name, Ref<FrameBuffer>);
		static void Remove(const std::string& name);
		static void Reset();
		static Ref<FrameBuffer> Fetch(const std::string& name);

	private:
		using FrameBufferMap = std::unordered_map<std::string, Ref<FrameBuffer>>;
		static FrameBufferMap m_Mapper;
	};
}