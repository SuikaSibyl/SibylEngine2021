#pragma once

namespace SIByL
{
	class Color;

	class CommandBuffer
	{
	public:
		CommandBuffer(const std::string& name);

		void BeginSample(const std::string& name);
		void EndSample(const std::string& name);

		void ClearRenderTarget(bool clearDepth, bool clearColor, Color backgroundColor, float depth);

		void Clear();
	private:
		std::string m_Name;
	};
}