#pragma once

namespace SIByL
{
	class GraphicContext
	{
	public:
		virtual void Init() = 0;
		virtual void SwipBuffers() = 0;
	};
}