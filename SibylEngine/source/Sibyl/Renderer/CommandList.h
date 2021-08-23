#pragma once

namespace SIByL
{
	class CommandList
	{
	public:
		virtual void Restart() = 0;
		virtual void Execute() = 0;

	};
}