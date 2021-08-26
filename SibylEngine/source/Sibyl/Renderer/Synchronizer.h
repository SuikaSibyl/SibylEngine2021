#pragma once

namespace SIByL
{
	class Synchronizer
	{
	public:
		virtual void StartFrame() = 0;
		virtual void EndFrame() = 0;
		virtual bool CheckFinish(UINT64 fence) = 0;
		virtual void ForceSynchronize() = 0;
	};
}