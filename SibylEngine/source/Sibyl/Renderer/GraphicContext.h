#pragma once

#include "Sibyl/Renderer/SwapChain.h"
#include "Sibyl/Renderer/CommandList.h"
#include "Sibyl/Renderer/Synchronizer.h"

namespace SIByL
{
	class GraphicContext
	{
	public:
		virtual void Init() = 0;

	public:
		Ref<SwapChain> GetSwapChain() { return m_SwapChain; }
		Ref<CommandList> GetCommandList() { return m_CommandList; }
		inline Ref<Synchronizer> GetSynchronizer() { return m_Synchronizer; }
		void SetCommandList(Ref<CommandList> cmdList) { m_CommandList.reset(cmdList.get()); }

	protected:
		Ref<SwapChain> m_SwapChain;
		Ref<CommandList> m_CommandList;
		Ref<Synchronizer> m_Synchronizer;
	};
}