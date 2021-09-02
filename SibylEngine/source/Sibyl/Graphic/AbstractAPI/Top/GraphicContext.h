#pragma once

#include "Sibyl/Graphic/AbstractAPI/Bottom/Synchronizer.h"
#include "Sibyl/Graphic/AbstractAPI/Middle/CommandList.h"
#include "Sibyl/Graphic/AbstractAPI/Middle/SwapChain.h"

namespace SIByL
{
	class GraphicContext
	{
	public:
		virtual void Init() = 0;
		virtual void OnWindowResize(uint32_t width, uint32_t height) = 0;

	public:
		virtual void StartCommandList() = 0;
		virtual void EndCommandList() = 0;
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