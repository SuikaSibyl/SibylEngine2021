#pragma once

namespace SIByL
{
	class SRGPassExecutor;
	class SRGVirtualResource
	{
		SRGVirtualResource* m_Parent;
		SRGPassExecutor* m_First;
		SRGPassExecutor* m_Last;

	};
}