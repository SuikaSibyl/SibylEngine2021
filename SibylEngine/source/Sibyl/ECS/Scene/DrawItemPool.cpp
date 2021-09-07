#include "SIByLpch.h"
#include "DrawItemPool.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/DrawItem.h"

namespace SIByL
{
	void DrawItemPool::Reset()
	{
		DrawItems.clear();
	}

	void DrawItemPool::Push(DrawItem* item)
	{
		DrawItems.push_back(item);
	}

}