#include "SIByLpch.h"
#include "DrawItemPool.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/DrawItem.h"

namespace SIByL
{
	void DrawItemPool::Reset()
	{
		AvailableDrawItems.insert(
			AvailableDrawItems.end(),
			DrawItems.begin(),
			DrawItems.end());

		DrawItems.clear();
	}

	void DrawItemPool::Push(Ref<DrawItem> item)
	{
		DrawItems.push_back(item);
	}

	Ref<DrawItem> DrawItemPool::Request()
	{
		Ref<DrawItem> ptr = nullptr;
		if (AvailableDrawItems.size() != 0)
		{
			ptr = AvailableDrawItems.back();
			AvailableDrawItems.pop_back();
		}
		else
		{
			ptr = CreateRef<DrawItem>();
		}
		return ptr;
	}

}