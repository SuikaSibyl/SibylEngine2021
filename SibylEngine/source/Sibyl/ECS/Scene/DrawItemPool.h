#pragma once

namespace SIByL
{
	class DrawItem;
	class MeshFilterComponent;

	class DrawItemPool
	{
	public:
		void Reset();
		void Push(DrawItem* item);
		
	private:
		std::vector<DrawItem*> DrawItems;
	};
}