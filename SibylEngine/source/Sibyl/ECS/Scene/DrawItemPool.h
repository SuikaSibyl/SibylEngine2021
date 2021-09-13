#pragma once

namespace SIByL
{
	class DrawItem;
	class MeshFilterComponent;

	class DrawItemPool
	{
	public:
		void Reset();
		void Push(Ref<DrawItem> item);
		Ref<DrawItem> Request();

		using iter = std::vector<Ref<DrawItem>>::iterator;
		iter begin() { return DrawItems.begin(); }
		iter end() { return DrawItems.end(); }
		
	private:
		std::vector<Ref<DrawItem>> DrawItems;
		std::vector<Ref<DrawItem>> AvailableDrawItems;
	};
}