#pragma once

namespace SIByL
{
	class CuBVH;
	class ICuBVH
	{
	public:
		ICuBVH();
		~ICuBVH();

		void LoadData(const std::vector<float>& vertices, const std::vector<uint32_t>& indices, unsigned int vnum);

	private:
		CuBVH* p_CuBVH;
	};
}