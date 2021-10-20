#pragma once

namespace SIByL
{
	class CuBVH
	{
	public:
		void LoadData(const std::vector<float>& vertices, const std::vector<uint32_t>& indices, unsigned int stepsize);
		
	private:
		void BuildBVH(const std::vector<float>& triangles, const std::vector<float>& bbs);

	};
}