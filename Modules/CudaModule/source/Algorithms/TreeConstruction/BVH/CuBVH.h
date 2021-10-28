#pragma once

namespace SIByL
{
	struct Triangle;

	class CuBVH
	{
	public:
		void LoadData(const std::vector<float>& vertices, const std::vector<uint32_t>& indices, unsigned int stepsize);
		
	private:
		void BuildBVH(const std::vector<Triangle>& triangles, const std::vector<float>& bbs);

	};
}