#pragma once

namespace SIByL
{
	class Primitive
	{
	public:
		Primitive() {};
		~Primitive() {};

		virtual void RasterDraw() = 0;

	private:

	};
}