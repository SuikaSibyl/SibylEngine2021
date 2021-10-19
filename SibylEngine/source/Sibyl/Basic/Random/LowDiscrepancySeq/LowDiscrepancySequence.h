#pragma once

namespace SIByL
{
	class RadicalInversion
	{
	public:
		static double IntegerRadicalInverse(int Base, int i);
		static double RadicalInverse(int Base, int i);
	};

	class Halton
	{
	public:
		static std::pair<double, double> Halton23(int Index);
	};
}