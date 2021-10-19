#include "SIByLpch.h"
#include "LowDiscrepancySequence.h"

namespace SIByL
{
	double RadicalInversion::IntegerRadicalInverse(int Base, int i)
	{
		int numPoints, inverse;
		numPoints = 1;
		// The loop flip i around the point in "Base" system
		for (inverse = 0; i > 0; i /= Base)
		{
			inverse = inverse * Base + (i % Base);
			numPoints = numPoints * Base;
		}
		// Divide Digit mirror the number
		return inverse / (double)numPoints;
	}

	double RadicalInversion::RadicalInverse(int Base, int i)
	{
		double Digit, Radical, Inverse;
		Digit = Radical = 1.0 / (double)Base;
		Inverse = 0.0;
		while (i)
		{
			Inverse += Digit * (double)(i % Base);
			Digit *= Radical;

			i /= Base;
		}
		return Inverse;
	}

	std::pair<double, double> Halton::Halton23(int Index)
	{
		return { RadicalInversion::RadicalInverse(2,Index), RadicalInversion::RadicalInverse(3,Index) };
	}

}