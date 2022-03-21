module;
#include <vector>
export module Interpolator.Hermite;
import Interpolator.Interpolator;

namespace SIByL::Interpolator
{
	export
	template <class T>
	class HermiteSpline :public IInterpolator
	{
	public:
		auto interpolate(T const& m, T const& n, T const& m_apo, T const& n_apo, float u) noexcept -> T;
		auto generateSamples(T const& m, T const& n, T const& m_apo, T const& n_apo, unsigned int count) noexcept -> std::vector<T>;
	};

	template <class T>
	auto HermiteSpline<T>::interpolate(T const& m, T const& n, T const& m_apo, T const& n_apo, float u) noexcept -> T
	{
		float u2 = u * u;
		float u3 = u2 * u;
		float a1 = 2 * u3 - 3 * u2 + 1;
		float a2 = -2 * u3 + 3 * u2;
		float a3 = u3 - 2 * u2 + u;
		float a4 = u3 - u2;
		return a1 * m + a2 * n + a3 * m_apo + a4 * n_apo;
	}

	template <class T>
	auto HermiteSpline<T>::generateSamples(T const& m, T const& n, T const& m_apo, T const& n_apo, unsigned int count) noexcept -> std::vector<T>
	{
		std::vector<T> res(count);
		float step = 1.f / (count - 1);
		float u = 0;
		for (int i = 0; i < count - 1; i++)
		{
			res[i] = interpolate(m, n, m_apo, n_apo, u);
			u += step;
		}
		res[count - 1] = interpolate(m, n, m_apo, n_apo, u);
		return res;
	}
}