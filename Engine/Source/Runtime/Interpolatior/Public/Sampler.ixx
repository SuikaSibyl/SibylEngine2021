module;
#include <vector>
#include <glm/glm.hpp>
export module Interpolator.Sampler;

namespace SIByL::Interpolator
{
	export class Sampler
	{
	public:
		auto LinearSampleCurveUniform01(std::vector<glm::vec2> const& samples, unsigned int count) noexcept -> std::vector<float>;
	};

	auto Sampler::LinearSampleCurveUniform01(std::vector<glm::vec2> const& samples, unsigned int count) noexcept -> std::vector<float>
	{
		std::vector<float> uniform_samples(count);
		float step = 1.f / (count - 1);
		float offset = 0;
		unsigned int left_sample = -1;
		unsigned int right_sample = 0;

		for (int i = 0; i < count; i++)
		{
			while (right_sample != samples.size() && samples[right_sample].x < offset)
			{
				left_sample++; right_sample++;
			}
			if (left_sample == -1)
				uniform_samples[i] = samples[right_sample].y;
			else if (right_sample == samples.size())
				uniform_samples[i] = samples[left_sample].y;
			else
			{
				uniform_samples[i] = samples[left_sample].y 
					+ (samples[right_sample].y - samples[left_sample].y) 
					* (offset - samples[left_sample].x) 
					/ (samples[right_sample].x - samples[left_sample].x);
			}
			offset += step;
		}
		return uniform_samples;
	}
}