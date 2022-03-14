module;
#include <glm/glm.hpp>
#include <random>
export module ParticleSystem.PrecomputedSample;

namespace SIByL::ParticleSystem
{
	export struct PrecomputedSample
	{
		std::default_random_engine randEngine;
		std::uniform_real_distribution<float> uniformFloat{ 0.f, 1.f };
	};

	export struct PrecomputedSampleTorus :public PrecomputedSample
	{
		PrecomputedSampleTorus(float R, float r)
			:R(R), r(r) {}

		auto sdf(glm::vec3 const& p)
		{
			float length = glm::length(glm::vec2(p.x, p.z)) - R;
			glm::vec2 q = glm::vec2(length, p.y);
			return glm::length(q) - r;
		}

		auto generateSample() noexcept -> glm::vec3
		{
			glm::vec3 sample;
			do
			{
				float x = 2 * (r + R) * (uniformFloat(randEngine) - 0.5f);
				float y = 2 * r * (uniformFloat(randEngine) - 0.5f);
				float z = 2 * (r + R) * (uniformFloat(randEngine) - 0.5f);
				sample = { x,y,z };
			} while (sdf(sample) > 0);
			return sample;
		}

		float R, r;
	};
}