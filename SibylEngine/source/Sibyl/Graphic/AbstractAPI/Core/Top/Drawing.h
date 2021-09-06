#pragma once

namespace SIByL
{
	class Camera;
	using ShaderTagId = uint64_t;

	enum class SortingCriteria
	{
		CommonOpaque,
	};
	struct SortingSettings
	{
		SortingSettings(Camera* camera) {};
		SortingCriteria criteria;
	};

	struct DrawingSettings
	{
		DrawingSettings() = default;
		DrawingSettings(const ShaderTagId& shaderTagId, const SortingSettings& sortingSetting) {}
	};

	struct Range
	{
		uint16_t lowerBound = 0;
		uint16_t upperBound = 0;
	};

	class RenderQueueRange
	{
	public:
		static RenderQueueRange all;
		static RenderQueueRange opaque;
		static RenderQueueRange transparent;

		RenderQueueRange() = default;
		RenderQueueRange(const uint16_t& low, const uint16_t& upper)
			:lowerBound(low), upperBound(upper) {}

	private:
		uint16_t lowerBound = 0;
		uint16_t upperBound = 0;
	};

	struct FilteringSettings
	{
		FilteringSettings() = default;
		FilteringSettings(RenderQueueRange range) {}
	};

}