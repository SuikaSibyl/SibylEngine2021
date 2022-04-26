module;
#include <cstdint>
export module RHI.ISampler;

namespace SIByL
{
	namespace RHI
	{
		export enum struct FilterMode {
			LINEAR,
			NEAREST,
		};

		export enum struct MipmapMode {
			LINEAR,
			NEAREST,
		};

		export enum struct AddressMode {
			CLAMP_TO_EDGE
		};

		export enum struct Extension {
			NONE,
			MIN_POOLING,
		};

		export struct SamplerDesc
		{
			FilterMode magFilter = FilterMode::LINEAR;
			FilterMode minFilter = FilterMode::LINEAR;
			MipmapMode mipmapMode = MipmapMode::LINEAR;
			AddressMode clampModeU = AddressMode::CLAMP_TO_EDGE;
			AddressMode clampModeV = AddressMode::CLAMP_TO_EDGE;
			AddressMode clampModeW = AddressMode::CLAMP_TO_EDGE;
			uint32_t minLod = 0;
			uint32_t maxLod = 0;
			Extension extension = Extension::NONE;
		};

		// Samplers will apply filtering and transformations to compute the final color that is retrieved.
		// These filters are helpful to deal with problems like oversampling.
		export class ISampler
		{
		public:
			ISampler() = default;
			virtual ~ISampler() = default;
		};
	}
}