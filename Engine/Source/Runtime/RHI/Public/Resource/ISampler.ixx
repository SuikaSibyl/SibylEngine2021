export module RHI.ISampler;

namespace SIByL
{
	namespace RHI
	{
		export struct SamplerDesc
		{

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