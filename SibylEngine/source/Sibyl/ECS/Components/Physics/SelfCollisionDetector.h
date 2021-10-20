#pragma once

namespace SIByL
{
	struct MeshFilterComponent;
	struct SelfCollisionDetectorComponent
	{
		SelfCollisionDetectorComponent() = default;

		void Init(MeshFilterComponent& mf);

		float ProcessTime;
	};
}