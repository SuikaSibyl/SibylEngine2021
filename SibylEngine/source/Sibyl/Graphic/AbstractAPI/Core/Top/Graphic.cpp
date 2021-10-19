#include "SIByLpch.h"
#include "Graphic.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/DrawItem.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/FrameConstantsManager.h"


namespace SIByL
{
	Material* Graphic::CurrentMaterial = nullptr;
	Camera* Graphic::CurrentCamera = nullptr;
	FrameConstantsManager* Graphic::CurrentFrameConstantsManager = nullptr;

	void Graphic::SetRenderTarget(const std::string& key)
	{

	}

	void Graphic::SetRenderTarget(Ref<RenderTarget> key)
	{

	}

	void Graphic::DrawDrawItemNow(Ref<DrawItem> drawItem)
	{
		Graphic::CurrentCamera->OnDrawCall();
		Graphic::CurrentMaterial->OnDrawCall();
		Graphic::CurrentFrameConstantsManager->OnDrawCall();
		drawItem->OnDrawCall();
	}
}