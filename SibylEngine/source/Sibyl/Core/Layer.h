#pragma once

#include "Sibyl/Core/Core.h"
#include "Sibyl/Events/Event.h"

namespace SIByL
{
	class SIByL_API Layer
	{
	public:
		Layer(const std::string& name = "Layer");
		virtual ~Layer();

		virtual void OnAttach() {}
		virtual void OnInitRenderer() {}
		virtual void OnDetach() {}
		virtual void OnUpdate() {}
		virtual void OnDraw() {}
		virtual void OnDrawImGui() {}
		virtual void OnEvent(Event& event) {}

		inline const std::string& GetName() const { return m_DebugName; }

	protected:
		std::string m_DebugName;
	};
}