module;
#include <string_view>
module Core.Layer;

import Core.Event;

namespace SIByL::Core
{
	void ILayer::onAwake() {}
	void ILayer::onShutdown() {}
	void ILayer::onUpdate() {}
	void ILayer::onAttach() {}
	void ILayer::onDetach() {}
	void ILayer::onDraw() {}
	void ILayer::onEvent(Event& event) {}
}