#pragma once


namespace SIByL
{
	class InspectorPanel
	{
	public:
		enum State
		{
			EntityComponents,
			MaterialEditor,
		};

	public:
		InspectorPanel();
		void OnImGuiRender();


	};
} 