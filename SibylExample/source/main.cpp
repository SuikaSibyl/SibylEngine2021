#include <iostream>

#include <SIByL.h>

class ExampleLayer :public SIByL::Layer
{
public:
	ExampleLayer()
		:Layer("Example")
	{

	}

	void OnUpdate() override
	{
		SIByL_APP_INFO("ExampleLayer::Update");
	}

	void OnEvent(SIByL::Event& event) override
	{
		SIByL_APP_TRACE("{0}", event);
	}
};

class Sandbox :public SIByL::Application
{
public:
	Sandbox()
	{
		PushLayer(new ExampleLayer());
		PushOverlay(SIByL::ImGuiLayer::Create());
	}

	~Sandbox()
	{

	}
};

SIByL::Application* SIByL::CreateApplication()
{
	SIByL_APP_TRACE("Create Application");
	return new Sandbox();
}