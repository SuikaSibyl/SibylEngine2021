#include <iostream>

#include <SIByL.h>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

class ExampleLayer :public SIByL::Layer
{
public:
	ExampleLayer()
		:Layer("Example")
	{

	}

	void OnUpdate() override
	{
		//SIByL_APP_INFO("ExampleLayer::Update");
		if (SIByL::Input::IsKeyPressed(SIByL_KEY_TAB))
			SIByL_APP_TRACE("Tab key is pressed!");
	}

	void OnEvent(SIByL::Event& event) override
	{
		//SIByL_APP_TRACE("{0}", event);
	}

	void OnDraw() override
	{
		SIByL_APP_TRACE("Example Draw");
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