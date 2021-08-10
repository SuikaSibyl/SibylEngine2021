#include <iostream>

#include <SIByL.h>


class Sandbox :public SIByL::Application
{
public:
	Sandbox()
	{

	}

	~Sandbox()
	{

	}
};

SIByL::Application* SIByL::CreateApplication()
{
	SIByL_CORE_TRACE("Create Application");
	return new Sandbox();
}