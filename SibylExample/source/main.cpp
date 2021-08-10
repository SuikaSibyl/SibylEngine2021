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
	return new Sandbox();
}