#pragma once

#ifdef SIBYL_PLATFORM_WINDOWS

extern SIByL::Application* SIByL::CreateApplication();

int main(int argc, char** argv)
{
	auto app = SIByL::CreateApplication();
	app->Run();
	delete app;
}

#endif // SIBYL_PLATFORM_WINDOWS
