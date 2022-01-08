import Core.Application;

extern auto SE_CREATE_APP() noexcept -> SIByL::IApplication*;

int main(int argc, char** argv)
{
	SIByL::IApplication* app = SE_CREATE_APP();

	app->awake();
	app->mainLoop();
	delete app; // app->shutdown();

	return 0;
}