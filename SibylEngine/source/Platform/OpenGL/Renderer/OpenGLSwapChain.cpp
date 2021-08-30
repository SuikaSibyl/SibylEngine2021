#include "SIByLpch.h"
#include "OpenGLSwapChain.h"

#include "SIByLpch.h"
#include "glad/glad.h"
#include "Platform/OpenGL/Window/GLFWWindow.h"

namespace SIByL
{
    OpenGLSwapChain::OpenGLSwapChain()
        :SwapChain(GLFWWindow::Get()->GetWidth(), GLFWWindow::Get()->GetHeight())
    {

    }

    OpenGLSwapChain::OpenGLSwapChain(int width, int height)
        : SwapChain(width, height)
    {

    }

    void OpenGLSwapChain::BindRenderTarget()
    {

    }

    void OpenGLSwapChain::SetRenderTarget()
    {
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    void OpenGLSwapChain::PreparePresent()
    {

    }

    void OpenGLSwapChain::Present()
    {
        glfwSwapBuffers((GLFWwindow*)GLFWWindow::Get()->GetNativeWindow());
    }

    void OpenGLSwapChain::Reisze(uint32_t width, uint32_t height)
    {
        glViewport(0, 0, width, height);
    }
}