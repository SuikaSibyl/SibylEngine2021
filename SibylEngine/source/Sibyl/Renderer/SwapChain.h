#pragma once

class SwapChain
{
public:
	SwapChain(int width, int height)
		:m_Width(width), m_Height(height) {}

private:
	int m_Width, m_Height;
};