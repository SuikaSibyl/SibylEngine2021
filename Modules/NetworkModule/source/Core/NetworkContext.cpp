#include "NetworkContext.h"

#include<conio.h>
#include<stdlib.h>
#include <ws2tcpip.h>
#include <WinSock2.h>
#include <iostream>
#include <string>

#pragma comment(lib,"ws2_32.lib")
#pragma warning(disable:4996)
namespace SIByLNetwork
{
	IP NetworkContext::localIP;

	void NetworkContext::Init()
	{
		// Init WinSock
		WSADATA wsa_Data;
		int wsa_ReturnCode = WSAStartup(0x101, &wsa_Data);

		// Get the local hostname
		char szHostName[255];
		gethostname(szHostName, 255);
		struct hostent* host_entry;
		host_entry = gethostbyname(szHostName);
		char* szLocalIP;
		szLocalIP = inet_ntoa(*(struct in_addr*)*host_entry->h_addr_list);

		std::string ipstr(szLocalIP);
		int seglast = -1;
		int segpos = -1;
		std::string segconent;
		
		for (int i = 0; i < 4; i++)
		{
			segpos = ipstr.find(".", seglast + 1);
			segconent = ipstr.substr(seglast + 1, segpos - seglast - 1);
			localIP.seg[i] = atoi(segconent.c_str());
			seglast = segpos;
		}

		WSACleanup();
	}

	IP NetworkContext::GetLocalIP()
	{
		return localIP;
	}
}