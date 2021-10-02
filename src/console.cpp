////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Fractal
// (C)2018-21 Liam Kirton <liam@int3.ws>
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <sstream>
#include <string>

#include <windows.h>

#include "console.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string fill_console_line() {
    CONSOLE_SCREEN_BUFFER_INFO buffer_info{ 0 };
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &buffer_info);

    std::stringstream s;
    for (auto i = buffer_info.dwCursorPosition.X; i <= buffer_info.srWindow.Right - buffer_info.srWindow.Left; ++i) {
        s << " ";
    }
    return s.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
