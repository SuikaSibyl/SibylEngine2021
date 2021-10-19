#pragma once

namespace SIByL
{
    std::string& replace_all(std::string& str, const std::string& old_value, const std::string& new_value)
    {
        while (true) {
            std::string::size_type pos(0);
            if ((pos = str.find(old_value)) != std::string::npos)
                str.replace(pos, old_value.length(), new_value);
            else   break;
        }
        return   str;
    }
}