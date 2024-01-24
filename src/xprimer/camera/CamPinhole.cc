#include <iostream>
#include "camera/CamPinhole.h"

std::string CamPinhole::ClassName() const {
    return "CamPinhole";
}

bool CamPinhole::SaveFile(const std::string &filename) const {
    Json::Value obj;
    SaveBaseCameraParameter(obj, *this);
    return JsonToFile(obj, filename);
}

bool CamPinhole::LoadFile(const std::string &filename) {
    Json::Value obj;

    if (JsonFromFile(obj, filename)) {
        return LoadBaseCameraParameter(obj, *this);
    }
    return false;
}
