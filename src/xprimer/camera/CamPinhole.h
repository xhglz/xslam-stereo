#pragma once

#include "camera/CamBase.h"

class CamPinhole : public CamBase {
public:
  CamPinhole() = default;
  ~CamPinhole() = default;

  std::string ClassName() const override;
  bool SaveFile(const std::string &filename) const override;
  bool LoadFile(const std::string &filename) override;
};
