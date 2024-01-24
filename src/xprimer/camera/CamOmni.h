#pragma once

#include "camera/CamBase.h"

class XRPRIMER_EXPORT CamOmni : public CamBase {
public:
  CamOmni();
  ~CamOmni() = default;

  float k1_, k2_, k3_, k4_, k5_, k6_, p1_, p2_, xi_;
  Eigen::Vector4f D_;

  std::string ClassName() const override;
  bool SaveFile(const std::string &filename) const override;
  bool LoadFile(const std::string &filename) override;
};
