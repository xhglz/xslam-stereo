#pragma once

#include "camera/CamBase.h"

// 适用场景：假设我们用了一个自定义library，为避免跟用到的其他库重名，
// 但又不得不开放public接口，
// 就可以应用到 generate_export_header 
// 导出 library export 宏定义，将非 public 接口隐藏，
// 将 public 接口设置为可见（<base_name>_EXPORT）。
class XPRIMER_EXPORT CamFisheye : public CamBase {
  public:
    CamFisheye();
    ~CamFisheye() = default;

    float k1_, k2_, k3_, k4_, k5_, k6_, p1_, p2_;

    std::string ClassName() const override;
    bool SaveFile(const std::string &filename) const override;
    bool LoadFile(const std::string &filename) override;
};
