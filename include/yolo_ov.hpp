#pragma once

#include "yolo.hpp"
#include <inference_engine.hpp>

class YOLO_OpenVINO : public CommonYOLO
{
public:
    YOLO_OpenVINO(std::string cfg_fpath);

    bool init(std::string ir_fpath, std::string device_type);

    void infer(cv::Mat raw_image, std::vector<DetectionObject> &detections, bool debug = false);

private:
    InferenceEngine::Core                mIeCore;
    InferenceEngine::ExecutableNetwork   mExecutableNetwork;
    InferenceEngine::CNNNetwork          mNetwork;

    std::vector<InferenceEngine::InferRequest::Ptr>     mInferRequests;
};
